import cv2
import cv2.aruco as aruco
import numpy as np
import time
import csv
import os
import sys
from datetime import datetime
import mediapipe as mp

# --- 設定 (Settings) ---
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

CORNER_IDS = [0, 1, 2, 3]
CORNER_INDEX_MAP = { 0: 2, 1: 3, 2: 0, 3: 1 }

# 保存設定
OUTPUT_DIR = "recordings"
CAPTURE_IMAGE_FILENAME = 'calligraphy_image.png'
CALIB_FILE_NAME = 'unpitsu_calibration.npz'
LENS_CALIB_FILE_NAME = 'top_camera_lens.npz'

# 座標系設定
COORD_LIMIT = 200.0
WARPED_SIZE = 800
GRID_SIZE = 8
CELL_SIZE_PX = WARPED_SIZE / GRID_SIZE
SAMPLING_INTERVAL = 0.05 

# Topカメラ用 色追跡設定
TOP_COLOR_LOWER = None
TOP_COLOR_UPPER = None

# Sideカメラ用 MediaPipe設定
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Sideカメラのみで使用
hands_side = mp.solutions.hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# ノイズ除去用カーネル (色追跡用)
KERNEL_OPEN = np.ones((5, 5), np.uint8)
KERNEL_CLOSE = np.ones((20, 20), np.uint8)

# --- グローバル変数 ---
is_recording_session = False
is_pen_down = False
is_area_locked = False
stroke_count = 0
last_record_time = 0
last_cell_id = -1

M_live = None
M_locked = None

# 筆圧 (Z軸 - MediaPipe由来)
Z_TOUCH_HEIGHT = -1
Z_PRESS_MAX_HEIGHT = -1
is_z_calibrated = False

# レンズ歪み
TOP_CAM_MTX = None
TOP_CAM_DIST = None
TOP_CAM_MAP_X = None
TOP_CAM_MAP_Y = None

# --- ヘルパー関数 ---

def select_camera_index(prompt_text):
    print(f"--- {prompt_text} のカメラを選択 ---")
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    if not available: return None
    
    for index in available:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened(): continue
        print(f"Checking Camera {index}...")
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.putText(frame, f"Index: {index} ({prompt_text}) - 'y':Use, 'n':Skip", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Camera Select", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                cv2.destroyAllWindows()
                cap.release()
                return index
            if key == ord('n'):
                break
        cap.release()
    cv2.destroyAllWindows()
    return None

def pick_color_range(event, x, y, flags, param):
    """マウスクリックでTopカメラのHSV色範囲を設定"""
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_hsv = param['frame_hsv']
        
        # クリックしたピクセルのHSV値を取得
        pixel = frame_hsv[y, x]
        
        # 許容範囲
        h_sens = 10 
        s_sens = 50 
        v_sens = 80 
        
        lower = np.array([max(0, pixel[0] - h_sens), max(50, pixel[1] - s_sens), max(50, pixel[2] - v_sens)])
        upper = np.array([min(180, pixel[0] + h_sens), 255, 255])
        
        global TOP_COLOR_LOWER, TOP_COLOR_UPPER
        TOP_COLOR_LOWER = lower
        TOP_COLOR_UPPER = upper
        
        print(f"[TOP] 色を設定しました: HSV {pixel}")

def setup_top_color_tracking(cap_top):
    """Topカメラのマーカー色を設定する"""
    print("\n--- カラーマーカー設定 (Topカメラのみ) ---")
    print("1. Topカメラウィンドウで、ペンのマーカーをクリックしてください。")
    print("2. 設定したら 'Esc' キーで完了します。")
    
    cv2.namedWindow("Top View (Setup)")
    
    while True:
        ret_top, frame_top = cap_top.read()
        if not ret_top: break
        
        if TOP_CAM_MAP_X is not None:
            frame_top = cv2.remap(frame_top, TOP_CAM_MAP_X, TOP_CAM_MAP_Y, cv2.INTER_LINEAR)
            
        hsv_top = cv2.cvtColor(frame_top, cv2.COLOR_BGR2HSV)
        
        cv2.setMouseCallback("Top View (Setup)", pick_color_range, {'frame_hsv': hsv_top})
        
        status_top = "OK" if TOP_COLOR_LOWER is not None else "Click Marker"
        
        if TOP_COLOR_LOWER is not None:
            mask = cv2.inRange(hsv_top, TOP_COLOR_LOWER, TOP_COLOR_UPPER)
            preview = cv2.bitwise_and(frame_top, frame_top, mask=mask)
            cv2.imshow("Top View (Setup)", preview)
        else:
            cv2.putText(frame_top, status_top, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Top View (Setup)", frame_top)
            
        if cv2.waitKey(1) & 0xFF == 27: # Esc
            if TOP_COLOR_LOWER is None:
                print("警告: 色が設定されていません。")
            else:
                break
                
    cv2.destroyAllWindows()
    print("--- 色設定完了 ---")

def track_color_blob(frame, lower, upper):
    """指定した色の重心を見つける"""
    if lower is None or upper is None: return None
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 50:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    return None

def get_marker_point(target_id, detected_ids, detected_corners):
    if detected_ids is not None:
        for i, marker_id in enumerate(detected_ids.flatten()):
            if marker_id == target_id:
                return detected_corners[i][0][CORNER_INDEX_MAP[target_id]].astype(int)
    return None

def load_lens_calibration(file_path, frame_size_wh):
    global TOP_CAM_MAP_X, TOP_CAM_MAP_Y
    if os.path.exists(file_path):
        try:
            with np.load(file_path) as data:
                mtx, dist = data['mtx'], data['dist']
                w, h = frame_size_wh
                new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                TOP_CAM_MAP_X, TOP_CAM_MAP_Y = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
                print("レンズ補正データを読み込みました。")
        except: pass

def convert_to_custom_coords(norm_x, norm_y):
    norm_x_01 = norm_x / WARPED_SIZE
    norm_y_01 = norm_y / WARPED_SIZE
    return (norm_x_01 - 1.0) * COORD_LIMIT, norm_y_01 * -COORD_LIMIT

def get_cell_id(norm_x, norm_y):
    cx = int(norm_x // CELL_SIZE_PX)
    cy = int(norm_y // CELL_SIZE_PX)
    return (max(0, min(cy, GRID_SIZE - 1)) * GRID_SIZE) + max(0, min(cx, GRID_SIZE - 1))

def record_data(event_type, timestamp, pressure, pen_pos_norm):
    global last_record_time, last_cell_id, stroke_count
    
    if event_type == 'move' and timestamp - last_record_time < SAMPLING_INTERVAL:
        return
    last_record_time = timestamp

    norm_x, norm_y = pen_pos_norm
    x, y = convert_to_custom_coords(norm_x, norm_y)
    cell_id = get_cell_id(norm_x, norm_y)
    
    drawing_data.append({
        'timestamp': timestamp, 'event_type': event_type, 'stroke_id': stroke_count,
        'x': f"{x:.2f}", 'y': f"{y:.2f}", 'pressure': f"{pressure:.4f}", 'cell_id': cell_id
    })
    last_cell_id = cell_id if event_type != 'up' else -1

def save_csv():
    if not drawing_data: return
    print("CSV保存中...")
    filename = f"unpitsu_hybrid_{int(time.time())}.csv"
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=drawing_data[0].keys())
        writer.writeheader()
        writer.writerows(drawing_data)
    print(f"保存完了: {filename}")

# --- メイン処理 ---
def main():
    global is_recording_session, is_pen_down, is_area_locked, stroke_count, M_locked, M_live
    global Z_TOUCH_HEIGHT, Z_PRESS_MAX_HEIGHT, is_z_calibrated

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 1. カメラ初期化
    top_idx = select_camera_index("Top-Down Camera (XY - Color)")
    if top_idx is None: return
    side_idx = select_camera_index("Side-View Camera (Z - MediaPipe)")
    if side_idx is None: return

    cap_top = cv2.VideoCapture(top_idx)
    cap_side = cv2.VideoCapture(side_idx)
    
    for cap in [cap_top, cap_side]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # レンズ補正読み込み
    load_lens_calibration(LENS_CALIB_FILE_NAME, (1280, 720))

    # 2. Topカメラの色設定 (SideはMediaPipeなので不要)
    setup_top_color_tracking(cap_top)

    # 録画用ライター
    writer_top = None
    writer_side = None
    
    print("\n--- メインループ開始 ---")
    print(" [l]: エリアロック (Top: ArUco 0-3)")
    print(" [z]: 筆圧0 (軽くタッチ) - Sideカメラの指位置で設定")
    print(" [m]: 筆圧8 (強く押す) - Sideカメラの指位置で設定")
    print(" [s]: 記録/録画 開始・停止")
    print(" [v]: 完成画像の撮影")
    print(" [q]: 終了")

    dst_pts = np.float32([[0, 0], [WARPED_SIZE, 0], [WARPED_SIZE, WARPED_SIZE], [0, WARPED_SIZE]])

    while True:
        ret_top, frame_top = cap_top.read()
        ret_side, frame_side = cap_side.read()
        if not ret_top or not ret_side: break

        # 歪み補正 (Top)
        if TOP_CAM_MAP_X is not None:
            frame_top = cv2.remap(frame_top, TOP_CAM_MAP_X, TOP_CAM_MAP_Y, cv2.INTER_LINEAR)

        current_time = time.time()

        # --- Top: ArUco エリア検出 ---
        corners, ids, _ = DETECTOR.detectMarkers(frame_top)
        if ids is not None: aruco.drawDetectedMarkers(frame_top, corners, ids)
        
        src_pts = [get_marker_point(id, ids, corners) for id in CORNER_IDS]
        if all(pt is not None for pt in src_pts):
            M_live = cv2.getPerspectiveTransform(np.float32(src_pts), dst_pts)
            cv2.polylines(frame_top, [np.float32(src_pts).astype(int)], True, (0, 255, 0), 2)
        
        M_use = M_locked if is_area_locked else M_live

        # --- Top: 色トラッキング (XY) ---
        pen_center_top = track_color_blob(frame_top, TOP_COLOR_LOWER, TOP_COLOR_UPPER)
        pen_pos_norm = None
        
        if pen_center_top:
            cv2.circle(frame_top, pen_center_top, 10, (0, 255, 255), -1)
            if M_use is not None:
                # 射影変換
                pt = np.array([[[pen_center_top[0], pen_center_top[1]]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(pt, M_use)
                pen_pos_norm = (warped[0][0][0], warped[0][0][1])

        # --- Side: MediaPipe (Z/筆圧) ---
        frame_side_rgb = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)
        results_side = hands_side.process(frame_side_rgb)
        
        current_pressure = 0
        is_touching = False
        pen_y_side = -1
        
        if results_side.multi_hand_landmarks:
            # 最初の手を使用
            hand_landmarks = results_side.multi_hand_landmarks[0]
            # 描画
            mp_drawing.draw_landmarks(frame_side, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 親指の先端 (THUMB_TIP) を使用
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            h, w, _ = frame_side.shape
            pen_y_side = int(thumb_tip.y * h)
            px_side = int(thumb_tip.x * w)
            
            # 座標表示
            cv2.circle(frame_side, (px_side, pen_y_side), 8, (255, 0, 0), -1)
            cv2.putText(frame_side, f"Y={pen_y_side}", (px_side+10, pen_y_side), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            # 筆圧計算
            if is_z_calibrated:
                if pen_y_side < Z_TOUCH_HEIGHT:
                    current_pressure = 0
                    is_touching = False
                else:
                    is_touching = True
                    # 0-8にマッピング
                    rng = max(1, Z_PRESS_MAX_HEIGHT - Z_TOUCH_HEIGHT)
                    val = (pen_y_side - Z_TOUCH_HEIGHT) / rng
                    current_pressure = min(8, max(0, int(val * 8)))

        # --- 記録ロジック ---
        if is_recording_session:
            # 録画保存
            if writer_top: writer_top.write(frame_top)
            if writer_side: writer_side.write(frame_side)
            
            # CSV記録 (Top座標があり、かつSideで手が検出されている場合)
            if pen_pos_norm is not None:
                if is_touching and not is_pen_down:
                    is_pen_down = True
                    stroke_count += 1
                    record_data('down', current_time, current_pressure, pen_pos_norm)
                elif is_touching and is_pen_down:
                    record_data('move', current_time, current_pressure, pen_pos_norm)
                elif not is_touching and is_pen_down:
                    is_pen_down = False
                    record_data('up', current_time, 0, pen_pos_norm)

            cv2.putText(frame_top, "REC", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # --- UI表示 ---
        if is_z_calibrated:
            cv2.line(frame_side, (0, Z_TOUCH_HEIGHT), (1280, Z_TOUCH_HEIGHT), (0,255,0), 2)
            cv2.line(frame_side, (0, Z_PRESS_MAX_HEIGHT), (1280, Z_PRESS_MAX_HEIGHT), (0,0,255), 2)

        cv2.imshow("Top View (Color Track)", frame_top)
        cv2.imshow("Side View (MediaPipe)", frame_side)

        # --- キー操作 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        if key == ord('l'):
            if M_live is not None:
                M_locked = M_live
                is_area_locked = True
                print("Area Locked.")
            else:
                is_area_locked = False
                M_locked = None
                print("Area Unlocked.")

        # z/m キー: 現在の MediaPipe Y座標 をセット
        if key == ord('z') and pen_y_side != -1:
            Z_TOUCH_HEIGHT = pen_y_side
            print(f"筆圧0設定 (MediaPipe): Y={Z_TOUCH_HEIGHT}")
            if Z_PRESS_MAX_HEIGHT != -1: is_z_calibrated = True

        if key == ord('m') and pen_y_side != -1:
            Z_PRESS_MAX_HEIGHT = pen_y_side
            print(f"筆圧8設定 (MediaPipe): Y={Z_PRESS_MAX_HEIGHT}")
            if Z_TOUCH_HEIGHT != -1: is_z_calibrated = True

        if key == ord('s'):
            if not is_recording_session:
                if is_area_locked and is_z_calibrated:
                    # 録画開始
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    session_path = os.path.join(OUTPUT_DIR, f"session_{ts}")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer_top = cv2.VideoWriter(f"{session_path}_top.mp4", fourcc, 20.0, (1280, 720))
                    writer_side = cv2.VideoWriter(f"{session_path}_side.mp4", fourcc, 20.0, (1280, 720))
                    
                    is_recording_session = True
                    print("記録開始")
                else:
                    print("エラー: エリアロックと筆圧設定(z/m)を完了してください。")
            else:
                is_recording_session = False
                if writer_top: writer_top.release()
                if writer_side: writer_side.release()
                print("記録停止 -> CSV保存")
                save_csv()
                print("★ [v]キーで完成画像を保存してください。")

        if key == ord('v'): # キャプチャ
            if M_locked is not None:
                warped = cv2.warpPerspective(frame_top, M_locked, (WARPED_SIZE, WARPED_SIZE))
                cv2.imwrite(CAPTURE_IMAGE_FILENAME, warped)
                print(f"完成画像を保存: {CAPTURE_IMAGE_FILENAME}")

    cap_top.release()
    cap_side.release()
    hands_side.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()