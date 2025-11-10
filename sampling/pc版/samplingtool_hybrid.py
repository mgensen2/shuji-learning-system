import cv2
import cv2.aruco as aruco
import numpy as np
import time
import csv
import os
import sys
import mediapipe as mp

# --- 1. 設定項目 ---
# ArUcoマーカーの設定 (範囲検出用)
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR_PARAMS.polygonalApproxAccuracyRate = 0.05
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

CORNER_IDS = [0, 1, 2, 3] # [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
CORNER_INDEX_MAP = { 0: 2, 1: 3, 2: 0, 3: 1 }

BRUSH_MARKER_ID = 4

# エリアロック/オフセット設定の保存ファイル名
CALIB_FILE_NAME = 'unpitsu_calibration.npz'

# Topカメラのレンズ歪み補正ファイル
LENS_CALIB_FILE_NAME = 'top_camera_lens.npz' 

# MediaPipe Hands の初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_top = mp.solutions.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_side = mp.solutions.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 座標系の設定
COORD_LIMIT = 200.0
GRID_SIZE = 8
SAMPLING_INTERVAL = 0.05
WARPED_SIZE = 800
CELL_SIZE_PX = WARPED_SIZE / GRID_SIZE

# --- 2. データ格納リスト ---
drawing_data = []     
cell_transitions = [] 

# --- 3. 状態変数 ---
is_recording_session = False
is_pen_down = False         
is_manually_paused = False  
is_area_locked = False        # lキーでエリアロックを管理
stroke_count = 0
last_cell_id = -1
last_record_time = 0
last_pen_pos_norm = None 
M_live = None   
M_locked = None 
TARGET_HAND = "Any"         

# 筆圧キャリブレーション用
Y_TOUCH_THRESHOLD = -1     
Y_MAX_PRESS_THRESHOLD = -1 

# 筆オフセット用
BRUSH_TIP_OFFSET_LOCAL = None 
is_brush_calibrated = False   

# レンズ歪み補正用
TOP_CAM_MTX = None
TOP_CAM_DIST = None
TOP_CAM_MAP_X = None
TOP_CAM_MAP_Y = None


# --- 4. ヘルパー関数 ---
def select_camera_index(prompt_text):
    """カメラを選択させる（'n'でスキップ可能にする）"""
    print(f"--- {prompt_text} のカメラを選択 ---")
    available_indices = []
    for i in range(10):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            available_indices.append(i)
            cap_test.release()
    if not available_indices: return None
    print(f"利用可能なインデックス: {available_indices}")

    selected_index = None
    for index in available_indices:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened(): continue
        print(f"--- カメラ {index} をテスト中 ---")
        while True:
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 80), (255,255,255), -1)
            cv2.putText(frame, f"Index: {index} ({prompt_text})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, "Use this? (y/n)", (10, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Camera Selection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                print(f"-> カメラ {index} を選択しました。")
                selected_index = index
                break
            if key == ord('n'):
                print("-> スキップします。")
                break
        cap.release()
        if selected_index is not None:
            break
    cv2.destroyAllWindows()
    return selected_index

def select_target_hand():
    """ユーザーに追跡する手を選択させる"""
    print("\n--- 追跡する手を選択 ---")
    print(" [l] 左手のみ (Left)")
    print(" [r] 右手のみ (Right)")
    print(" [a] どちらでも (Any) - デフォルト")
    
    while True:
        key_in = input("使用する手を選んでください (l/r/a): ").strip().lower()
        if key_in == 'l':
            print("-> 左手のみを追跡します。")
            return "Left"
        elif key_in == 'r':
            print("-> 右手のみを追跡します。")
            return "Right"
        elif key_in == 'a' or key_in == '': 
            print("-> 検出された最初の手を追跡します (Any)。")
            return "Any"
        else:
            print("無効な入力です。'l', 'r', 'a' のいずれかを入力してください。")

def get_target_hand_landmarks(results, target_hand_label):
    """
    検出結果から、指定された手（"Left", "Right", "Any"）のランドマークを取得する
    """
    if not results.multi_hand_landmarks:
        return None 

    if target_hand_label == "Any":
        return results.multi_hand_landmarks[0] 

    if not results.multi_handedness:
        return None 

    for i, handedness in enumerate(results.multi_handedness):
        label = handedness.classification[0].label
        if label == target_hand_label:
            return results.multi_hand_landmarks[i] 
    
    return None 

def calibrate_pressure_range(cap_side, target_hand_label): 
    """Side-Viewカメラの筆圧(Z軸)キャリブレーションを2段階で行う"""
    global Y_TOUCH_THRESHOLD, Y_MAX_PRESS_THRESHOLD
    
    # --- (1/2) 筆圧 0 (タッチ) のキャリブレーション ---
    print("--- 筆圧 (Z軸) キャリブレーション (1/2) ---")
    print("筆（人差し指の先端）を紙に「軽く触れさせた」状態で 'c' キーを押してください。 (筆圧 0)")
    current_y = -1
    while True:
        ret, frame = cap_side.read()
        if not ret: return False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_side.process(frame_rgb)
        h, w, _ = frame.shape
        hand_landmarks = get_target_hand_landmarks(results, target_hand_label) 
        if hand_landmarks:
            landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] 
            current_y = int(landmark.y * h)
            current_x = int(landmark.x * w)
            cv2.putText(frame, f"Detected Y: {current_y}", (current_x + 10, current_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(frame, (current_x, current_y), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (0, 0), (w, 40), (0,0,0), -1)
        cv2.putText(frame, "Touch pen to paper (Pressure 0), then press 'c'", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Side Camera Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if current_y != -1:
                Y_TOUCH_THRESHOLD = current_y
                print(f"キャリブレーション (1/2) 完了。タッチしきい値(Y座標) = {Y_TOUCH_THRESHOLD}")
                break 
            else:
                print("エラー: 手が検出されていません。 'c' を押す前に手を映してください。")
        if key == ord('q'):
            print("キャリブレーションがキャンセルされました。")
            cv2.destroyAllWindows()
            return False
            
    # --- (2/2) 筆圧 8 (最大) のキャリブレーション ---
    print("\n--- 筆圧 (Z軸) キャリブレーション (2/2) ---")
    print("筆（人差し指の先端）を紙に「強く押し付けた」状態で 'm' キーを押してください。 (筆圧 8)")
    current_y = -1
    while True:
        ret, frame = cap_side.read()
        if not ret: return False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_side.process(frame_rgb)
        h, w, _ = frame.shape
        hand_landmarks = get_target_hand_landmarks(results, target_hand_label) 
        if hand_landmarks:
            landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] 
            current_y = int(landmark.y * h)
            current_x = int(landmark.x * w)
            cv2.putText(frame, f"Detected Y: {current_y}", (current_x + 10, current_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(frame, (current_x, current_y), 5, (0, 0, 255), -1)
        cv2.line(frame, (0, Y_TOUCH_THRESHOLD), (w, Y_TOUCH_THRESHOLD), (0, 255, 255), 2)
        cv2.putText(frame, f"TOUCH_Y_LEVEL (Pressure 0): {Y_TOUCH_THRESHOLD}", (10, Y_TOUCH_THRESHOLD - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(frame, (0, 0), (w, 40), (0,0,0), -1)
        cv2.putText(frame, "Press firmly (Pressure 8), then press 'm'", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Side Camera Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            if current_y != -1:
                if current_y > Y_TOUCH_THRESHOLD:
                    Y_MAX_PRESS_THRESHOLD = current_y
                    print(f"キャリブレーション (2/2) 完了。最大筆圧しきい値(Y座標) = {Y_MAX_PRESS_THRESHOLD}")
                    cv2.destroyAllWindows()
                    return True 
                else:
                    print(f"エラー: 最大筆圧({current_y})は、タッチしきい値({Y_TOUCH_THRESHOLD})より大きくする必要があります。")
            else:
                print("エラー: 手が検出されていません。 'm' を押す前に手を映してください。")
        if key == ord('q'):
            print("キャリブレーションがキャンセルされました。")
            cv2.destroyAllWindows()
            return False
            
    return False


def get_marker_point(target_id, detected_ids, detected_corners):
    """
    検出されたマーカーリストから、指定されたIDの「基準点」ピクセル座標を取得する
    """
    if detected_ids is not None:
        for i, marker_id in enumerate(detected_ids.flatten()): # ★ .flatten() を追加
            if marker_id == target_id:
                if target_id in CORNER_INDEX_MAP:
                    corners = detected_corners[i][0]
                    corner_index = CORNER_INDEX_MAP[target_id]
                    point = corners[corner_index]
                    return point.astype(int)
    return None

def convert_to_custom_coords(norm_x, norm_y):
    norm_x_01 = norm_x / WARPED_SIZE
    norm_y_01 = norm_y / WARPED_SIZE
    converted_x = (norm_x_01 - 1.0) * COORD_LIMIT
    converted_y = norm_y_01 * -COORD_LIMIT
    return converted_x, converted_y

def get_cell_id(norm_x, norm_y):
    cell_x = int(norm_x // CELL_SIZE_PX)
    cell_y = int(norm_y // CELL_SIZE_PX)
    cell_x = max(0, min(cell_x, GRID_SIZE - 1))
    cell_y = max(0, min(cell_y, GRID_SIZE - 1))
    return (cell_y * GRID_SIZE) + cell_x

def save_all_data():
    """プログラム終了時にデータをCSVに保存"""
    print(f"Saving {len(drawing_data)} data points...")
    if drawing_data:
        headers = drawing_data[0].keys()
        with open('unpitsu_data_full.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(drawing_data)
        print("Saved unpitsu_data_full.csv")
    if cell_transitions:
        headers = cell_transitions[0].keys()
        with open('unpitsu_data_transitions.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(cell_transitions)
        print("Saved unpitsu_data_transitions.csv")

def save_calibration_data():
    """現在のキャリブレーションデータを.npzに保存する"""
    if not is_area_locked:
        print("エラー: エリアがロックされていません。 [l]キーでロックしてください。")
        return
    if not is_brush_calibrated:
        print("エラー: 筆オフセットが調整されていません。 [o]キーで調整してください。")
        return
    
    try:
        np.savez(CALIB_FILE_NAME,
                 M_locked=M_locked,
                 BRUSH_TIP_OFFSET_LOCAL=BRUSH_TIP_OFFSET_LOCAL
                )
        print(f"--- キャリブレーションデータを保存しました ({CALIB_FILE_NAME}) ---")
    except Exception as e:
        print(f"エラー: キャリブレーションデータの保存に失敗しました: {e}")

def load_calibration_data():
    """起動時にキャリブレーションデータを読み込む"""
    global M_locked, BRUSH_TIP_OFFSET_LOCAL, is_area_locked, is_brush_calibrated
    
    if os.path.exists(CALIB_FILE_NAME):
        try:
            with np.load(CALIB_FILE_NAME) as data:
                M_locked = data['M_locked']
                # .npzから読み込んだ配列が0次元配列(array(tuple))の場合、正しくタプルに変換する
                offset_data = data['BRUSH_TIP_OFFSET_LOCAL']
                if offset_data.shape == ():
                    BRUSH_TIP_OFFSET_LOCAL = tuple(offset_data.item())
                else:
                    BRUSH_TIP_OFFSET_LOCAL = tuple(offset_data)
                
                is_area_locked = True
                is_brush_calibrated = True
                print(f"--- キャリブレーションデータを読み込みました ({CALIB_FILE_NAME}) ---")
                print("  エリアロックと筆オフセットが復元されました。")
        except Exception as e:
            print(f"エラー: キャリブレーションデータの読み込みに失敗しました: {e}")
            M_locked = None
            BRUSH_TIP_OFFSET_LOCAL = None
            is_area_locked = False
            is_brush_calibrated = False
    else:
        print(f"--- キャリブレーションファイルが見つかりません ({CALIB_FILE_NAME}) ---")

def load_lens_calibration(file_path, frame_size_wh):
    """Top-Downカメラのレンズ歪み補正データを.npzから読み込む"""
    global TOP_CAM_MTX, TOP_CAM_DIST, TOP_CAM_MAP_X, TOP_CAM_MAP_Y
    
    if os.path.exists(file_path):
        try:
            with np.load(file_path) as data:
                if 'mtx' not in data or 'dist' not in data:
                    print(f"エラー: {file_path} に 'mtx' または 'dist' が含まれていません。")
                    return
                
                mtx = data['mtx']
                dist = data['dist']
                w, h = frame_size_wh
                
                # 最適なカメラ行列を取得
                new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                
                # 高速化のため、remap用のマップを計算
                TOP_CAM_MAP_X, TOP_CAM_MAP_Y = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
                
                TOP_CAM_MTX = mtx # (参考用に保持)
                TOP_CAM_DIST = dist # (参考用に保持)
                
                print(f"--- レンズ歪み補正データを読み込みました ({file_path}) ---")
        except Exception as e:
            print(f"エラー: レンズ歪み補正データの読み込みに失敗しました: {e}")
    else:
        print(f"--- レンズ歪み補正ファイルが見つかりません ({file_path}) ---")
        print("    歪み補正なしで続行します。")

def record_data(event_type, timestamp, pressure, pen_pos_norm):
    """データを整形してリストに追加する"""
    global last_cell_id, last_record_time, stroke_count

    if event_type == 'move':
        if timestamp - last_record_time < SAMPLING_INTERVAL:
            return # 間引く
    last_record_time = timestamp

    (norm_x, norm_y) = pen_pos_norm
    x, y = convert_to_custom_coords(norm_x, norm_y)
    cell_id = get_cell_id(norm_x, norm_y)
    
    drawing_data.append({
        'timestamp': timestamp, 'event_type': event_type, 'stroke_id': stroke_count,
        'x': f"{x:.2f}", 'y': f"{y:.2f}", 'pressure': f"{pressure:.4f}", 'cell_id': cell_id
    })
    
    if event_type != 'up' and last_cell_id != -1 and cell_id != last_cell_id:
        curr_x, curr_y = cell_id % GRID_SIZE, cell_id // GRID_SIZE
        prev_x, prev_y = last_cell_id % GRID_SIZE, last_cell_id // GRID_SIZE
        if abs(curr_x - prev_x) + abs(curr_y - prev_y) == 1:
            cell_transitions.append({
                'timestamp': timestamp, 'stroke_id': stroke_count,
                'from_cell': last_cell_id, 'to_cell': cell_id
            })
            
    last_cell_id = cell_id if event_type != 'up' else -1


# --- 5. メイン処理 ---
# 5.1. カメラの選択
top_cam_index = select_camera_index("Top-Down (X/Y) Camera")
if top_cam_index is None: sys.exit("Top-Downカメラが選択されませんでした。")
cap_top = cv2.VideoCapture(top_cam_index)
cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# ★★★ Topカメラの解像度をここで取得 ★★★
top_w = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
top_h = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Topカメラ解像度: {top_w} x {top_h}")

side_cam_index = select_camera_index("Side-View (Z/Pressure) Camera")
if side_cam_index is None: sys.exit("Side-Viewカメラが選択されませんでした。")
cap_side = cv2.VideoCapture(side_cam_index)
cap_side.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_side.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Sideカメラ解像度: {cap_side.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# 5.1b. 追跡する手の選択
TARGET_HAND = select_target_hand()

# 5.1c. NPZキャリブレーションデータの自動読み込み (エリアロック/オフセット)
load_calibration_data()

# 5.1d. NPZレンズ歪み補正データの自動読み込み (新規)
load_lens_calibration(LENS_CALIB_FILE_NAME, (top_w, top_h))

# 5.2. Sideカメラのキャリブレーション
if not calibrate_pressure_range(cap_side, TARGET_HAND):
    sys.exit("キャリブレーションがキャンセルされました。")

# 5.3. メインループの準備
dst_pts = np.float32([[0, 0], [WARPED_SIZE, 0], [WARPED_SIZE, WARPED_SIZE], [0, WARPED_SIZE]])
print("\n--- トラッキング開始 ---")
print(" [k] キー: 現在のエリアロックと筆オフセットを .npz に保存")
print(" [l] キー: 記録エリアのロック / アンロック")
print(" [o] キー: 筆オフセット調整 (エリアロック後に実行)")
print(" [s] キー: 記録セッションの開始/停止 (エリアロック＆オフセット調整後に実行)")
print(" [p] キー: 記録の手動一時停止/再開 (セッション中のみ)")
print(" [q] キー: 終了してCSV保存")
print("\n★ 推奨手順: ( [l] -> [o] -> [k] ) -> [s] で記録")
print("★ 次回起動時: [k] で保存した .npz が自動で読み込まれます。")

while True:
    # 5.4. 両方のカメラからフレームを取得
    ret_top, frame_top = cap_top.read()
    ret_side, frame_side = cap_side.read()
    if not ret_top or not ret_side:
        print("エラー: カメラフレームを読み込めません")
        break
    
    # ★★★ 新規追加: Topカメラの歪み補正 (remap) ★★★
    if TOP_CAM_MAP_X is not None and TOP_CAM_MAP_Y is not None:
        frame_top = cv2.remap(frame_top, TOP_CAM_MAP_X, TOP_CAM_MAP_Y, cv2.INTER_LINEAR)
    
    current_time = time.time()
    
    # 5.5. Topカメラの処理 (ArUco 範囲検出)
    (corners_top, ids_top, _) = DETECTOR.detectMarkers(frame_top)
    if ids_top is not None:
        aruco.drawDetectedMarkers(frame_top, corners_top, ids_top)

    src_pts = [get_marker_point(id, ids_top, corners_top) for id in CORNER_IDS]
    
    if all(pt is not None for pt in src_pts):
        src_pts_np = np.float32(src_pts)
        M_live = cv2.getPerspectiveTransform(src_pts_np, dst_pts) 
        cv2.polylines(frame_top, [src_pts_np.astype(int)], True, (0, 255, 0), 2)
    else:
        M_live = None # 4隅が揃っていなければ M_live は None
    
    # 座標系行列の選択ロジック
    M_to_use = M_locked if is_area_locked else M_live

    # 5.5b. 筆マーカーの検出 (新規)
    brush_marker_corners_pixel = None
    brush_marker_center_pixel = None
    brush_marker_xaxis_pixel = None
    brush_marker_yaxis_pixel = None

    if ids_top is not None:
        for i, marker_id in enumerate(ids_top.flatten()):
            if marker_id == BRUSH_MARKER_ID:
                brush_marker_corners_pixel = corners_top[i][0].astype(int)
                brush_marker_center_pixel = brush_marker_corners_pixel.mean(axis=0).astype(int)
                # マーカーのローカルX軸 (corner[1] - corner[0])
                brush_marker_xaxis_pixel = brush_marker_corners_pixel[1] - brush_marker_corners_pixel[0]
                # マーカーのローカルY軸 (corner[3] - corner[0])
                brush_marker_yaxis_pixel = brush_marker_corners_pixel[3] - brush_marker_corners_pixel[0]
                # マーカーをシアン(Cyan)で囲む
                cv2.polylines(frame_top, [brush_marker_corners_pixel], True, (255, 255, 0), 2)
                break

    # 5.6. Sideカメラの処理 (筆圧Z軸の検出)
    is_touching_now = False 
    current_pressure_level = 0 
    frame_side_rgb = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)
    results_side = hands_side.process(frame_side_rgb)
    hand_landmarks_side = get_target_hand_landmarks(results_side, TARGET_HAND) 
    
    pen_y_side = -1 
    if hand_landmarks_side: 
        h_side, w_side, _ = frame_side.shape
        landmark = hand_landmarks_side.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] 
        pen_y_side = int(landmark.y * h_side)
        
        if pen_y_side < Y_TOUCH_THRESHOLD:
            is_touching_now = False
            current_pressure_level = 0
        elif pen_y_side >= Y_MAX_PRESS_THRESHOLD:
            is_touching_now = True
            current_pressure_level = 8
        else: # Y_TOUCH_THRESHOLD <= pen_y_side < Y_MAX_PRESS_THRESHOLD
            is_touching_now = True
            touch_range = float(Y_MAX_PRESS_THRESHOLD - Y_TOUCH_THRESHOLD)
            current_depth = float(pen_y_side - Y_TOUCH_THRESHOLD)
            if touch_range > 0: 
                normalized_pressure = current_depth / touch_range
                current_pressure_level = int(round(normalized_pressure * 8))
            else:
                current_pressure_level = 0 
        
    cv2.line(frame_side, (0, Y_TOUCH_THRESHOLD), (w_side, Y_TOUCH_THRESHOLD), (0, 255, 255), 2)
    cv2.putText(frame_side, f"P=0 Y:{Y_TOUCH_THRESHOLD}", (10, Y_TOUCH_THRESHOLD - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(frame_side, (0, Y_MAX_PRESS_THRESHOLD), (w_side, Y_MAX_PRESS_THRESHOLD), (0, 165, 255), 2)
    cv2.putText(frame_side, f"P=8 Y:{Y_MAX_PRESS_THRESHOLD}", (10, Y_MAX_PRESS_THRESHOLD - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    if pen_y_side != -1 and hand_landmarks_side: 
        px_side = int(hand_landmarks_side.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w_side) 
        color = (0, 0, 255) if is_touching_now else (0, 255, 0)
        cv2.circle(frame_side, (px_side, pen_y_side), 8, color, -1)
        cv2.putText(frame_side, f"Pressure: {current_pressure_level}", (px_side + 10, pen_y_side), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 5.7. Topカメラの処理 (X/Y座標検出) - ハイブリッド方式
    pen_pos_norm = None
    if M_to_use is not None:
        
        # --- 1. [優先] 筆マーカー(ID=50)から計算 ---
        # オフセット調整が完了しており、かつ筆マーカーが見えている場合
        if is_brush_calibrated and brush_marker_center_pixel is not None:
            local_x, local_y = BRUSH_TIP_OFFSET_LOCAL
            
            # ローカルオフセットを現在のマーカーの向き（X軸, Y軸）に基づいて回転・適用
            tip_vector_pixel = (brush_marker_xaxis_pixel * local_x) + (brush_marker_yaxis_pixel * local_y)
            pen_tip_pixel = (brush_marker_center_pixel + tip_vector_pixel).astype(int)
            
            # ピクセル座標を射影変換
            pen_pixel_np = np.float32([[pen_tip_pixel]])
            pen_pos_norm_raw = cv2.perspectiveTransform(pen_pixel_np, M_to_use)
            pen_pos_norm = (pen_pos_norm_raw[0][0][0], pen_pos_norm_raw[0][0][1])
            
            # ★ 計算された筆の先端を「青」で描画
            cv2.circle(frame_top, tuple(pen_tip_pixel), 8, (255, 100, 0), -1)

        # --- 2. [予備] 指の先端で追跡 ---
        # 筆マーカーで計算できなかった場合
        if pen_pos_norm is None:
            frame_top_rgb = cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB)
            results_top = hands_top.process(frame_top_rgb)
            hand_landmarks_top = get_target_hand_landmarks(results_top, TARGET_HAND) 
            
            if hand_landmarks_top: 
                h_top, w_top, _ = frame_top.shape
                landmark_top = hand_landmarks_top.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] 
                pen_center_pixel = (int(landmark_top.x * w_top), int(landmark_top.y * h_top))
                
                pen_pixel_np = np.float32([[pen_center_pixel]])
                pen_pos_norm_raw = cv2.perspectiveTransform(pen_pixel_np, M_to_use)
                pen_pos_norm = (pen_pos_norm_raw[0][0][0], pen_pos_norm_raw[0][0][1])
                
                # ★ 指の先端を「赤」で描画
                cv2.circle(frame_top, tuple(pen_center_pixel), 8, (0, 0, 255), -1)

        # 最終的に座標が取得できていれば、それを保持する
        if pen_pos_norm is not None:
            last_pen_pos_norm = pen_pos_norm

    # 5.8. 状態機械 (State Machine) による自動記録
    if is_recording_session and not is_manually_paused and last_pen_pos_norm is not None:
        if is_touching_now and not is_pen_down:
            is_pen_down = True
            stroke_count += 1
            print(f"Stroke {stroke_count} START (Down) - Pressure: {current_pressure_level}")
            record_data('down', current_time, current_pressure_level, last_pen_pos_norm)
        elif is_touching_now and is_pen_down:
            record_data('move', current_time, current_pressure_level, last_pen_pos_norm)
        elif not is_touching_now and is_pen_down:
            is_pen_down = False
            print(f"Stroke {stroke_count} END (Up)")
            record_data('up', current_time, 0, last_pen_pos_norm)

    # 5.9. 画面表示 (変更)
    status_text = "RECORDING" if is_recording_session else "STOPPED" # PAUSED -> STOPPED
    color = (0, 0, 255) if is_recording_session else (100, 100, 100) # PAUSEDの色をやめる

    if is_recording_session and is_manually_paused: 
        status_text = "MANUALLY PAUSED"
        color = (0, 255, 255) 
    elif M_live is None and not is_area_locked: # M_live がなく、ロックもされてない
        status_text = "AREA NOT FOUND"
        color = (100, 100, 100)
    
    cv2.putText(frame_top, f"STATUS: {status_text} (Hand: {TARGET_HAND})", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # エリアロック状態の表示 (新規追加)
    if is_area_locked:
        area_status_text = "AREA LOCKED"
        area_color = (0, 255, 0) # 緑
    else:
        area_status_text = "AREA UNLOCKED"
        area_color = (0, 165, 255) # オレンジ
    cv2.putText(frame_top, area_status_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, area_color, 2)

    # オフセット調整がまだなら、警告を出す
    if is_area_locked and not is_brush_calibrated: # エリアロック後のみ警告
        cv2.putText(frame_top, "CALIBRATE BRUSH OFFSET (Press 'o')", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    elif not is_area_locked and M_live is None:
         cv2.putText(frame_top, "Find Area Markers (0,1,2,3) to Lock", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)


    cv2.imshow("Top-Down View (X/Y)", frame_top)
    cv2.imshow("Side View (Z/Pressure)", frame_side)

    # 5.10. キー入力 (大幅に変更)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('k'):
        save_calibration_data()

    if key == ord('p'): 
        if is_recording_session: 
            is_manually_paused = not is_manually_paused
            if is_manually_paused:
                print("--- 記録を一時停止 (Manual Pause) ---")
            else:
                print("--- 記録を再開 ---")

    # 'l' キーの処理 (新規追加)
    if key == ord('l'):
        if is_recording_session:
            print("エラー: 記録セッション中はエリアロックを変更できません。")
        else:
            if not is_area_locked:
                # これからロックする
                if M_live is not None:
                    M_locked = M_live
                    is_area_locked = True
                    print("--- エリアをロックしました ---")
                    print("★ 次に [o] キーで筆のオフセットを調整してください。")
                else:
                    print("エラー: 4隅のマーカーが認識されていません。ロックできません。")
            else:
                # ロックを解除する
                M_locked = None
                is_area_locked = False
                is_brush_calibrated = False # エリアが変わったらオフセットもリセット
                print("--- エリアのロックを解除しました (筆オフセットもリセットされました) ---")

    # 'o' キーの処理 (条件変更)
    if key == ord('o'):
        if is_recording_session:
            print("エラー: オフセット調整は記録セッション開始前に行ってください。")
        elif not is_area_locked: # ロックされていることが必須
            print("エラー: 'l'キーでエリアをロックしてからオフセット調整を行ってください。")
        elif M_to_use is None: # M_locked が None (lキーでセットされるはずだが念のため)
            print("エラー: エリアがロックされていません。")
        elif brush_marker_center_pixel is None:
            print(f"エラー: 筆マーカー (ID={BRUSH_MARKER_ID}) が認識できません。")
        
        elif src_pts[0] is None: 
            print("エラー: 左上マーカー (ID=0) が認識できません。")
        else:
            # --- オフセット計算を実行 ---
            # ★ ターゲット = ID=0 の「右下の角」ピクセル座標
            # (src_pts[0] は 5.5. で get_marker_point(0, ...) によって計算済み)
            target_pos_pixel = src_pts[0].astype(int)
            
            tip_vector_pixel = target_pos_pixel - brush_marker_center_pixel
            norm_x_sq = np.linalg.norm(brush_marker_xaxis_pixel)**2
            norm_y_sq = np.linalg.norm(brush_marker_yaxis_pixel)**2

            if norm_x_sq == 0 or norm_y_sq == 0:
                print("エラー: 筆マーカーが歪んでいます。")
            else:
                local_x = np.dot(tip_vector_pixel, brush_marker_xaxis_pixel) / norm_x_sq
                local_y = np.dot(tip_vector_pixel, brush_marker_yaxis_pixel) / norm_y_sq
                BRUSH_TIP_OFFSET_LOCAL = (local_x, local_y)
                is_brush_calibrated = True
                print(f"--- 筆オフセット調整 完了 ---")
                print(f"  ターゲット (ID=0 Corner): {target_pos_pixel}")
                print(f"  マーカー中心 (ID=50): {brush_marker_center_pixel}")
                print(f"  ローカルオフセット: ({local_x:.4f}, {local_y:.4f})")
                print("★ [k] キーで保存できます。")
                print("★ [s] キーで記録を開始できます。")


    # 's' キーの処理 (条件変更)
    if key == ord('s'):
        if not is_recording_session:
            # エリアロックとオフセット調整が完了しているかチェック
            if not is_area_locked:
                print("エラー: 'l'キーでエリアをロックしてください。")
            elif not is_brush_calibrated:
                print("エラー: 'o'キーで筆のオフセット調整を先に行ってください。")
            else:
                is_recording_session = True
                print("--- 記録セッション開始 ---")
        else:
            is_recording_session = False
            is_pen_down = False 
            is_manually_paused = False 
            print("--- 記録セッション停止 ---")

# --- 6. 終了処理 ---
cap_top.release()
cap_side.release()
hands_top.close()
hands_side.close()
cv2.destroyAllWindows()
save_all_data()