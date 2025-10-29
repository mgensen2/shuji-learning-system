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

# MediaPipe Hands の初期化 (2つのインスタンスを作成)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# TOPカメラ用
hands_top = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# SIDEカメラ用
hands_side = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 座標系の設定
COORD_LIMIT = 200.0
PRESSURE_MAX = 8.0
GRID_SIZE = 8
SAMPLING_INTERVAL = 0.05
WARPED_SIZE = 800
CELL_SIZE_PX = WARPED_SIZE / GRID_SIZE

# --- 2. データ格納リスト ---
drawing_data = []     
cell_transitions = [] 

# --- 3. 状態変数 ---
is_recording_session = False # 's'キーでトグルするセッション全体の状態
is_pen_down = False          # 筆が紙に触れているか (自動検出)
stroke_count = 0
last_cell_id = -1
last_record_time = 0
last_pen_pos_norm = None 
M_live = None   
M_locked = None 

# ★ 筆圧キャリブレーション用の変数
Y_TOUCH_THRESHOLD = -1 # キャリブレーションで決定されるY座標


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

def calibrate_side_camera(cap_side):
    """Side-Viewカメラの筆圧(Z軸)キャリブレーションを行う"""
    global Y_TOUCH_THRESHOLD
    print("--- 筆圧 (Z軸) キャリブレーション ---")
    print("筆（人差し指の先端）を紙に「触れさせた」状態で 'c' キーを押してください。")
    
    current_y = -1
    
    while True:
        ret, frame = cap_side.read()
        if not ret:
            print("エラー: Sideカメラから読み込めません")
            return False
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_side.process(frame_rgb)
        
        h, w, _ = frame.shape
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 人差し指の先端(ID 8)を取得
            landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # ピクセル座標に変換 (Y座標のみ重要)
            current_y = int(landmark.y * h)
            current_x = int(landmark.x * w)
            
            # 検出中のY座標を画面に表示
            cv2.putText(frame, f"Detected Y: {current_y}", (current_x + 10, current_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(frame, (current_x, current_y), 5, (0, 0, 255), -1)

        # 画面に指示を表示
        cv2.rectangle(frame, (0, 0), (w, 40), (0,0,0), -1)
        cv2.putText(frame, "Touch pen to paper, then press 'c' to calibrate", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Side Camera Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if current_y != -1:
                Y_TOUCH_THRESHOLD = current_y
                print(f"キャリブレーション完了。タッチしきい値(Y座標) = {Y_TOUCH_THRESHOLD}")
                cv2.destroyAllWindows()
                return True
            else:
                print("エラー: 手が検出されていません。 'c' を押す前に手を映してください。")
        
        if key == ord('q'):
            print("キャリブレーションがキャンセルされました。")
            cv2.destroyAllWindows()
            return False
            
    return False

def get_marker_point(target_id, detected_ids, detected_corners):
    if detected_ids is not None:
        for i, marker_id in enumerate(detected_ids):
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

def record_data(event_type, timestamp, pressure, pen_pos_norm):
    """データを整形してリストに追加する"""
    global last_cell_id, last_record_time, stroke_count

    # 50msの間隔チェック ('down', 'up' は間引かない)
    if event_type == 'move':
        if timestamp - last_record_time < SAMPLING_INTERVAL:
            return # 間引く
    
    last_record_time = timestamp

    # 座標とセルIDの計算
    (norm_x, norm_y) = pen_pos_norm
    x, y = convert_to_custom_coords(norm_x, norm_y)
    cell_id = get_cell_id(norm_x, norm_y)
    
    drawing_data.append({
        'timestamp': timestamp, 'event_type': event_type, 'stroke_id': stroke_count,
        'x': f"{x:.2f}", 'y': f"{y:.2f}", 'pressure': f"{pressure:.4f}", 'cell_id': cell_id
    })
    
    # セル移動の検出
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
print(f"Topカメラ解像度: {cap_top.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

side_cam_index = select_camera_index("Side-View (Z/Pressure) Camera")
if side_cam_index is None: sys.exit("Side-Viewカメラが選択されませんでした。")
cap_side = cv2.VideoCapture(side_cam_index)
cap_side.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_side.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Sideカメラ解像度: {cap_side.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# 5.2. Sideカメラのキャリブレーション
if not calibrate_side_camera(cap_side):
    sys.exit("キャリブレーションがキャンセルされました。")

# 5.3. メインループの準備
dst_pts = np.float32([[0, 0], [WARPED_SIZE, 0], [WARPED_SIZE, WARPED_SIZE], [0, WARPED_SIZE]])
print("--- トラッキング開始 ---")
print(" [s] キー: 記録セッションの開始/停止 (筆の上下で自動記録)")
print(" [q] キー: 終了してCSV保存")

while True:
    # 5.4. 両方のカメラからフレームを取得
    ret_top, frame_top = cap_top.read()
    ret_side, frame_side = cap_side.read()
    if not ret_top or not ret_side:
        print("エラー: カメラフレームを読み込めません")
        break
    
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
    
    M_to_use = M_locked if (is_recording_session and M_locked is not None) else M_live

    # 5.6. Sideカメラの処理 (筆圧Z軸の検出)
    is_touching_now = False # このフレームでタッチしているか
    frame_side_rgb = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)
    results_side = hands_side.process(frame_side_rgb)
    
    if results_side.multi_hand_landmarks:
        h_side, w_side, _ = frame_side.shape
        landmark = results_side.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pen_y_side = int(landmark.y * h_side)
        
        # しきい値と比較 (注意: ピクセルY座標は「下」に行くほど値が大きくなる)
        if pen_y_side >= Y_TOUCH_THRESHOLD:
            is_touching_now = True
            
        # Sideカメラのプレビューに状態を描画
        cv2.line(frame_side, (0, Y_TOUCH_THRESHOLD), (w_side, Y_TOUCH_THRESHOLD), (0, 255, 255), 2)
        cv2.putText(frame_side, f"TOUCH_Y_LEVEL: {Y_TOUCH_THRESHOLD}", (10, Y_TOUCH_THRESHOLD - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        px_side = int(landmark.x * w_side)
        color = (0, 0, 255) if is_touching_now else (0, 255, 0)
        cv2.circle(frame_side, (px_side, pen_y_side), 8, color, -1)


    # 5.7. Topカメラの処理 (X/Y座標検出)
    pen_pos_norm = None
    if M_to_use is not None:
        frame_top_rgb = cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB)
        results_top = hands_top.process(frame_top_rgb)
        
        if results_top.multi_hand_landmarks:
            h_top, w_top, _ = frame_top.shape
            landmark_top = results_top.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pen_center_pixel = (int(landmark_top.x * w_top), int(landmark_top.y * h_top))
            
            pen_pixel_np = np.float32([[pen_center_pixel]])
            pen_pos_norm_raw = cv2.perspectiveTransform(pen_pixel_np, M_to_use)
            pen_pos_norm = (pen_pos_norm_raw[0][0][0], pen_pos_norm_raw[0][0][1])
            last_pen_pos_norm = pen_pos_norm # 最後に検出したX/Y位置を保持

    # 5.8. 状態機械 (State Machine) による自動記録
    if is_recording_session and last_pen_pos_norm is not None:
        
        if is_touching_now and not is_pen_down:
            # --- 状態：Pen Down (触れた瞬間) ---
            is_pen_down = True
            stroke_count += 1
            print(f"Stroke {stroke_count} START (Down)")
            record_data('down', current_time, PRESSURE_MAX, last_pen_pos_norm)
        
        elif is_touching_now and is_pen_down:
            # --- 状態：Pen Move (触れ続けている) ---
            record_data('move', current_time, PRESSURE_MAX, last_pen_pos_norm)
            
        elif not is_touching_now and is_pen_down:
            # --- 状態：Pen Up (離れた瞬間) ---
            is_pen_down = False
            print(f"Stroke {stroke_count} END (Up)")
            record_data('up', current_time, 0, last_pen_pos_norm)
        
        # (not is_touching_now and not is_pen_down の場合は「空中」なので記録しない)

    # 5.9. 画面表示
    status_text = "RECORDING" if is_recording_session else "PAUSED"
    color = (0, 0, 255) if is_recording_session else (0, 165, 255)
    if M_live is None and not is_recording_session:
        status_text = "AREA NOT FOUND"
        color = (100, 100, 100)
    cv2.putText(frame_top, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Top-Down View (X/Y)", frame_top)
    cv2.imshow("Side View (Z/Pressure)", frame_side)

    # 5.10. キー入力
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        if not is_recording_session:
            if M_live is not None:
                M_locked = M_live # 範囲をロック
                is_recording_session = True
                print("--- 記録セッション開始 --- (筆の上下を自動検出します)")
            else:
                print("エラー: 4隅のマーカーが認識されていません。")
        else:
            is_recording_session = False
            is_pen_down = False # 記録停止時にペンステータスをリセット
            M_locked = None
            print("--- 記録セッション停止 ---")

# --- 6. 終了処理 ---
cap_top.release()
cap_side.release()
hands_top.close()
hands_side.close()
cv2.destroyAllWindows()
save_all_data()