import cv2
import cv2.aruco as aruco
import numpy as np
import time
import csv
import os
import sys
import mediapipe as mp # ★ 変更点：MediaPipeをインポート

# --- 1. 設定項目 ---
# ArUcoマーカーの設定 (範囲検出用)
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR_PARAMS.polygonalApproxAccuracyRate = 0.05
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

# 範囲マーカーIDの割り当て
CORNER_IDS = [0, 1, 2, 3] # [Top-Left, Top-Right, Bottom-Right, Bottom-Left]

# 範囲マーカーID -> 使用する角のインデックス
CORNER_INDEX_MAP = {
    0: 2, # ID 0 (エリア左上) -> 右下の角
    1: 3, # ID 1 (エリア右上) -> 左下の角
    2: 0, # ID 2 (エリア右下) -> 左上の角
    3: 1  # ID 3 (エリア左下) -> 右上の角
}

# ★ 変更点：MediaPipe Hands の初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# 検出する手を1つに制限（パフォーマンス向上のため）
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
is_recording = False
stroke_count = 0
last_cell_id = -1
last_record_time = 0
last_pen_pos_norm = None # 最後に認識したペンの「正規化後」座標
M_live = None   
M_locked = None 

# --- 4. ヘルパー関数 ---

def select_camera_index():
    """利用可能なカメラをプレビューし、ユーザーに使用するカメラのインデックスを選択させる。"""
    print("利用可能なカメラを探しています...")
    available_indices = []
    for i in range(10):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            available_indices.append(i)
            cap_test.release()

    if not available_indices:
        print("エラー: 利用可能なカメラが見つかりません。")
        return None
    print(f"利用可能なカメラのインデックス: {available_indices}")
    print("プレビューウィンドウで 'y' (Yes) または 'n' (No) を押してください。")

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
            cv2.putText(frame, f"Camera Index: {index}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, "Use this camera? (y/n)", (10, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Camera Selection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                print(f"-> カメラ {index} を選択しました。")
                selected_index = index
                break
            if key == ord('n'):
                print(f"-> カメラ {index} をスキップします。")
                break
        cap.release()
        if selected_index is not None:
            break
    cv2.destroyAllWindows()
    if selected_index is None:
        print("カメラが選択されませんでした。")
    return selected_index

# ★ 変更点：get_marker_point (PEN_IDのロジックを削除し、範囲マーカー専用に)
def get_marker_point(target_id, detected_ids, detected_corners):
    """
    検出された範囲マーカーから、指定IDの「内側の角」の座標(ピクセル)を返す。
    """
    if detected_ids is not None:
        for i, marker_id in enumerate(detected_ids):
            if marker_id == target_id:
                # 範囲マーカー(ID 0-3)かチェック
                if target_id in CORNER_INDEX_MAP:
                    corners = detected_corners[i][0]
                    corner_index = CORNER_INDEX_MAP[target_id]
                    point = corners[corner_index]
                    return point.astype(int)
    return None

def convert_to_custom_coords(norm_x, norm_y):
    """内部座標(0-800)を、指定の座標系(-200〜0)に変換"""
    norm_x_01 = norm_x / WARPED_SIZE
    norm_y_01 = norm_y / WARPED_SIZE
    converted_x = (norm_x_01 - 1.0) * COORD_LIMIT
    converted_y = norm_y_01 * -COORD_LIMIT
    return converted_x, converted_y

def get_cell_id(norm_x, norm_y):
    """内部座標(0-800)から、セルID(0-63)を計算"""
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

def record_data(event_type, timestamp, stroke_id, pen_pos_norm):
    """データを整形してリストに追加する"""
    global last_cell_id, last_record_time

    (norm_x, norm_y) = pen_pos_norm
    x, y = convert_to_custom_coords(norm_x, norm_y)
    cell_id = get_cell_id(norm_x, norm_y)
    pressure = PRESSURE_MAX if (event_type == 'down' or event_type == 'move') else 0
    
    drawing_data.append({
        'timestamp': timestamp, 'event_type': event_type, 'stroke_id': stroke_id,
        'x': f"{x:.2f}", 'y': f"{y:.2f}", 'pressure': f"{pressure:.4f}", 'cell_id': cell_id
    })
    
    if event_type != 'up' and last_cell_id != -1 and cell_id != last_cell_id:
        curr_x, curr_y = cell_id % GRID_SIZE, cell_id // GRID_SIZE
        prev_x, prev_y = last_cell_id % GRID_SIZE, last_cell_id // GRID_SIZE
        if abs(curr_x - prev_x) + abs(curr_y - prev_y) == 1:
            cell_transitions.append({
                'timestamp': timestamp, 'stroke_id': stroke_id,
                'from_cell': last_cell_id, 'to_cell': cell_id
            })
            
    last_cell_id = cell_id if event_type != 'up' else -1
    if event_type != 'up':
        last_record_time = timestamp


# --- 5. メイン処理 ---
camera_index = select_camera_index()
if camera_index is None:
    print("カメラが選択されなかったため、プログラムを終了します。")
    sys.exit()

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"エラー: 選択されたカメラ {camera_index} を開けませんでした。")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"カメラ解像度を {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} で起動します。")

# 出力先（真上から見た）座標の定義
dst_pts = np.float32([
    [0, 0], [WARPED_SIZE, 0], [WARPED_SIZE, WARPED_SIZE], [0, WARPED_SIZE]
])

print("--- カメラトラッキング開始 ---")
print(" [s] キー: 記録の開始/停止 (4隅のマーカー認識時に開始可能)")
print(" [q] キー: 終了してCSV保存")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    
    # ★ 変更点：MediaPipeはRGB画像、ArUcoはBGR画像(frame)を使用
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. ArUco (範囲) の検出 (BGRフレームを使用)
    (detected_corners, detected_ids, rejected) = DETECTOR.detectMarkers(frame)
    if detected_ids is not None:
        aruco.drawDetectedMarkers(frame, detected_corners, detected_ids)

    src_pts = [get_marker_point(id, detected_ids, detected_corners) for id in CORNER_IDS]
    
    if all(pt is not None for pt in src_pts):
        src_pts_np = np.float32(src_pts)
        M_live = cv2.getPerspectiveTransform(src_pts_np, dst_pts) 
        cv2.polylines(frame, [src_pts_np.astype(int)], True, (0, 255, 0), 2)
    
    # 2. どの変換行列(M)を使用するか決定
    M_to_use = None
    if is_recording and M_locked is not None:
        M_to_use = M_locked 
    elif M_live is not None:
        M_to_use = M_live 

    # 3. MediaPipe (筆) の処理
    pen_pos_norm = None # このフレームでのペンの位置をリセット

    if M_to_use is not None:
        # ★ 変更点：MediaPipeで手を処理 (RGBフレームを使用)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # 手が検出された場合
            hand_landmarks = results.multi_hand_landmarks[0] # 最初の手
            
            # 手の骨格をBGRフレームに描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- ★ 追跡する指先を選択 ---
            # 人差し指の先端 (ID 8)
            landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # (もし親指の先端 (ID 4) にしたい場合は、上の行をコメントアウトし、下の行を使用)
            # landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            # --------------------------

            # landmark.x, landmark.y は (0.0-1.0) の正規化座標
            h, w, _ = frame.shape
            pen_center_pixel_x = int(landmark.x * w)
            pen_center_pixel_y = int(landmark.y * h)
            pen_center_pixel = (pen_center_pixel_x, pen_center_pixel_y)

            # 座標を射影変換
            pen_pixel_np = np.float32([[pen_center_pixel]])
            pen_pos_norm_raw = cv2.perspectiveTransform(pen_pixel_np, M_to_use)
            pen_pos_norm = (pen_pos_norm_raw[0][0][0], pen_pos_norm_raw[0][0][1])
            last_pen_pos_norm = pen_pos_norm # 最後に検出した位置を保持

            # 記録中('move'イベント)
            if is_recording:
                if current_time - last_record_time >= SAMPLING_INTERVAL:
                    record_data('move', current_time, stroke_count, pen_pos_norm)

    # 4. 録画状態を画面に表示
    status_text = "RECORDING" if is_recording else "PAUSED"
    color = (0, 0, 255) if is_recording else (0, 165, 255)
    if M_live is None and not is_recording:
        status_text = "AREA NOT FOUND"
        color = (100, 100, 100)
        
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 5. キー入力の処理
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break # ループを抜けて終了処理へ

    if key == ord('s'):
        if not is_recording:
            # --- 記録開始 ---
            if M_live is not None:
                M_locked = M_live # ★ M をロック
                is_recording = True
                stroke_count += 1
                print(f"Recording START (Stroke {stroke_count}) - 範囲をロックしました")
                
                # 'down' イベントを記録 (最後に認識したペン位置で)
                if last_pen_pos_norm:
                    record_data('down', current_time, stroke_count, last_pen_pos_norm)
                else:
                    print("Warning: ペン(手)が認識されていませんが、記録を開始します。")
            else:
                print("Error: 4隅のマーカーが認識されていません。記録を開始できません。")
        
        else:
            # --- 記録停止 ---
            is_recording = False
            M_locked = None # ★ M をアンロック
            print("Recording STOP - 範囲のロックを解除しました")
            if last_pen_pos_norm: # 最後に認識したペン位置があれば
                record_data('up', current_time, stroke_count, last_pen_pos_norm)

    cv2.imshow("Camera View (Press 's' to Record, 'q' to Quit)", frame)

# --- 6. 終了処理 ---
cap.release()
hands.close() # ★ 変更点：MediaPipeをクローズ
cv2.destroyAllWindows()
save_all_data()