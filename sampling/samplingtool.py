import cv2
import cv2.aruco as aruco
import numpy as np
import time
import csv
import os
import sys


def select_camera_index():
    """
    利用可能なカメラをプレビューし、ユーザーに使用するカメラのインデックスを選択させる。

    Returns:
        int: 選択されたカメラのインデックス。見つからないか選択されなかった場合は None。
    """
    print("利用可能なカメラを探しています...")
    
    available_indices = []
    # 一般的に0から9までのインデックスをチェック
    for i in range(10):
        # cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # Windowsで高速化する場合
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
        if not cap.isOpened():
            continue

        print(f"--- カメラ {index} をテスト中 ---")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"エラー: カメラ {index} からフレームを読み込めません。")
                break
            
            # ウィンドウに情報を表示
            h, w = frame.shape[:2]
            text_size = 1
            text_color = (0, 0, 255) # 赤
            bg_color = (255, 255, 255) # 白
            
            # テキスト背景
            cv2.rectangle(frame, (0, 0), (w, 80), bg_color, -1)
            
            cv2.putText(frame, f"Camera Index: {index}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 2)
            cv2.putText(frame, "Use this camera? (y/n)", (10, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 2)
            
            cv2.imshow("Camera Selection", frame)
            
            # 1ms待機
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('y'):
                print(f"-> カメラ {index} を選択しました。")
                selected_index = index
                break # 'y' が押されたら while ループを抜ける
            
            if key == ord('n'):
                print(f"-> カメラ {index} をスキップします。")
                break # 'n' が押されたら while ループを抜ける
        
        cap.release()
        
        if selected_index is not None:
            break # 選択されたら for ループも抜ける

    cv2.destroyAllWindows()
    
    if selected_index is None:
        print("カメラが選択されませんでした。")
        
    return selected_index

# --- 1. 設定項目 ---
# ArUcoマーカーの設定
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

# マーカーIDの割り当て
CORNER_IDS = [0, 1, 2, 3] # [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
PEN_ID = 4

# 座標系の設定 (iPad版と同じ)
COORD_LIMIT = 200.0 # 座標 (0 〜 -200)
PRESSURE_MAX = 8.0  # 記録する筆圧 (書いている=8, 離している=0)
GRID_SIZE = 8
SAMPLING_INTERVAL = 0.05 # 記録間隔 (秒) = 50ms

# 内部処理用の解像度 (このサイズに射影変換される)
WARPED_SIZE = 800
CELL_SIZE_PX = WARPED_SIZE / GRID_SIZE


# --- 2. データ格納リスト ---
drawing_data = []     # (timestamp, event_type, stroke_id, x, y, pressure, cell_id)
cell_transitions = [] # (timestamp, stroke_id, from_cell, to_cell)

# --- 3. 状態変数 ---
is_recording = False
stroke_count = 0
last_cell_id = -1
last_record_time = 0
last_pen_pos_norm = None # 描画用

# --- 4. ヘルパー関数 ---

# マッピング: 範囲マーカーID -> 使用する角のインデックス (0:TL, 1:TR, 2:BR, 3:BL)
CORNER_INDEX_MAP = {
    0: 2, # ID 0 (エリア左上) -> 右下の角
    1: 3, # ID 1 (エリア右上) -> 左下の角
    2: 0, # ID 2 (エリア右下) -> 左上の角
    3: 1  # ID 3 (エリア左下) -> 右上の角
}

def get_marker_point(target_id, detected_ids, detected_corners):
    """
    検出されたマーカーから、指定IDの座標(ピクセル)を返す。
    - 範囲マーカー(0-3)の場合: IDに応じた「内側の角」の座標を返す。
    - 筆マーカー(4)の場合: 「中心」の座標を返す。
    """
    if detected_ids is not None:
        for i, marker_id in enumerate(detected_ids):
            if marker_id == target_id:
                corners = detected_corners[i][0]
                
                if target_id == PEN_ID:
                    # 筆マーカー(ID 4)は中心を返す
                    center = np.mean(corners, axis=0)
                    return center.astype(int)
                
                elif target_id in CORNER_INDEX_MAP:
                    # 範囲マーカー(ID 0-3)は指定された角を返す
                    corner_index = CORNER_INDEX_MAP[target_id]
                    point = corners[corner_index]
                    return point.astype(int)
                    
    return None

def convert_to_custom_coords(norm_x, norm_y):
    """内部座標(0-800)を、指定の座標系(-200〜0)に変換"""
    # 0.0〜1.0に正規化
    norm_x_01 = norm_x / WARPED_SIZE
    norm_y_01 = norm_y / WARPED_SIZE
    
    # X座標 (右上が0, 左が-200)
    converted_x = (norm_x_01 - 1.0) * COORD_LIMIT
    # Y座標 (右上が0, 下が-200)
    converted_y = norm_y_01 * -COORD_LIMIT
    
    return converted_x, converted_y

def get_cell_id(norm_x, norm_y):
    """内部座標(0-800)から、セルID(0-63)を計算"""
    cell_x = int(norm_x // CELL_SIZE_PX)
    cell_y = int(norm_y // CELL_SIZE_PX)
    
    # 範囲外にはみ出ないようにクリップ
    cell_x = max(0, min(cell_x, GRID_SIZE - 1))
    cell_y = max(0, min(cell_y, GRID_SIZE - 1))
    
    return (cell_y * GRID_SIZE) + cell_x

def save_all_data():
    """プログラム終了時にデータをCSVに保存"""
    print(f"Saving {len(drawing_data)} data points...")
    
    # 1. 全データ (full)
    if drawing_data:
        headers = drawing_data[0].keys()
        with open('unpitsu_data_full.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(drawing_data)
        print("Saved unpitsu_data_full.csv")

    # 2. セル移動データ (transitions)
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
    
    # 座標変換
    x, y = convert_to_custom_coords(norm_x, norm_y)
    
    # セルID
    cell_id = get_cell_id(norm_x, norm_y)
    
    # 筆圧 (0 or 8)
    pressure = PRESSURE_MAX if (event_type == 'down' or event_type == 'move') else 0
    
    # 全データに追加
    drawing_data.append({
        'timestamp': timestamp,
        'event_type': event_type,
        'stroke_id': stroke_id,
        'x': f"{x:.2f}",
        'y': f"{y:.2f}",
        'pressure': f"{pressure:.4f}",
        'cell_id': cell_id
    })
    
    # セル移動の検出
    if event_type != 'up' and last_cell_id != -1 and cell_id != last_cell_id:
        # マンハッタン距離が1（上下左右）かチェック
        curr_x, curr_y = cell_id % GRID_SIZE, cell_id // GRID_SIZE
        prev_x, prev_y = last_cell_id % GRID_SIZE, last_cell_id // GRID_SIZE
        
        if abs(curr_x - prev_x) + abs(curr_y - prev_y) == 1:
            cell_transitions.append({
                'timestamp': timestamp,
                'stroke_id': stroke_id,
                'from_cell': last_cell_id,
                'to_cell': cell_id
            })
            
    # 状態更新
    last_cell_id = cell_id if event_type != 'up' else -1
    if event_type != 'up':
        last_record_time = timestamp


# --- 5. メイン処理 ---
# カメラを選択
camera_index = select_camera_index()
if camera_index is None:
    print("カメラが選択されなかったため、プログラムを終了します。")
    sys.exit()

# 選択されたカメラでキャプチャを開始
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"エラー: 選択されたカメラ {camera_index} を開けませんでした。")
    sys.exit()

# 出力先（真上から見た）座標の定義
dst_pts = np.float32([
    [0, 0],                  # Top-Left
    [WARPED_SIZE, 0],        # Top-Right
    [WARPED_SIZE, WARPED_SIZE], # Bottom-Right
    [0, WARPED_SIZE]         # Bottom-Left
])

print("--- カメラトラッキング開始 ---")
print(" [s] キー: 記録の開始/停止")
print(" [q] キー: 終了してCSV保存")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # マーカーを検出
    (detected_corners, detected_ids, rejected) = DETECTOR.detectMarkers(frame)
    
    # 検出されたマーカーを映像に描画
    if detected_ids is not None:
        aruco.drawDetectedMarkers(frame, detected_corners, detected_ids)

    # 4隅のマーカーの中心座標を取得
    src_pts = [get_marker_point(id, detected_ids, detected_corners) for id in CORNER_IDS]

    # --- 4隅がすべて検出された場合のみ処理
    if all(pt is not None for pt in src_pts):
        src_pts_np = np.float32(src_pts)
        
        # 射影変換行列を計算
        M = cv2.getPerspectiveTransform(src_pts_np, dst_pts)
        
        # 範囲を緑色の線で囲む
        cv2.polylines(frame, [src_pts_np.astype(int)], True, (0, 255, 0), 2)
        
        # 筆マーカー(ID 4)の中心座標を取得
        pen_center_pixel = get_marker_point(PEN_ID, detected_ids, detected_corners)
        
        # --- 筆マーカーが検出された場合
        if pen_center_pixel is not None:
            # 筆マーカーの座標を射影変換
            # (cv2.perspectiveTransformは[1, N, 2]の形式のNumpy配列を要求する)
            pen_pixel_np = np.float32([[pen_center_pixel]])
            pen_pos_norm_raw = cv2.perspectiveTransform(pen_pixel_np, M)
            
            # (x, y) 座標タプルとして取得
            pen_pos_norm = (pen_pos_norm_raw[0][0][0], pen_pos_norm_raw[0][0][1])
            last_pen_pos_norm = pen_pos_norm # 描画用に保持
            
            current_time = time.time()

            # --- 記録中('move'イベント) ---
            if is_recording:
                # 50msの間隔チェック
                if current_time - last_record_time >= SAMPLING_INTERVAL:
                    record_data('move', current_time, stroke_count, pen_pos_norm)

    # --- 録画状態を画面に表示 ---
    status_text = "RECORDING" if is_recording else "PAUSED"
    color = (0, 0, 255) if is_recording else (0, 165, 255)
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # --- キー入力の処理 ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break # ループを抜けて終了処理へ

    if key == ord('s'):
        current_time = time.time()
        
        if not is_recording:
            # --- 記録開始 ---
            if last_pen_pos_norm: # ペンが認識されている場合のみ開始
                is_recording = True
                stroke_count += 1
                print(f"Recording START (Stroke {stroke_count})")
                # 'down' イベントを記録
                record_data('down', current_time, stroke_count, last_pen_pos_norm)
        else:
            # --- 記録停止 ---
            is_recording = False
            print("Recording STOP")
            if last_pen_pos_norm:
                # 'up' イベントを記録
                record_data('up', current_time, stroke_count, last_pen_pos_norm)


    cv2.imshow("Camera View (Press 's' to Record, 'q' to Quit)", frame)

# --- 6. 終了処理 ---
cap.release()
cv2.destroyAllWindows()
save_all_data()