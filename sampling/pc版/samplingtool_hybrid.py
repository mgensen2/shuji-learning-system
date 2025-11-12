import cv2
import cv2.aruco as aruco
import numpy as np
import time
import csv
import os
import sys
import mediapipe as mp

# --- 1. 設定項目 (Settings) ---
# ArUcoマーカーの設定 (範囲検出用)
# ArUco marker settings (for area detection)
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR_PARAMS.polygonalApproxAccuracyRate = 0.05
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

CORNER_IDS = [0, 1, 2, 3] # [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
CORNER_INDEX_MAP = { 0: 2, 1: 3, 2: 0, 3: 1 } # マーカーのどの角を基準点にするか (Which corner to use as the reference point)

BRUSH_MARKER_ID = 4 # 筆に取り付けるマーカーのID (ID for the marker attached to the brush)

# エリアロック/オフセット設定の保存ファイル名
# Filename to save area lock / offset settings
CALIB_FILE_NAME = 'unpitsu_calibration.npz'

# Topカメラのレンズ歪み補正ファイル
# Top camera lens distortion correction file
LENS_CALIB_FILE_NAME = 'top_camera_lens.npz' 

# analyze_calligraphy.py と連携するためのファイル名
# Filename to link with analyze_calligraphy.py
CAPTURE_IMAGE_FILENAME = 'calligraphy_image.png' 

# MediaPipe Hands の初期化 (Initialization)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_top = mp.solutions.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_side = mp.solutions.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 座標系の設定 (Coordinate system settings)
COORD_LIMIT = 200.0
GRID_SIZE = 8
SAMPLING_INTERVAL = 0.05 # 記録間隔 (秒) (Recording interval (seconds))
WARPED_SIZE = 800       # 補正後の画像サイズ (px) (Warped image size)
CELL_SIZE_PX = WARPED_SIZE / GRID_SIZE

# --- 2. データ格納リスト (Data storage lists) ---
drawing_data = []     
cell_transitions = [] 

# --- 3. 状態変数 (State variables) ---
is_recording_session = False
is_pen_down = False         
is_manually_paused = False  
is_area_locked = False        # lキーでエリアロックを管理 (Manage area lock with 'l' key)
stroke_count = 0
last_cell_id = -1             # ★ 最後に記録されたセルID (Last recorded cell ID)
last_record_time = 0
last_pen_pos_norm = None 
M_live = None   # リアルタイムの変換行列 (Live transformation matrix)
M_locked = None # 固定された変換行列 (Locked transformation matrix)
TARGET_HAND = "Any"
TRACKING_MODE = "Hybrid"      # "Hybrid", "MarkerOnly", "MediaPipeOnly"
RECORDING_MODE = "Time"       # "Time" (時間) or "Spatial" (空間/セル移動)

# 筆圧キャリブレーション用 (For pressure calibration)
Y_TOUCH_THRESHOLD = -1     
Y_MAX_PRESS_THRESHOLD = -1 

# 筆オフセット用 (For brush offset)
BRUSH_TIP_OFFSET_LOCAL = None 
is_brush_calibrated = False   

# レンズ歪み補正用 (For lens distortion correction)
TOP_CAM_MTX = None
TOP_CAM_DIST = None
TOP_CAM_MAP_X = None
TOP_CAM_MAP_Y = None


# --- 4. ヘルパー関数 (Helper functions) ---
def select_camera_index(prompt_text):
    """カメラを選択させる（'n'でスキップ可能にする）"""
    """Lets user select a camera (allows skipping with 'n')"""
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
    """Lets user select the hand to track"""
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

def select_tracking_mode():
    """ユーザーにXY軸の追跡モードを選択させる"""
    """Lets user select the XY tracking mode"""
    print("\n--- XY軸 追跡モードを選択 ---")
    print(" [h] ハイブリッド (Hybrid) - [推奨] マーカー優先 + 指で自動補完")
    print(" [m] マーカーのみ (Marker Only) - マーカー(ID=50)のみ追跡")
    print(" [p] MediaPipeのみ (MediaPipe Only) - 指の先端のみ追跡")
    
    while True:
        key_in = input("追跡モードを選んでください (h/m/p): ").strip().lower()
        if key_in == 'h' or key_in == '':
            print("-> [ハイブリッド] モードを選択しました。")
            return "Hybrid"
        elif key_in == 'm':
            print("-> [マーカーのみ] モードを選択しました。")
            return "MarkerOnly"
        elif key_in == 'p':
            print("-> [MediaPipeのみ] モードを選択しました。")
            return "MediaPipeOnly"
        else:
            print("無効な入力です。'h', 'm', 'p' のいずれかを入力してください。")

def select_recording_mode():
    """ユーザーに記録モードを選択させる"""
    """Lets user select the recording mode"""
    print("\n--- 記録モードを選択 ---")
    print(" [t] 時間ベース (Time) - [推奨] 0.05秒ごとに記録")
    print(" [s] 空間ベース (Spatial) - セルを移動した時のみ記録")
    
    while True:
        key_in = input("記録モードを選んでください (t/s): ").strip().lower()
        if key_in == 't' or key_in == '':
            print(f"-> [時間ベース] モードを選択しました。 (間隔: {SAMPLING_INTERVAL}秒)")
            return "Time"
        elif key_in == 's':
            print("-> [空間ベース] モードを選択しました。 (セル移動時のみ)")
            return "Spatial"
        else:
            print("無効な入力です。't', 's' のいずれかを入力してください。")

def get_target_hand_landmarks(results, target_hand_label):
    """
    検出結果から、指定された手（"Left", "Right", "Any"）のランドマークを取得する
    """
    """Gets landmarks for the specified hand ("Left", "Right", or "Any") from the results"""
    if not results.multi_hand_landmarks:
        return None 

    if target_hand_label == "Any":
        return results.multi_hand_landmarks[0] # 最初のもの (Return the first one)

    if not results.multi_handedness:
        return None 

    for i, handedness in enumerate(results.multi_handedness):
        label = handedness.classification[0].label
        if label == target_hand_label:
            return results.multi_hand_landmarks[i] 
    
    return None 

def calibrate_pressure_range(cap_side, target_hand_label): 
    """Side-Viewカメラの筆圧(Z軸)キャリブレーションを2段階で行う"""
    """Performs 2-step pressure (Z-axis) calibration for the Side-View camera"""
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
        # 基準線を表示 (Show reference line)
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
                if current_y > Y_TOUCH_THRESHOLD: # Y座標は下が大きい (Y coordinate is larger downwards)
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
    """Gets the pixel coordinates of the "reference point" for a specific ID from the list of detected markers"""
    if detected_ids is not None:
        for i, marker_id in enumerate(detected_ids.flatten()):
            if marker_id == target_id:
                if target_id in CORNER_INDEX_MAP:
                    corners = detected_corners[i][0]
                    corner_index = CORNER_INDEX_MAP[target_id]
                    point = corners[corner_index]
                    return point.astype(int)
    return None

def convert_to_custom_coords(norm_x, norm_y):
    """(0,0)-(WARPED_SIZE, WARPED_SIZE) の座標を (-COORD_LIMIT, 0)-(0, -COORD_LIMIT) に変換"""
    """Converts coordinates from (0,0)-(WARPED_SIZE, WARPED_SIZE) to (-COORD_LIMIT, 0)-(0, -COORD_LIMIT)"""
    norm_x_01 = norm_x / WARPED_SIZE
    norm_y_01 = norm_y / WARPED_SIZE
    converted_x = (norm_x_01 - 1.0) * COORD_LIMIT
    converted_y = norm_y_01 * -COORD_LIMIT
    return converted_x, converted_y

def get_cell_id(norm_x, norm_y):
    """(0,0)-(WARPED_SIZE, WARPED_SIZE) の座標からグリッドID (0-63) を計算"""
    """Calculates grid ID (0-63) from (0,0)-(WARPED_SIZE, WARPED_SIZE) coordinates"""
    cell_x = int(norm_x // CELL_SIZE_PX)
    cell_y = int(norm_y // CELL_SIZE_PX)
    cell_x = max(0, min(cell_x, GRID_SIZE - 1))
    cell_y = max(0, min(cell_y, GRID_SIZE - 1))
    return (cell_y * GRID_SIZE) + cell_x

def save_all_data():
    """プログラム終了時にデータをCSVに保存"""
    """Saves data to CSV upon program exit"""
    global drawing_data, cell_transitions
    
    # ★ 'w'キーで保存されてデータが空の場合、何もしない
    # ★ If data is empty (already saved via 'w'), do nothing
    if not drawing_data:
        print("--- 終了処理: 保存すべきデータはありません ---")
        return

    print(f"--- 終了処理: 残りの {len(drawing_data)} 件のデータを保存します... ---")
    if drawing_data:
        headers = drawing_data[0].keys()
        # ★ デフォルトのファイル名に保存
        # ★ Save to default filenames
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
    
    # データをクリア
    drawing_data = []
    cell_transitions = []

# ★★★ 新規追加 ★★★
def save_data_with_prompt():
    """
    ユーザーにファイル名を入力させ、現在のデータを保存し、内部データをクリアする
    """
    """
    Prompts user for a filename, saves current data, and clears internal data
    """
    global drawing_data, cell_transitions, stroke_count
    
    if not drawing_data:
        print("--- 保存するデータがありません ---")
        return
    
    print("\n--- CSV保存 ---")
    print("!!! OpenCVウィンドウは一時停止します。このコンソールを見てください。!!!")
    
    # データをコピー (Copy data)
    temp_drawing_data = drawing_data.copy()
    temp_cell_transitions = cell_transitions.copy()
    
    # ファイル名を入力させる (Get filename input)
    filename_base = input("CSVのベース名を入力してください (例: my_writing_session): ").strip()
    
    if not filename_base:
        # デフォルトのファイル名 (Default filename)
        filename_base = f"unpitsu_data_{int(time.time())}"
        print(f"名前が入力されなかったので、デフォルト名を使用します: {filename_base}")

    # 保存パスを生成 (Generate save paths)
    full_csv_path = f"{filename_base}_full.csv"
    transitions_csv_path = f"{filename_base}_transitions.csv"

    # drawing_data (full) の保存
    try:
        headers = temp_drawing_data[0].keys()
        with open(full_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(temp_drawing_data)
        print(f"--- 成功: {full_csv_path} として保存しました。 ({len(temp_drawing_data)} 件)")
    except Exception as e:
        print(f"エラー: {full_csv_path} の保存に失敗しました: {e}")

    # cell_transitions の保存
    if temp_cell_transitions:
        try:
            headers = temp_cell_transitions[0].keys()
            with open(transitions_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(temp_cell_transitions)
            print(f"--- 成功: {transitions_csv_path} として保存しました。 ({len(temp_cell_transitions)} 件)")
        except Exception as e:
            print(f"エラー: {transitions_csv_path} の保存に失敗しました: {e}")

    # 内部データをリセット (Reset internal data)
    drawing_data = []
    cell_transitions = []
    stroke_count = 0
    print("--- 内部データをクリアしました。新しい記録を開始できます。 ---")
    print("\n--- OpenCVウィンドウに戻ります ---")


def save_calibration_data():
    """現在のキャリブレーションデータを.npzに保存する"""
    """Saves the current calibration data to .npz"""
    if not is_area_locked:
        print("エラー: エリアがロックされていません。 [l]キーでロックしてください。")
        return
    
    if TRACKING_MODE == "MediaPipeOnly":
        offset_to_save = np.array((0.0, 0.0)) 
        print("--- MediaPipeOnlyモード: エリアロックのみ保存します ---")
    else:
        if not is_brush_calibrated:
            print(f"エラー: 筆オフセットが調整されていません。 [o]キーで調整してください。")
            return
        offset_to_save = np.array(BRUSH_TIP_OFFSET_LOCAL)
        print("--- エリアロックと筆オフセットを保存します ---")
    
    try:
        np.savez(CALIB_FILE_NAME,
                 M_locked=M_locked,
                 BRUSH_TIP_OFFSET_LOCAL=offset_to_save
                )
        print(f"--- キャリブレーションデータを保存しました ({CALIB_FILE_NAME}) ---")
    except Exception as e:
        print(f"エラー: キャリブレーションデータの保存に失敗しました: {e}")

def load_calibration_data():
    """起動時にキャリブレーションデータを読み込む"""
    """Loads calibration data on startup"""
    global M_locked, BRUSH_TIP_OFFSET_LOCAL, is_area_locked, is_brush_calibrated
    
    if os.path.exists(CALIB_FILE_NAME):
        try:
            with np.load(CALIB_FILE_NAME) as data:
                M_locked = data['M_locked']
                is_area_locked = True
                
                if TRACKING_MODE != "MediaPipeOnly":
                    offset_data = data['BRUSH_TIP_OFFSET_LOCAL']
                    if offset_data.shape == ():
                        BRUSH_TIP_OFFSET_LOCAL = tuple(offset_data.item())
                    else:
                        BRUSH_TIP_OFFSET_LOCAL = tuple(offset_data)
                    is_brush_calibrated = True
                    print(f"--- キャリブレーションデータを読み込みました ({CALIB_FILE_NAME}) ---")
                    print("  エリアロックと筆オフセットが復元されました。")
                else:
                    is_brush_calibrated = False 
                    print(f"--- キャリブレーションデータを読み込みました ({CALIB_FILE_NAME}) ---")
                    print("  エリアロックが復元されました。(MediaPipeモードのためオフセットは無視)")
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
    """Loads lens distortion correction data for Top-Down camera from .npz"""
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
                
                new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                
                TOP_CAM_MAP_X, TOP_CAM_MAP_Y = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
                
                TOP_CAM_MTX = mtx 
                TOP_CAM_DIST = dist 
                
                print(f"--- レンズ歪み補正データを読み込みました ({file_path}) ---")
        except Exception as e:
            print(f"エラー: レンズ歪み補正データの読み込みに失敗しました: {e}")
    else:
        print(f"--- レンズ歪み補正ファイルが見つかりません ({file_path}) ---")
        print("    歪み補正なしで続行します。")

def record_data(event_type, timestamp, pressure, pen_pos_norm):
    """データを整形してリストに追加する"""
    """Formats and adds data to the list"""
    global last_cell_id, last_record_time, stroke_count

    if RECORDING_MODE == "Time" and event_type == 'move':
        if timestamp - last_record_time < SAMPLING_INTERVAL:
            return # 間引く (Thin out)
    
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


# --- 5. メイン処理 (Main process) ---
# 5.1. カメラの選択 (Select cameras)
top_cam_index = select_camera_index("Top-Down (X/Y) Camera")
if top_cam_index is None: sys.exit("Top-Downカメラが選択されませんでした。")
cap_top = cv2.VideoCapture(top_cam_index)
cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
top_w = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
top_h = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Topカメラ解像度: {top_w} x {top_h}")

side_cam_index = select_camera_index("Side-View (Z/Pressure) Camera")
if side_cam_index is None: sys.exit("Side-Viewカメラが選択されませんでした。")
cap_side = cv2.VideoCapture(side_cam_index)
cap_side.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_side.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Sideカメラ解像度: {cap_side.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap_side.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# 5.1b. 追跡する手の選択 (Select target hand)
TARGET_HAND = select_target_hand()

# 5.1c. 追跡モードの選択 (Select tracking mode)
TRACKING_MODE = select_tracking_mode()

# 5.1d. 記録モードの選択 (Select recording mode)
RECORDING_MODE = select_recording_mode()

# 5.1e. NPZキャリブレーションデータの自動読み込み (Auto-load NPZ calibration data)
load_calibration_data()

# 5.1f. NPZレンズ歪み補正データの自動読み込み (Auto-load NPZ lens correction data)
load_lens_calibration(LENS_CALIB_FILE_NAME, (top_w, top_h))

# 5.2. Sideカメラのキャリブレーション (Calibrate Side camera)
if not calibrate_pressure_range(cap_side, TARGET_HAND):
    sys.exit("キャリブレーションがキャンセルされました。")

# 5.3. メインループの準備 (Prepare for main loop)
dst_pts = np.float32([[0, 0], [WARPED_SIZE, 0], [WARPED_SIZE, WARPED_SIZE], [0, WARPED_SIZE]])
print("\n--- トラッキング開始 (Tracking Start) ---")
print(f"★ 追跡モード: {TRACKING_MODE} | 記録モード: {RECORDING_MODE} ★")
print(" [l] キー: 記録エリアのロック / アンロック")
if TRACKING_MODE != "MediaPipeOnly":
    print(" [o] キー: 筆オフセット調整 (エリアロック後に実行)")
else:
    print(" [o] キー: (MediaPipeモードでは不要です)")
print(" [s] キー: 記録セッションの開始/停止")
print(" [c] キー: 完成した作品を撮影 (エリアロック後に実行)")
print(" [p] キー: 記録の手動一時停止/再開 (セッション中のみ)")
print(" [k] キー: 現在のエリアロックとオフセットを .npz に保存")
print(" [w] キー: ★ 現在のデータを名前を付けて保存 (記録停止中のみ)")
print(" [q] キー: 終了してCSV保存")

print("\n★ 推奨手順 (Recommended procedure):")
if TRACKING_MODE != "MediaPipeOnly":
    print(" (1) [l] -> (2) [o] -> (3) [s] ... [s] -> (4) [w]で保存 -> (5) [s] ...")
else:
    print(" (1) [l] -> (2) [s] ... [s] -> (3) [w]で保存 -> (4) [s] ...")
print("★ 撮影は [s]停止中 に [c] で行えます。")
print("★ 終了は [q] です (保存していないデータは 'unpitsu_data_...' に保存されます)")


while True:
    # 5.4. 両方のカメラからフレームを取得 (Get frames from both cameras)
    ret_top, frame_top = cap_top.read()
    ret_side, frame_side = cap_side.read()
    if not ret_top or not ret_side:
        print("エラー: カメラフレームを読み込めません")
        break
    
    # Topカメラのレンズ歪み補正 (Top camera lens distortion correction)
    if TOP_CAM_MAP_X is not None and TOP_CAM_MAP_Y is not None:
        frame_top = cv2.remap(frame_top, TOP_CAM_MAP_X, TOP_CAM_MAP_Y, cv2.INTER_LINEAR)
    
    current_time = time.time()
    
    # 5.5. Topカメラの処理 (ArUco 範囲検出) (Top camera processing - ArUco area detection)
    (corners_top, ids_top, _) = DETECTOR.detectMarkers(frame_top)
    if ids_top is not None:
        aruco.drawDetectedMarkers(frame_top, corners_top, ids_top)

    src_pts = [get_marker_point(id, ids_top, corners_top) for id in CORNER_IDS]
    
    if all(pt is not None for pt in src_pts):
        src_pts_np = np.float32(src_pts)
        M_live = cv2.getPerspectiveTransform(src_pts_np, dst_pts) 
        cv2.polylines(frame_top, [src_pts_np.astype(int)], True, (0, 255, 0), 2)
    else:
        M_live = None 
    
    M_to_use = M_locked if is_area_locked else M_live

    # 5.5b. 筆マーカーの検出 (Detect brush marker)
    brush_marker_corners_pixel = None
    brush_marker_center_pixel = None
    brush_marker_xaxis_pixel = None
    brush_marker_yaxis_pixel = None

    if ids_top is not None and (TRACKING_MODE == "Hybrid" or TRACKING_MODE == "MarkerOnly"):
        for i, marker_id in enumerate(ids_top.flatten()):
            if marker_id == BRUSH_MARKER_ID:
                brush_marker_corners_pixel = corners_top[i][0].astype(int)
                brush_marker_center_pixel = brush_marker_corners_pixel.mean(axis=0).astype(int)
                brush_marker_xaxis_pixel = brush_marker_corners_pixel[1] - brush_marker_corners_pixel[0]
                brush_marker_yaxis_pixel = brush_marker_corners_pixel[3] - brush_marker_corners_pixel[0]
                cv2.polylines(frame_top, [brush_marker_corners_pixel], True, (255, 255, 0), 2)
                break

    # 5.6. Sideカメラの処理 (筆圧Z軸の検出) (Side camera processing - Z-axis pressure detection)
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
        else: 
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

    # 5.7. Topカメラの処理 (X/Y座標検出) - モード分岐
    # Top camera processing (X/Y coordinate detection) - Mode branching
    pen_pos_norm = None
    finger_pos_norm = None
    marker_pos_norm = None
    
    if M_to_use is not None:
        
        # --- [A] MediaPipe（指）の座標を計算 ---
        if TRACKING_MODE == "MediaPipeOnly" or TRACKING_MODE == "Hybrid":
            frame_top_rgb = cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB)
            results_top = hands_top.process(frame_top_rgb)
            hand_landmarks_top = get_target_hand_landmarks(results_top, TARGET_HAND) 
            
            if hand_landmarks_top: 
                h_top, w_top, _ = frame_top.shape
                landmark_top = hand_landmarks_top.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] 
                pen_center_pixel = (int(landmark_top.x * w_top), int(landmark_top.y * h_top))
                pen_pixel_np = np.float32([[pen_center_pixel]])
                pen_pos_norm_raw = cv2.perspectiveTransform(pen_pixel_np, M_to_use)
                finger_pos_norm = (pen_pos_norm_raw[0][0][0], pen_pos_norm_raw[0][0][1])
                cv2.circle(frame_top, tuple(pen_center_pixel), 8, (0, 0, 255), -1) # 指 = 赤 (Finger = Red)

        # --- [B] 筆マーカーの座標を計算 ---
        if (TRACKING_MODE == "MarkerOnly" or TRACKING_MODE == "Hybrid"):
            if is_brush_calibrated and brush_marker_center_pixel is not None:
                local_x, local_y = BRUSH_TIP_OFFSET_LOCAL
                tip_vector_pixel = (brush_marker_xaxis_pixel * local_x) + (brush_marker_yaxis_pixel * local_y)
                pen_tip_pixel = (brush_marker_center_pixel + tip_vector_pixel).astype(int)
                pen_pixel_np = np.float32([[pen_tip_pixel]])
                pen_pos_norm_raw = cv2.perspectiveTransform(pen_pixel_np, M_to_use)
                marker_pos_norm = (pen_pos_norm_raw[0][0][0], pen_pos_norm_raw[0][0][1])
                cv2.circle(frame_top, tuple(pen_tip_pixel), 8, (255, 100, 0), -1) # マーカー = 青 (Marker = Blue)

        # --- [C] 最終的な pen_pos_norm を決定 ---
        if TRACKING_MODE == "Hybrid":
            pen_pos_norm = marker_pos_norm if marker_pos_norm is not None else finger_pos_norm
        elif TRACKING_MODE == "MarkerOnly":
            pen_pos_norm = marker_pos_norm
        elif TRACKING_MODE == "MediaPipeOnly":
            pen_pos_norm = finger_pos_norm

        if pen_pos_norm is not None:
            last_pen_pos_norm = pen_pos_norm

    # 5.8. 状態機械 (State Machine) - 記録モード分岐
    if is_recording_session and not is_manually_paused and last_pen_pos_norm is not None:
        
        if is_touching_now and not is_pen_down:
            # --- 状態：Pen Down (触れた瞬間) ---
            is_pen_down = True
            stroke_count += 1
            print(f"Stroke {stroke_count} START (Down) - Pressure: {current_pressure_level}")
            record_data('down', current_time, current_pressure_level, last_pen_pos_norm)
        
        elif is_touching_now and is_pen_down:
            # --- 状態：Pen Move (触れ続けている) ---
            
            if RECORDING_MODE == "Time":
                record_data('move', current_time, current_pressure_level, last_pen_pos_norm)
            
            elif RECORDING_MODE == "Spatial":
                (norm_x, norm_y) = last_pen_pos_norm
                current_cell_id = get_cell_id(norm_x, norm_y)
                if current_cell_id != last_cell_id:
                    record_data('move', current_time, current_pressure_level, last_pen_pos_norm)
            
        elif not is_touching_now and is_pen_down:
            # --- 状態：Pen Up (離れた瞬間) ---
            is_pen_down = False
            print(f"Stroke {stroke_count} END (Up)")
            record_data('up', current_time, 0, last_pen_pos_norm)

    # 5.9. 画面表示 (Screen display)
    status_text = "RECORDING" if is_recording_session else "STOPPED"
    color = (0, 0, 255) if is_recording_session else (100, 100, 100) 

    if is_recording_session and is_manually_paused: 
        status_text = "MANUALLY PAUSED"
        color = (0, 255, 255) 
    elif M_live is None and not is_area_locked: 
        status_text = "AREA NOT FOUND"
        color = (100, 100, 100)
    
    cv2.putText(frame_top, f"STATUS: {status_text}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame_top, f"T-MODE: {TRACKING_MODE} (Hand: {TARGET_HAND})", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame_top, f"R-MODE: {RECORDING_MODE}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if is_area_locked:
        area_status_text = "AREA LOCKED"
        area_color = (0, 255, 0) # 緑 (Green)
    else:
        area_status_text = "AREA UNLOCKED"
        area_color = (0, 165, 255) # オレンジ (Orange)
    cv2.putText(frame_top, area_status_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, area_color, 2) 

    if is_area_locked and not is_brush_calibrated and (TRACKING_MODE != "MediaPipeOnly"): 
        cv2.putText(frame_top, "CALIBRATE BRUSH OFFSET (Press 'o')", (20, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    elif not is_area_locked and M_live is None:
         cv2.putText(frame_top, "Find Area Markers (0,1,2,3) to Lock", (20, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)


    cv2.imshow("Top-Down View (X/Y)", frame_top)
    cv2.imshow("Side View (Z/Pressure)", frame_side)

    # 5.10. キー入力 (Key input)
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

    # ★★★ 'w' キーの処理 (新規追加) ★★★
    if key == ord('w'):
        if is_recording_session:
            print("エラー: 's'キーで記録を停止してから保存してください。")
        else:
            # 映像を一時停止し、コンソールで名前を入力させて保存する
            # This pauses the video feed to prompt for a name in the console
            save_data_with_prompt()

    if key == ord('l'):
        if is_recording_session:
            print("エラー: 記録セッション中はエリアロックを変更できません。")
        else:
            if not is_area_locked:
                if M_live is not None:
                    M_locked = M_live
                    is_area_locked = True
                    print("--- エリアをロックしました ---")
                    if TRACKING_MODE != "MediaPipeOnly":
                        print("★ 次に [o] キーで筆のオフセットを調整してください。")
                    else:
                        print("★ [s] キーで記録を開始できます。")
                else:
                    print("エラー: 4隅のマーカーが認識されていません。ロックできません。")
            else:
                M_locked = None
                is_area_locked = False
                is_brush_calibrated = False 
                print("--- エリアのロックを解除しました (筆オフセットもリセットされました) ---")

    if key == ord('o'):
        if TRACKING_MODE == "MediaPipeOnly":
            print("--- MediaPipeOnlyモードでは、筆オフセット調整 [o] は不要です。 ---")
            continue
            
        if is_recording_session:
            print("エラー: オフセット調整は記録セッション開始前に行ってください。")
        elif not is_area_locked:
            print("エラー: 'l'キーでエリアをロックしてからオフセット調整を行ってください。")
        elif M_to_use is None: 
            print("エラー: エリアがロックされていません。")
        elif brush_marker_center_pixel is None:
            print(f"エラー: 筆マーカー (ID={BRUSH_MARKER_ID}) が認識できません。")
        elif src_pts[0] is None: 
            print("エラー: 左上マーカー (ID=0) が認識できません。")
        else:
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

    if key == ord('c'):
        if is_recording_session:
            print("エラー: 撮影は記録セッションを停止（[s]キー）してから行ってください。")
        elif not is_area_locked or M_locked is None:
            print("エラー: 撮影するには、まず [l] キーでエリアをロックしてください。")
        else:
            print("--- 作品画像を撮影します... ---")
            frame_to_capture = frame_top.copy() 
            warped_image = cv2.warpPerspective(frame_to_capture, M_locked, (WARPED_SIZE, WARPED_SIZE))
            output_filename = CAPTURE_IMAGE_FILENAME # 'calligraphy_image.png'
            try:
                cv2.imwrite(output_filename, warped_image)
                print(f"--- 成功: {output_filename} として保存しました。 ---")
                print(f"    (次に analyze_calligraphy.py を実行できます)")
            except Exception as e:
                print(f"エラー: 画像の保存に失敗しました: {e}")

    if key == ord('s'):
        if not is_recording_session:
            if not is_area_locked:
                print("エラー: 'l'キーでエリアをロックしてください。")
            elif not is_brush_calibrated and (TRACKING_MODE != "MediaPipeOnly"):
                print("エラー: 'o'キーで筆のオフセット調整を先に行ってください。")
            else:
                is_recording_session = True
                print("--- 記録セッション開始 ---")
        else:
            is_recording_session = False
            is_pen_down = False 
            is_manually_paused = False 
            print("--- 記録セッション停止 ---")

# --- 6. 終了処理 (Cleanup) ---
cap_top.release()
cap_side.release()
hands_top.close()
hands_side.close()
cv2.destroyAllWindows()
save_all_data()