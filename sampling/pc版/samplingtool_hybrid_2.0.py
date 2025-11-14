import cv2
import cv2.aruco as aruco
import numpy as np
import time
import csv
import os
import sys
import mediapipe as mp

# --- 1. 設定項目 (Settings) ---
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR_PARAMS.polygonalApproxAccuracyRate = 0.05
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

CORNER_IDS = [0, 1, 2, 3] 
CORNER_INDEX_MAP = { 0: 2, 1: 3, 2: 0, 3: 1 } 

# ★ マーカー追跡は使用しないためコメントアウト
# BRUSH_MARKER_ID = 50 

CALIB_FILE_NAME = 'unpitsu_calibration.npz'
LENS_CALIB_FILE_NAME = 'top_camera_lens.npz' 
CAPTURE_IMAGE_FILENAME = 'calligraphy_image.png' 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_top = mp.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_side = mp.hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

COORD_LIMIT = 200.0
GRID_SIZE = 8
SAMPLING_INTERVAL = 0.05 
WARPED_SIZE = 800       
CELL_SIZE_PX = WARPED_SIZE / GRID_SIZE

# ★★★ 新規追加：画像補正設定 ★★★
IMAGE_CORRECTION_INTERVAL = 0.5 # 画像処理による軌跡補正の間隔 (秒)
IMAGE_CORRECTION_THRESHOLD = 100 # 墨として検出する2値化のしきい値 (0-255, 低いほど濃い黒)

# --- 2. データ格納リスト (Data storage lists) ---
drawing_data = []     
cell_transitions = [] 

# --- 3. 状態変数 (State variables) ---
is_recording_session = False
is_pen_down = False         
is_manually_paused = False  
is_area_locked = False        
stroke_count = 0
last_cell_id = -1             
last_record_time = 0
last_pen_pos_norm = None 
M_live = None                 
M_locked = None               
M_side_live = None            
M_side_locked = None          
TARGET_HAND = "Any"
TRACKING_MODE = "MediaPipeOnly" # ★ MediaPipeOnly に固定
RECORDING_MODE = "Time"       

Z_TOUCH_HEIGHT = -1     
Z_PRESS_MAX_HEIGHT = -1 
is_z_calibrated = False 

# ★ MediaPipeOnlyモードなので、筆オフセットは常に不要 (is_brush_calibrated = True 扱い)
BRUSH_TIP_OFFSET_LOCAL = None 
is_brush_calibrated = True # ★ MediaPipeOnlyモードでは自動的に True 扱い

TOP_CAM_MTX = None
TOP_CAM_DIST = None
TOP_CAM_MAP_X = None
TOP_CAM_MAP_Y = None

# ★★★ 新規追加：画像補正用の状態変数 ★★★
last_image_proc_time = 0    # 最後に画像処理を実行した時間
last_binary_image = None    # 最後に処理した2値化画像 (基準)
# ★ 5.8 (move) が参照するための、マスク適用済みの最新2値化画像
last_binary_image_masked = None 


# --- 4. ヘルパー関数 (Helper functions) ---
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

# ★★★ 削除: select_tracking_mode() は不要になったため削除 ★★★

def select_recording_mode():
    """ユーザーに記録モードを選択させる"""
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

def get_marker_point(target_id, detected_ids, detected_corners):
    """
    検出されたマーカーリストから、指定されたIDの「基準点」ピクセル座標を取得する
    """
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
    norm_x_01 = norm_x / WARPED_SIZE
    norm_y_01 = norm_y / WARPED_SIZE
    converted_x = (norm_x_01 - 1.0) * COORD_LIMIT
    converted_y = norm_y_01 * -COORD_LIMIT
    return converted_x, converted_y

def get_cell_id(norm_x, norm_y):
    """(0,0)-(WARPED_SIZE, WARPED_SIZE) の座標からグリッドID (0-63) を計算"""
    cell_x = int(norm_x // CELL_SIZE_PX)
    cell_y = int(norm_y // CELL_SIZE_PX)
    cell_x = max(0, min(cell_x, GRID_SIZE - 1))
    cell_y = max(0, min(cell_y, GRID_SIZE - 1))
    return (cell_y * GRID_SIZE) + cell_x

def save_all_data():
    """プログラム終了時にデータをCSVに保存"""
    global drawing_data, cell_transitions
    
    if not drawing_data:
        print("--- 終了処理: 保存すべきデータはありません ---")
        return

    print(f"--- 終了処理: 残りの {len(drawing_data)} 件のデータを保存します... ---")
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
    
    drawing_data = []
    cell_transitions = []

def save_data_with_prompt():
    """
    ユーザーにファイル名を入力させ、現在のデータを保存し、内部データをクリアする
    """
    global drawing_data, cell_transitions, stroke_count
    
    if not drawing_data:
        print("--- 保存するデータがありません ---")
        return
    
    print("\n--- CSV保存 ---")
    print("!!! OpenCVウィンドウは一時停止します。このコンソールを見てください。!!!")
    
    temp_drawing_data = drawing_data.copy()
    temp_cell_transitions = cell_transitions.copy()
    
    filename_base = input("CSVのベース名を入力してください (例: my_writing_session): ").strip()
    
    if not filename_base:
        filename_base = f"unpitsu_data_{int(time.time())}"
        print(f"名前が入力されなかったので、デフォルト名を使用します: {filename_base}")

    full_csv_path = f"{filename_base}_full.csv"
    transitions_csv_path = f"{filename_base}_transitions.csv"

    try:
        headers = temp_drawing_data[0].keys()
        with open(full_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(temp_drawing_data)
        print(f"--- 成功: {full_csv_path} として保存しました。 ({len(temp_drawing_data)} 件)")
    except Exception as e:
        print(f"エラー: {full_csv_path} の保存に失敗しました: {e}")

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

    drawing_data = []
    cell_transitions = []
    stroke_count = 0
    print("--- 内部データをクリアしました。新しい記録を開始できます。 ---")
    print("\n--- OpenCVウィンドウに戻ります ---")


def save_calibration_data():
    """現在のキャリブレーションデータを.npzに保存する"""
    global is_z_calibrated
    
    if not is_area_locked:
        print("エラー: エリアがロックされていません。 [l]キーでロックしてください。")
        return

    is_z_calibrated = (Z_TOUCH_HEIGHT != -1 and Z_PRESS_MAX_HEIGHT != -1)
    if not is_z_calibrated:
         print("エラー: 筆圧が調整されていません。 [z]キーと[m]キーで調整してください。")
         return
         
    offset_to_save = np.array((0.0, 0.0)) 

    try:
        np.savez(CALIB_FILE_NAME,
                 M_locked=M_locked,
                 M_side_locked=M_side_locked,
                 BRUSH_TIP_OFFSET_LOCAL=offset_to_save, 
                 Z_TOUCH_HEIGHT=np.array(Z_TOUCH_HEIGHT),
                 Z_PRESS_MAX_HEIGHT=np.array(Z_PRESS_MAX_HEIGHT)
                )
        print(f"--- キャリブレーションデータを保存しました ({CALIB_FILE_NAME}) ---")
        print("  (Top/Sideエリアロック, 筆圧設定)")
    except Exception as e:
        print(f"エラー: キャリブレーションデータの保存に失敗しました: {e}")

def load_calibration_data():
    """起動時にキャリブレーションデータを読み込む"""
    global M_locked, M_side_locked, BRUSH_TIP_OFFSET_LOCAL, is_area_locked, is_brush_calibrated
    global Z_TOUCH_HEIGHT, Z_PRESS_MAX_HEIGHT, is_z_calibrated
    
    if os.path.exists(CALIB_FILE_NAME):
        try:
            with np.load(CALIB_FILE_NAME) as data:
                M_locked = data['M_locked']
                M_side_locked = data['M_side_locked']
                is_area_locked = True
                
                Z_TOUCH_HEIGHT = data['Z_TOUCH_HEIGHT'].item()
                Z_PRESS_MAX_HEIGHT = data['Z_PRESS_MAX_HEIGHT'].item()
                is_z_calibrated = (Z_TOUCH_HEIGHT != -1 and Z_PRESS_MAX_HEIGHT != -1)

                print(f"--- キャリブレーションデータを読み込みました ({CALIB_FILE_NAME}) ---")
                print("  Top/Sideエリアロック、筆圧設定を復元しました。")

                is_brush_calibrated = True 

        except Exception as e:
            print(f"エラー: キャリブレーションデータの読み込みに失敗しました: {e}")
            M_locked = None
            M_side_locked = None
            is_area_locked = False
            is_z_calibrated = False
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
    global last_cell_id, last_record_time, stroke_count

    if RECORDING_MODE == "Time" and event_type == 'move':
        if timestamp - last_record_time < SAMPLING_INTERVAL:
            return 
    
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


def get_warped_binary_image(frame, matrix, threshold):
    """
    Topカメラのフレームを射影変換し、2値化して返す
    """
    if matrix is None:
        return None
    
    warped_img = cv2.warpPerspective(frame, matrix, (WARPED_SIZE, WARPED_SIZE))
    gray_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    # 墨(暗)を255(白)、紙(明)を0(黒)にする
    _, binary_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return binary_img

def get_hand_mask(hand_landmarks, frame_shape_hw, matrix, out_size):
    """
    MediaPipeの手のランドマークから凸包(Convex Hull)を作成し、
    射影変換して800x800のマスク画像を生成する。
    """
    if hand_landmarks is None or matrix is None:
        return np.zeros((out_size, out_size), dtype=np.uint8) 

    h, w = frame_shape_hw
    
    landmark_points_raw = []
    for landmark in hand_landmarks.landmark:
        px = int(landmark.x * w)
        py = int(landmark.y * h)
        landmark_points_raw.append([px, py])
    
    try:
        hull_points_raw = cv2.convexHull(np.array(landmark_points_raw, dtype=np.int32))
    except Exception as e:
        print(f"Warning: Hand hull calculation failed: {e}")
        return np.zeros((out_size, out_size), dtype=np.uint8)

    hull_points_raw_float = np.float32(hull_points_raw).reshape(-1, 1, 2)
    warped_hull_points = cv2.perspectiveTransform(hull_points_raw_float, matrix)
    
    warped_hull_points_int = warped_hull_points.astype(np.int32)
    
    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    cv2.fillPoly(mask, [warped_hull_points_int], (255)) 
    
    return mask


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

# 5.1c. 追跡モードは MediaPipeOnly に固定
print(f"\n--- XY軸 追跡モード ---")
print(f"-> [MediaPipeのみ] モードに固定されました。")
print(f"-> (軌跡は{IMAGE_CORRECTION_INTERVAL}秒ごとに画像処理で補正されます)")
TRACKING_MODE = "MediaPipeOnly"
is_brush_calibrated = True # オフセット調整は不要

# 5.1d. 記録モードの選択 (Select recording mode)
RECORDING_MODE = select_recording_mode()

# 5.1e. NPZキャリブレーションデータの自動読み込み (Auto-load NPZ calibration data)
load_calibration_data()

# 5.1f. NPZレンズ歪み補正データの自動読み込み (Auto-load NPZ lens correction data)
load_lens_calibration(LENS_CALIB_FILE_NAME, (top_w, top_h))

# 5.3. メインループの準備 (Prepare for main loop)
dst_pts = np.float32([[0, 0], [WARPED_SIZE, 0], [WARPED_SIZE, WARPED_SIZE], [0, WARPED_SIZE]])
print("\n--- トラッキング開始 (Tracking Start) ---")
print(f"★ 追跡モード: {TRACKING_MODE} (画像補正ON) | 記録モード: {RECORDING_MODE} ★")
print(" [l] キー: エリア/平面ロック (Top/Sideカメラ両方でマーカー(0-3)を検出)")
print(" [o] キー: (MediaPipeモードでは不要です)")
print(" [z] キー: ★ 筆圧 0 (タッチ) の設定 (エリアロック後に実行)")
print(" [m] キー: ★ 筆圧 8 (最大) の設定 (エリアロック後に実行)")
print(" [s] キー: 記録セッションの開始/停止 (全キャリブレーション後に実行)")
print(" [c] キー: 完成した作品を撮影 (エリアロック後に実行)")
print(" [p] キー: 記録の手動一時停止/再開 (セッション中のみ)")
print(" [k] キー: 現在の全キャリブレーションを .npz に保存")
print(" [w] キー: 現在のデータを名前を付けて保存 (記録停止中のみ)")
print(" [q] キー: 終了してCSV保存")

print("\n★ 推奨手順 (Recommended procedure):")
print(" (1) [l] -> (2) [z] -> (3) [m] -> (4) [s] ...")
print("★ 終了時: [s] -> [w]で保存 -> [q]")


while True:
    # 5.4. 両方のカメラからフレームを取得 (Get frames from both cameras)
    ret_top, frame_top_raw = cap_top.read() 
    ret_side, frame_side = cap_side.read()
    if not ret_top or not ret_side:
        print("エラー: カメラフレームを読み込めません")
        break
    
    if TOP_CAM_MAP_X is not None and TOP_CAM_MAP_Y is not None:
        frame_top = cv2.remap(frame_top_raw, TOP_CAM_MAP_X, TOP_CAM_MAP_Y, cv2.INTER_LINEAR)
    else:
        frame_top = frame_top_raw 
    
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

    # 5.5b. MediaPipe(Top) の処理を 5.7/5.8b で共有するため、ここで実行
    frame_top_rgb = cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB)
    results_top = hands_top.process(frame_top_rgb)
    hand_landmarks_top = get_target_hand_landmarks(results_top, TARGET_HAND) 

    # 5.6. Sideカメラの処理 (筆圧Z軸の検出)
    is_touching_now = False 
    current_pressure_level = 0 
    pen_y_side_warped = -1 
    
    (corners_side, ids_side, _) = DETECTOR.detectMarkers(frame_side)
    if ids_side is not None:
        aruco.drawDetectedMarkers(frame_side, corners_side, ids_side)
    
    src_pts_side = [get_marker_point(id, ids_side, corners_side) for id in CORNER_IDS]
    
    if all(pt is not None for pt in src_pts_side):
        src_pts_side_np = np.float32(src_pts_side)
        M_side_live = cv2.getPerspectiveTransform(src_pts_side_np, dst_pts) 
        cv2.polylines(frame_side, [src_pts_side_np.astype(int)], True, (0, 255, 0), 2)
    else:
        M_side_live = None

    M_side_to_use = M_side_locked if is_area_locked else M_side_live
    
    frame_side_display = frame_side 

    if M_side_to_use is not None:
        frame_side_warped = cv2.warpPerspective(frame_side, M_side_to_use, (WARPED_SIZE, WARPED_SIZE))
        frame_side_display = frame_side_warped 
        
        frame_side_warped_rgb = cv2.cvtColor(frame_side_warped, cv2.COLOR_BGR2RGB)
        results_side_warped = hands_side.process(frame_side_warped_rgb)
        hand_landmarks_side = get_target_hand_landmarks(results_side_warped, TARGET_HAND)
        
        if hand_landmarks_side:
            h_side_w, w_side_w, _ = frame_side_warped.shape 
            
            landmark = hand_landmarks_side.landmark[mp_hands.HandLandmark.THUMB_TIP] 
            
            pen_y_side_warped = int(landmark.y * h_side_w) 
            
            if is_z_calibrated: 
                if pen_y_side_warped < Z_TOUCH_HEIGHT:
                    is_touching_now = False
                    current_pressure_level = 0
                elif pen_y_side_warped >= Z_PRESS_MAX_HEIGHT:
                    is_touching_now = True
                    current_pressure_level = 8
                else: 
                    is_touching_now = True
                    touch_range = float(Z_PRESS_MAX_HEIGHT - Z_TOUCH_HEIGHT)
                    current_depth = float(pen_y_side_warped - Z_TOUCH_HEIGHT)
                    if touch_range > 0: 
                        normalized_pressure = current_depth / touch_range
                        current_pressure_level = int(round(normalized_pressure * 8))
                    else:
                        current_pressure_level = 0 
            
            px_side_warped = int(landmark.x * w_side_w)
            color = (0, 0, 255) if is_touching_now else (0, 255, 0)
            cv2.circle(frame_side_display, (px_side_warped, pen_y_side_warped), 8, color, -1)
            cv2.putText(frame_side_display, f"Pressure: {current_pressure_level}", (px_side_warped + 10, pen_y_side_warped), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if is_z_calibrated:
        h_disp, w_disp, _ = frame_side_display.shape
        cv2.line(frame_side_display, (0, Z_TOUCH_HEIGHT), (w_disp, Z_TOUCH_HEIGHT), (0, 255, 255), 2)
        cv2.putText(frame_side_display, f"P=0 Y:{Z_TOUCH_HEIGHT}", (10, Z_TOUCH_HEIGHT - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.line(frame_side_display, (0, Z_PRESS_MAX_HEIGHT), (w_disp, Z_PRESS_MAX_HEIGHT), (0, 165, 255), 2)
        cv2.putText(frame_side_display, f"P=8 Y:{Z_PRESS_MAX_HEIGHT}", (10, Z_PRESS_MAX_HEIGHT - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)


    # 5.7. Topカメラの処理 (X/Y座標検出) - MediaPipeOnly
    pen_pos_norm = None
    
    if M_to_use is not None:
        if hand_landmarks_top: 
            h_top, w_top, _ = frame_top.shape
            landmark_top = hand_landmarks_top.landmark[mp_hands.HandLandmark.THUMB_TIP] 
            pen_center_pixel = (int(landmark_top.x * w_top), int(landmark_top.y * h_top))
            pen_pixel_np = np.float32([[pen_center_pixel]])
            pen_pos_norm_raw = cv2.perspectiveTransform(pen_pixel_np, M_to_use)
            pen_pos_norm = (pen_pos_norm_raw[0][0][0], pen_pos_norm_raw[0][0][1])
            cv2.circle(frame_top, tuple(pen_center_pixel), 8, (0, 0, 255), -1) 

        if pen_pos_norm is not None:
            last_pen_pos_norm = pen_pos_norm

    # ★★★ 5.8. 状態機械 (State Machine) - 交差判定ロジック追加 ★★★
    if is_recording_session and not is_manually_paused and last_pen_pos_norm is not None:
        
        if not is_z_calibrated:
             if is_pen_down: 
                 is_pen_down = False
                 print("--- 筆圧未調整のため Pen Up ---")
                 record_data('up', current_time, 0, last_pen_pos_norm)
        
        elif is_touching_now and not is_pen_down:
            is_pen_down = True
            stroke_count += 1
            print(f"Stroke {stroke_count} START (Down) - Pressure: {current_pressure_level}")
            record_data('down', current_time, current_pressure_level, last_pen_pos_norm)
        
        elif is_touching_now and is_pen_down:
            # --- 状態：Pen Move (触れ続けている) ---
            
            # ★★★ 交差判定ロジック (Crossing detection logic) ★★★
            event_to_record = 'move' # デフォルト (Default)
            
            # 0.5秒経過し、マスク済み基準画像が利用可能な場合
            # (If 0.5s has passed and the masked base image is available)
            if last_binary_image_masked is not None:
                (norm_x, norm_y) = last_pen_pos_norm
                px, py = convert_to_custom_coords(norm_x, norm_y)
                
                if 0 <= py < WARPED_SIZE and 0 <= px < WARPED_SIZE:
                    # 指の先端が「過去の墨」(255) の上にあるかチェック
                    # (Check if fingertip is on "past ink" (255))
                    if last_binary_image_masked[py, px] == 255:
                        event_to_record = 'crossing' # ★ イベントタイプを変更 (Change event type)
            
            # ★★★ 記録モード分岐 (Recording mode branch) ★★★
            if RECORDING_MODE == "Time":
                # event_to_record ('move' or 'crossing') を記録
                record_data(event_to_record, current_time, current_pressure_level, last_pen_pos_norm)
            
            elif RECORDING_MODE == "Spatial":
                (norm_x, norm_y) = last_pen_pos_norm
                current_cell_id = get_cell_id(norm_x, norm_y)
                if current_cell_id != last_cell_id:
                    # event_to_record ('move' or 'crossing') を記録
                    record_data(event_to_record, current_time, current_pressure_level, last_pen_pos_norm)
            
        elif not is_touching_now and is_pen_down:
            is_pen_down = False
            print(f"Stroke {stroke_count} END (Up)")
            record_data('up', current_time, 0, last_pen_pos_norm)

    # ★★★ 5.8b. 画像処理による軌跡補正 (手のマスク機能付き) ★★★
    if is_recording_session and not is_manually_paused and M_locked is not None:
        
        if current_time - last_image_proc_time > IMAGE_CORRECTION_INTERVAL:
            
            current_binary_image = get_warped_binary_image(frame_top, M_locked, IMAGE_CORRECTION_THRESHOLD)
            
            # ★ 5.8 が参照するために、現在のマスク済み画像を計算
            hand_mask = get_hand_mask(hand_landmarks_top, (top_h, top_w), M_locked, WARPED_SIZE)
            inverted_hand_mask = cv2.bitwise_not(hand_mask)
            current_binary_image_masked = cv2.bitwise_and(current_binary_image, inverted_hand_mask)

            if last_binary_image is not None and current_binary_image is not None:
                # ★ 差分計算用の 'last_binary_image' はマスクを適用しない
                # (古い 'last_binary_image' には、0.5秒前の手の影が含まれているため)
                # (We use the unmasked 'last_binary_image' for diff calculation)
                
                # (1) 0.5秒前の画像から「現在の手」をマスク (Remove "current hand" from "last image")
                last_binary_image_no_hand = cv2.bitwise_and(last_binary_image, inverted_hand_mask)
                # (2) 現在の画像から「現在の手」をマスク (Remove "current hand" from "current image")
                # (current_binary_image_masked は上で計算済み)
                
                # (3) 手を除去した画像同士で差分を計算
                diff_image = cv2.absdiff(current_binary_image_masked, last_binary_image_no_hand)
                
                kernel = np.ones((3, 3), np.uint8)
                diff_image = cv2.morphologyEx(diff_image, cv2.MORPH_OPEN, kernel, iterations=1)

                moments = cv2.moments(diff_image)
                if moments["m00"] > 10: 
                    cx_norm = int(moments["m10"] / moments["m00"]) 
                    cy_norm = int(moments["m01"] / moments["m00"]) 
                    
                    x_corr, y_corr = convert_to_custom_coords(cx_norm, cy_norm)
                    cell_id_corr = get_cell_id(cx_norm, cy_norm)
                    pressure_corr = current_pressure_level if is_touching_now else 0
                    
                    drawing_data.append({
                        'timestamp': current_time,
                        'event_type': 'correction', 
                        'stroke_id': stroke_count,
                        'x': f"{x_corr:.2f}",
                        'y': f"{y_corr:.2f}",
                        'pressure': f"{pressure_corr:.4f}",
                        'cell_id': cell_id_corr
                    })
                    print(f"--- 画像補正: 新しい軌跡をセル {cell_id_corr} に検出 ---")

            # (6) 状態を更新
            last_image_proc_time = current_time
            last_binary_image = current_binary_image # 次回のための「過去」画像
            last_binary_image_masked = current_binary_image_masked # 5.8 が参照するための「過去(マスク済)」画像
            

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
        area_color = (0, 255, 0) 
    else:
        area_status_text = "AREA UNLOCKED"
        area_color = (0, 165, 255) 
    cv2.putText(frame_top, area_status_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, area_color, 2) 

    warning_y = 190
    if not is_area_locked and M_live is None:
        cv2.putText(frame_top, "Find Area Markers (0,1,2,3) to Lock", (20, warning_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    elif is_area_locked:
        if not is_z_calibrated:
             cv2.putText(frame_top, "CALIBRATE PRESSURE (Press 'z' and 'm')", (20, warning_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)


    cv2.imshow("Top-Down View (X/Y)", frame_top)
    cv2.imshow("Side View (Z/Pressure)", frame_side_display)

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

    if key == ord('w'):
        if is_recording_session:
            print("エラー: 's'キーで記録を停止してから保存してください。")
        else:
            save_data_with_prompt()

    if key == ord('l'):
        if is_recording_session:
            print("エラー: 記録セッション中はエリアロックを変更できません。")
        else:
            if not is_area_locked:
                if M_live is None:
                    print("エラー: Topカメラで4隅のマーカーが認識されていません。")
                elif M_side_live is None:
                    print("エラー: Sideカメラで4隅のマーカーが認識されていません。")
                else:
                    M_locked = M_live
                    M_side_locked = M_side_live 
                    is_area_locked = True
                    print("--- Top/Side エリアをロックしました ---")
                    print("★ 次に [z](筆圧0), [m](筆圧8) を調整してください。")
            else:
                M_locked = None
                M_side_locked = None 
                is_area_locked = False
                is_z_calibrated = False 
                print("--- 全てのロックとキャリブレーションを解除しました ---")

    if key == ord('o'):
        print("--- MediaPipeOnlyモードでは、筆オフセット調整 [o] は不要です。 ---")
        continue
            
    if key == ord('z'):
        if is_recording_session:
            print("エラー: 筆圧調整は記録セッション開始前に行ってください。")
        elif not is_area_locked:
            print("エラー: 'l'キーでエリア/平面をロックしてから調整してください。")
        elif pen_y_side_warped == -1: 
             print("エラー: Sideカメラで指が検出されていません。")
        else:
            Z_TOUCH_HEIGHT = pen_y_side_warped
            print(f"--- 筆圧 0 (タッチ) を設定しました (補正Y={Z_TOUCH_HEIGHT}) ---")
            if Z_PRESS_MAX_HEIGHT != -1:
                is_z_calibrated = True
                print("--- 筆圧キャリブレーション 完了 ---")

    if key == ord('m'):
        if is_recording_session:
            print("エラー: 筆圧調整は記録セッション開始前に行ってください。")
        elif not is_area_locked:
            print("エラー: 'l'キーでエリア/平面をロックしてから調整してください。")
        elif pen_y_side_warped == -1: 
             print("エラー: Sideカメラで指が検出されていません。")
        elif Z_TOUCH_HEIGHT != -1 and pen_y_side_warped <= Z_TOUCH_HEIGHT:
            print(f"エラー: 最大筆圧(Y={pen_y_side_warped})は、筆圧0(Y={Z_TOUCH_HEIGHT})より大きくする必要があります。")
        else:
            Z_PRESS_MAX_HEIGHT = pen_y_side_warped
            print(f"--- 筆圧 8 (最大) を設定しました (補正Y={Z_PRESS_MAX_HEIGHT}) ---")
            if Z_TOUCH_HEIGHT != -1:
                is_z_calibrated = True
                print("--- 筆圧キャリブレーション 完了 ---")

    if key == ord('c'):
        if is_recording_session:
            print("エラー: 撮影は記録セッションを停止（[s]キー）してから行ってください。")
        elif not is_area_locked or M_locked is None:
            print("エラー: 撮影するには、まず [l] キーでエリアをロックしてください。")
        else:
            print("--- 作品画像を撮影します... ---")
            frame_to_capture = frame_top_raw.copy()
            if TOP_CAM_MAP_X is not None:
                 frame_to_capture = cv2.remap(frame_to_capture, TOP_CAM_MAP_X, TOP_CAM_MAP_Y, cv2.INTER_LINEAR)

            warped_image = cv2.warpPerspective(frame_to_capture, M_locked, (WARPED_SIZE, WARPED_SIZE))
            output_filename = CAPTURE_IMAGE_FILENAME 
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
            elif not is_z_calibrated:
                print("エラー: 'z'キーと'm'キーで筆圧を調整してください。")
            else:
                print("--- 画像処理の基準（現在の紙の状態）を取得します... ---")
                frame_top_now = frame_top 
                last_binary_image = get_warped_binary_image(frame_top_now, M_locked, IMAGE_CORRECTION_THRESHOLD)
                
                if last_binary_image is None:
                    print("エラー: 2値化画像の取得に失敗しました。")
                else:
                    hand_mask_now = get_hand_mask(hand_landmarks_top, (top_h, top_w), M_locked, WARPED_SIZE)
                    # ★ 5.8 が参照する「マスク済み基準画像」も初期化
                    last_binary_image_masked = cv2.bitwise_and(last_binary_image, cv2.bitwise_not(hand_mask_now))
                    print("--- 基準画像から現在の手をマスクしました ---")

                    last_image_proc_time = time.time()
                    is_recording_session = True
                    print("--- 記録セッション開始 (MediaPipe記録 + 画像補正) ---")
        else:
            is_recording_session = False
            is_pen_down = False 
            is_manually_paused = False 
            last_binary_image = None 
            last_binary_image_masked = None # ★ リセット
            print("--- 記録セッション停止 ---")

# --- 6. 終了処理 (Cleanup) ---
cap_top.release()
cap_side.release()
hands_top.close()
hands_side.close()
cv2.destroyAllWindows()
save_all_data()