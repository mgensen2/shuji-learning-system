import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import os
import mediapipe as mp
import sys

# --- 設定 ---
# 録画したセッション名のベース (例: "recordings/session_20231027_123456")
SESSION_BASE_PATH = "" 

# 解析パラメータ
WARPED_SIZE = 800
GRID_SIZE = 8
COORD_LIMIT = 200.0
CORRECTION_SEARCH_RADIUS = 50 

# --- 初期化 ---
mp_hands = mp.solutions.hands
hands_top = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_side = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)
CORNER_IDS = [0, 1, 2, 3]
CORNER_INDEX_MAP = { 0: 2, 1: 3, 2: 0, 3: 1 }

# --- ヘルパー関数 ---
def get_marker_point(detected_ids, detected_corners, target_id):
    if detected_ids is None: return None
    for i, marker_id in enumerate(detected_ids.flatten()):
        if marker_id == target_id:
            return detected_corners[i][0][CORNER_INDEX_MAP[target_id]].astype(int)
    return None

def select_corners_manually(image):
    """マウスクリックで4隅を指定させる"""
    print("\n--- 手動エリア指定モード ---")
    print("ウィンドウ上で 4つの角 を順番にクリックしてください:")
    print("1. 左上 (Top-Left)")
    print("2. 右上 (Top-Right)")
    print("3. 右下 (Bottom-Right)")
    print("4. 左下 (Bottom-Left)")
    
    points = []
    img_display = image.copy()
    window_name = "Select 4 Corners (TL -> TR -> BR -> BL)"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, img_display)
            print(f"Click {len(points)}: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, img_display)

    while len(points) < 4:
        if cv2.waitKey(1) & 0xFF == 27: # ESCで中止
            sys.exit("手動指定がキャンセルされました。")
    
    cv2.destroyAllWindows()
    print("--- 4隅が設定されました ---")
    return points

def get_transform_matrix(frame):
    """フレームからArUcoマーカーを探し、失敗したら手動指定に移行して射影変換行列Mを返す"""
    src_pts = []
    
    # 1. 自動検出を試みる
    corners, ids, _ = DETECTOR.detectMarkers(frame)
    if ids is not None:
        src_pts = [get_marker_point(ids, corners, id) for id in CORNER_IDS]
    
    # 2. 失敗した場合 (Noneが含まれる場合) は手動指定へ
    if not src_pts or any(pt is None for pt in src_pts):
        print("警告: マーカーの自動検出に失敗しました。")
        user_input = input("手動で範囲を指定しますか？ (y/n): ").strip().lower()
        if user_input == 'y':
            selected_points = select_corners_manually(frame)
            src_pts = [np.array(pt) for pt in selected_points]
        else:
            return None
    
    src_pts_np = np.float32(src_pts)
    dst_pts = np.float32([[0, 0], [WARPED_SIZE, 0], [WARPED_SIZE, WARPED_SIZE], [0, WARPED_SIZE]])
    return cv2.getPerspectiveTransform(src_pts_np, dst_pts)

def create_ink_distance_map(final_img_path, M):
    """
    完成画像を読み込み、歪み補正し、「一番近い墨までの距離とベクトル」を持つマップを作成する
    """
    if not os.path.exists(final_img_path):
        print("エラー: 完成画像が見つかりません")
        return None, None

    img = cv2.imread(final_img_path)
    # 歪み補正
    warped = cv2.warpPerspective(img, M, (WARPED_SIZE, WARPED_SIZE))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # 2値化 (Otsu) - 黒が0、白が255
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 距離変換 (Distance Transform)
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    return binary, dist_map

def analyze_session():
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = input("セッションのベースパスを入力 (例: recordings/session_2023...): ").strip()

    top_video = f"{base_path}_top.mp4"
    side_video = f"{base_path}_side.mp4"
    final_image = f"{base_path}_final.png"
    
    if not os.path.exists(top_video): sys.exit("動画ファイルが見つかりません")

    print("--- 解析を開始します ---")
    
    # 1. 完成画像から変換行列Mと「墨の距離マップ」を作成
    print("1. 完成画像を解析中...")
    img_final = cv2.imread(final_image)
    if img_final is None: sys.exit("完成画像の読み込みに失敗しました。")

    M_locked = get_transform_matrix(img_final)
    if M_locked is None: sys.exit("エラー: エリアを特定できませんでした。")
    
    ink_binary, ink_dist_map = create_ink_distance_map(final_image, M_locked)
    
    # 2. 動画を読み込み
    cap_top = cv2.VideoCapture(top_video)
    cap_side = cv2.VideoCapture(side_video)
    total_frames = int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT))
    
    csv_data = []
    stroke_id = 0
    is_pen_down = False
    frame_idx = 0
    
    print(f"2. 動画解析中 (全 {total_frames} フレーム)...")
    
    while True:
        ret_top, frame_top = cap_top.read()
        ret_side, frame_side = cap_side.read()
        if not ret_top or not ret_side: break
        
        frame_idx += 1
        if frame_idx % 100 == 0: print(f"   Processing frame {frame_idx}/{total_frames}...")

        # --- MediaPipe処理 (Top) ---
        frame_top_rgb = cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB)
        res_top = hands_top.process(frame_top_rgb)
        
        # --- MediaPipe処理 (Side) -> 簡易筆圧 ---
        frame_side_rgb = cv2.cvtColor(frame_side, cv2.COLOR_BGR2RGB)
        res_side = hands_side.process(frame_side_rgb)
        
        raw_pressure = 0
        if res_side.multi_hand_landmarks:
            # 簡易的に親指Y座標を筆圧とする（キャリブレーションなしの相対値）
            # 本格的にやるならここでZ軸キャリブレーション値を適用
            thumb_side = res_side.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
            raw_pressure = thumb_side.y # 下に行くほど大きい
        
        # --- 座標計算と補正 ---
        if res_top.multi_hand_landmarks:
            thumb_top = res_top.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
            h, w, _ = frame_top.shape
            px, py = int(thumb_top.x * w), int(thumb_top.y * h)
            
            # 射影変換 (ピクセル -> 800x800平面)
            pen_point = np.array([[[px, py]]], dtype=np.float32)
            warped_point = cv2.perspectiveTransform(pen_point, M_locked)
            wx, wy = warped_point[0][0] # Warped X, Y
            
            wx_int, wy_int = int(wx), int(wy)
            
            # --- ★★★ ここが補正の核心 ★★★ ---
            final_x, final_y = wx, wy
            correction_flag = 0
            
            # 座標がエリア内にあるか確認
            if 0 <= wx_int < WARPED_SIZE and 0 <= wy_int < WARPED_SIZE:
                dist_to_ink = ink_dist_map[wy_int, wx_int]
                
                # もし「墨の上(距離0)」でなければ、近くの墨を探す
                if dist_to_ink > 0 and dist_to_ink < CORRECTION_SEARCH_RADIUS:
                    # 距離マップの勾配を使って、最も近い墨の方へ座標をずらす
                    # (簡易実装: 周囲を探索して一番 dist が小さいピクセルを探す)
                    
                    min_d = dist_to_ink
                    best_dx, best_dy = 0, 0
                    
                    # 近傍探索 (処理速度優先で範囲を絞る)
                    search_r = int(dist_to_ink) + 2
                    for dy in range(-search_r, search_r+1, 2): # 2px刻みで高速化
                        for dx in range(-search_r, search_r+1, 2):
                            ny, nx = wy_int + dy, wx_int + dx
                            if 0 <= ny < WARPED_SIZE and 0 <= nx < WARPED_SIZE:
                                d = ink_dist_map[ny, nx]
                                if d < min_d:
                                    min_d = d
                                    best_dx, best_dy = dx, dy
                    
                    # 補正適用
                    final_x += best_dx
                    final_y += best_dy
                    correction_flag = 1 # 補正したフラグ
            
            # --- データ記録 ---
            # 筆圧によるDown/Up判定 (簡易的な閾値)
            # 必要に応じて調整してください
            PRESSURE_THRESHOLD = 0.5 # 画面半分より下なら描画とみなす例
            
            is_touching = (raw_pressure > PRESSURE_THRESHOLD)
            
            event_type = 'move'
            if is_touching and not is_pen_down:
                is_pen_down = True
                stroke_id += 1
                event_type = 'down'
            elif not is_touching and is_pen_down:
                is_pen_down = False
                event_type = 'up'
            elif not is_touching:
                event_type = 'hover' # 書いていない移動
            
            if event_type != 'hover':
                # カスタム座標系 (-200 ~ 0) に変換
                norm_x_01 = final_x / WARPED_SIZE
                norm_y_01 = final_y / WARPED_SIZE
                custom_x = (norm_x_01 - 1.0) * COORD_LIMIT
                custom_y = norm_y_01 * -COORD_LIMIT
                
                csv_data.append({
                    'timestamp': frame_idx / 30.0, # 30fps仮定
                    'event_type': event_type,
                    'stroke_id': stroke_id,
                    'x': f"{custom_x:.2f}",
                    'y': f"{custom_y:.2f}",
                    'pressure': f"{raw_pressure:.4f}",
                    'cell_id': -1, # 必要なら計算
                    'corrected': correction_flag # 補正されたかどうかのフラグ
                })

    # CSV保存
    csv_path = f"{base_path}_analyzed.csv"
    if csv_data:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"--- 解析完了: {csv_path} に保存しました ---")
    else:
        print("データがありませんでした。")

    cap_top.release()
    cap_side.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_session()