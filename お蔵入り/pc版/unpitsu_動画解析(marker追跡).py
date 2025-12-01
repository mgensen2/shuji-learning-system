import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import os
import sys

# --- 設定 ---
WARPED_SIZE = 800
COORD_LIMIT = 200.0
CORRECTION_SEARCH_RADIUS = 50 # 補正探索範囲(px)

# ArUco
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)
CORNER_IDS = [0, 1, 2, 3]
CORNER_INDEX_MAP = { 0: 2, 1: 3, 2: 0, 3: 1 }

# 色設定用
TARGET_COLOR_LOWER = None
TARGET_COLOR_UPPER = None

def pick_color_click(event, x, y, flags, param):
    global TARGET_COLOR_LOWER, TARGET_COLOR_UPPER
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_hsv = param
        pixel = frame_hsv[y, x]
        
        h_sens, s_sens, v_sens = 10, 50, 80
        TARGET_COLOR_LOWER = np.array([max(0, pixel[0]-h_sens), max(50, pixel[1]-s_sens), max(50, pixel[2]-v_sens)])
        TARGET_COLOR_UPPER = np.array([min(180, pixel[0]+h_sens), 255, 255])
        print(f"Tracking Color Set: {pixel}")

def get_marker_point(detected_ids, detected_corners, target_id):
    if detected_ids is None: return None
    for i, marker_id in enumerate(detected_ids.flatten()):
        if marker_id == target_id:
            return detected_corners[i][0][CORNER_INDEX_MAP[target_id]].astype(int)
    return None

def get_transform_matrix(frame):
    corners, ids, _ = DETECTOR.detectMarkers(frame)
    src_pts = [get_marker_point(ids, corners, id) for id in CORNER_IDS]
    if any(pt is None for pt in src_pts): return None
    
    src_pts_np = np.float32(src_pts)
    dst_pts = np.float32([[0, 0], [WARPED_SIZE, 0], [WARPED_SIZE, WARPED_SIZE], [0, WARPED_SIZE]])
    return cv2.getPerspectiveTransform(src_pts_np, dst_pts)

def track_color(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 20:
            M = cv2.moments(c)
            if M["m00"] != 0:
                return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
    return None

def analyze():
    global TARGET_COLOR_LOWER
    
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = input("セッションパスを入力 (recordings/session_...): ").strip()

    top_video = f"{base_path}_top.mp4"
    side_video = f"{base_path}_side.mp4"
    final_image = f"{base_path}_final.png" # 録画ツールでは保存していない場合は別途用意
    
    # 1. 色設定 (動画の最初のフレームから)
    cap = cv2.VideoCapture(top_video)
    ret, first_frame = cap.read()
    if not ret: sys.exit("動画が開けません")
    
    print("--- 色設定 ---")
    print("表示されたウィンドウで、追跡するマーカーをクリックしてください。")
    print("決定したらキーを押してください。")
    
    hsv_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("Pick Color")
    cv2.setMouseCallback("Pick Color", pick_color_click, hsv_first)
    
    while TARGET_COLOR_LOWER is None:
        cv2.imshow("Pick Color", first_frame)
        if cv2.waitKey(10) == 27: sys.exit()
    cv2.destroyWindow("Pick Color")
    
    # 2. 完成画像から距離マップ作成 (補正用)
    print("完成画像を解析中...")
    # 完成画像がない場合は、動画の最終フレームを使うなどのフォールバックも可能だが今回は必須とする
    if not os.path.exists(final_image):
        print("警告: 完成画像(_final.png)が見つかりません。動画の最終フレームを使用しますか？(y/n)")
        if input() == 'y':
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-5)
            _, final_frame = cap.read()
            img_final = final_frame
        else:
            return
    else:
        img_final = cv2.imread(final_image)

    M_locked = get_transform_matrix(img_final)
    if M_locked is None:
        print("エラー: マーカー(0-3)が見つかりません。")
        return

    warped_final = cv2.warpPerspective(img_final, M_locked, (WARPED_SIZE, WARPED_SIZE))
    gray_final = cv2.cvtColor(warped_final, cv2.COLOR_BGR2GRAY)
    _, bin_final = cv2.threshold(gray_final, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 距離マップ (黒い部分への距離)
    dist_map = cv2.distanceTransform(bin_final, cv2.DIST_L2, 5)

    # 3. 解析実行
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 最初に戻す
    cap_side = cv2.VideoCapture(side_video)
    
    csv_data = []
    stroke_id = 0
    is_down = False
    frame_idx = 0
    
    print("解析中...")
    
    while True:
        ret, frame = cap.read()
        ret_s, frame_s = cap_side.read()
        if not ret: break
        
        frame_idx += 1
        
        # Top: XY座標
        pt = track_color(frame, TARGET_COLOR_LOWER, TARGET_COLOR_UPPER)
        
        # Side: 簡易筆圧 (色のY座標)
        # ※Side動画の色設定も必要だが、簡略化のためTopと同じ色か、
        #   もしくはSideは固定位置のボール等を追うならロジック追加が必要。
        #   ここでは「Topと同じ色のマーカーがSideでも見える」と仮定。
        pt_side = track_color(frame_s if ret_s else frame, TARGET_COLOR_LOWER, TARGET_COLOR_UPPER)
        pressure = pt_side[1] if pt_side else 0 # 簡易値 (Y座標)
        
        if pt:
            # 射影変換
            src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, M_locked)
            wx, wy = int(dst[0][0][0]), int(dst[0][0][1])
            
            # ★ 吸着補正 ★
            final_x, final_y = wx, wy
            corrected = 0
            
            if 0 <= wx < WARPED_SIZE and 0 <= wy < WARPED_SIZE:
                d = dist_map[wy, wx]
                # 墨の上(0)でなく、かつ近くに墨がある場合
                if d > 0 and d < CORRECTION_SEARCH_RADIUS:
                    # 近傍探索して距離0(墨)または最小の場所へ移動
                    # (簡易実装: 周囲を探索)
                    min_local = d
                    best_dx, best_dy = 0, 0
                    r = int(d) + 2
                    for dy in range(-r, r+1, 2):
                        for dx in range(-r, r+1, 2):
                            ny, nx = wy+dy, wx+dx
                            if 0<=ny<WARPED_SIZE and 0<=nx<WARPED_SIZE:
                                if dist_map[ny, nx] < min_local:
                                    min_local = dist_map[ny, nx]
                                    best_dx, best_dy = dx, dy
                    final_x += best_dx
                    final_y += best_dy
                    corrected = 1
            
            # 座標変換
            nx = (final_x / WARPED_SIZE - 1.0) * COORD_LIMIT
            ny = (final_y / WARPED_SIZE) * -COORD_LIMIT
            
            # Down/Up判定 (簡易)
            # 実際にはSideカメラのキャリブレーション値が必要だが、動画解析では相対変化を見る等で対応
            is_touching = (pt_side is not None) # 色が見えていればタッチとみなす(仮)
            
            evt = 'move'
            if is_touching and not is_down:
                is_down = True
                stroke_id += 1
                evt = 'down'
            elif not is_touching and is_down:
                is_down = False
                evt = 'up'
            
            if is_touching:
                csv_data.append({
                    'timestamp': frame_idx/30.0,
                    'event_type': evt,
                    'stroke_id': stroke_id,
                    'x': f"{nx:.2f}",
                    'y': f"{ny:.2f}",
                    'pressure': pressure,
                    'corrected': corrected
                })

    # 保存
    out_csv = f"{base_path}_color_analyzed.csv"
    if csv_data:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"完了: {out_csv}")

    cap.release()
    cap_side.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze()