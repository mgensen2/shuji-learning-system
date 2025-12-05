import cv2
import cv2.aruco as aruco
import numpy as np
import sys
import os
import time

# --- 1. 設定項目 ---
# ArUcoマーカー設定
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR_PARAMS.polygonalApproxAccuracyRate = 0.05
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

CORNER_IDS = [0, 1, 2, 3] 
CORNER_INDEX_MAP = { 0: 2, 1: 3, 2: 0, 3: 1 } 

WARPED_SIZE = 800       
LENS_CALIB_FILE_NAME = 'top_camera_lens.npz' 

# 状態変数
M_live = None       
M_locked = None     
is_area_locked = False

# レンズ補正用変数
TOP_CAM_MTX = None
TOP_CAM_DIST = None
TOP_CAM_MAP_X = None
TOP_CAM_MAP_Y = None

# --- 2. ヘルパー関数 ---

def select_camera_index(prompt_text):
    """カメラを選択させる"""
    print(f"--- {prompt_text} のカメラを選択 ---")
    available_indices = []
    for i in range(10):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            available_indices.append(i)
            cap_test.release()
    
    if not available_indices:
        print("利用可能なカメラが見つかりません。")
        return None

    if len(available_indices) == 1:
        print(f"カメラ {available_indices[0]} を自動選択しました。")
        return available_indices[0]

    print(f"利用可能なインデックス: {available_indices}")
    selected_index = None
    
    for index in available_indices:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened(): continue
        print(f"--- カメラ {index} をテスト中 ---")
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.putText(frame, f"Index: {index} ({prompt_text})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, "Use this? (y/n)", (10, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Camera Selection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                selected_index = index
                break
            if key == ord('n'):
                break
        cap.release()
        if selected_index is not None:
            break
    cv2.destroyAllWindows()
    return selected_index

def get_marker_point(target_id, detected_ids, detected_corners):
    """検出されたマーカーリストから指定IDの座標を取得"""
    if detected_ids is not None:
        for i, marker_id in enumerate(detected_ids.flatten()):
            if marker_id == target_id:
                if target_id in CORNER_INDEX_MAP:
                    corners = detected_corners[i][0]
                    corner_index = CORNER_INDEX_MAP[target_id]
                    point = corners[corner_index]
                    return point.astype(int)
    return None

def load_lens_calibration(file_path, frame_size_wh):
    """レンズ歪み補正データを読み込む"""
    global TOP_CAM_MTX, TOP_CAM_DIST, TOP_CAM_MAP_X, TOP_CAM_MAP_Y
    if os.path.exists(file_path):
        try:
            with np.load(file_path) as data:
                mtx = data['mtx']
                dist = data['dist']
                w, h = frame_size_wh
                new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                TOP_CAM_MAP_X, TOP_CAM_MAP_Y = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
                print(f"レンズ補正データを適用: {file_path}")
        except Exception as e:
            print(f"レンズ補正データの読み込み失敗: {e}")

# --- 3. メイン処理 ---

def main():
    global M_live, M_locked, is_area_locked

    cam_index = select_camera_index("撮影用カメラ")
    if cam_index is None: return

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    load_lens_calibration(LENS_CALIB_FILE_NAME, (w, h))

    dst_pts = np.float32([[0, 0], [WARPED_SIZE, 0], [WARPED_SIZE, WARPED_SIZE], [0, WARPED_SIZE]])

    print("\n--- 撮影ツール起動 ---")
    print(" [l] キー: エリアのロック / 解除")
    print(" [s] キー: ★名前を付けて保存")
    print(" [q] キー: 終了")

    while True:
        ret, frame_raw = cap.read()
        if not ret: break

        if TOP_CAM_MAP_X is not None and TOP_CAM_MAP_Y is not None:
            frame = cv2.remap(frame_raw, TOP_CAM_MAP_X, TOP_CAM_MAP_Y, cv2.INTER_LINEAR)
        else:
            frame = frame_raw.copy()

        display_frame = frame.copy()

        corners, ids, _ = DETECTOR.detectMarkers(frame)
        if ids is not None:
            aruco.drawDetectedMarkers(display_frame, corners, ids)

        src_pts = [get_marker_point(id, ids, corners) for id in CORNER_IDS]

        if all(pt is not None for pt in src_pts):
            src_pts_np = np.float32(src_pts)
            M_live = cv2.getPerspectiveTransform(src_pts_np, dst_pts)
            cv2.polylines(display_frame, [src_pts_np.astype(int)], True, (0, 255, 0), 2)
        else:
            M_live = None

        M_current = M_locked if is_area_locked else M_live

        status_text = "Area: LOCKED" if is_area_locked else ("Area: DETECTED" if M_live is not None else "Area: SEARCHING...")
        color = (0, 255, 0) if (is_area_locked or M_live is not None) else (0, 0, 255)
        
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display_frame, "[s]:Save Name  [l]:Lock/Unlock  [q]:Quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Capture Tool", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('l'):
            if is_area_locked:
                is_area_locked = False
                M_locked = None
                print("-> エリアロック解除")
            else:
                if M_live is not None:
                    M_locked = M_live
                    is_area_locked = True
                    print("-> エリアロック完了")
                else:
                    print("エラー: マーカーが認識されていないためロックできません。")

        if key == ord('s'):
            if M_current is not None:
                # 切り出し処理
                warped = cv2.warpPerspective(frame, M_current, (WARPED_SIZE, WARPED_SIZE))
                
                # --- ここでユーザーに入力を求める ---
                print("\n" + "="*40)
                print("【保存モード】 コンソールでファイル名を入力してください。")
                print(" ※ 空白でEnterキーを押すと、日時がファイル名になります。")
                
                user_input = input(">> ファイル名 (拡張子省略可): ").strip()
                
                if not user_input:
                    # 入力が空ならタイムスタンプを使用
                    filename_full = f"calligraphy_{int(time.time())}.png"
                else:
                    # 拡張子補完
                    if not user_input.lower().endswith(('.png', '.jpg', '.jpeg')):
                        filename_full = user_input + ".png"
                    else:
                        filename_full = user_input
                
                try:
                    cv2.imwrite(filename_full, warped)
                    print(f"★ 保存しました: {filename_full}")
                    print("="*40 + "\n")
                    
                    # 保存画像を少し長く表示して確認しやすくする
                    cv2.imshow("Saved Image", warped)
                    cv2.waitKey(1500) 
                except Exception as e:
                    print(f"保存エラー: {e}")
            else:
                print("エラー: エリアが特定されていません。")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()