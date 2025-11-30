import cv2
import cv2.aruco as aruco
import numpy as np
import sys
import time

# --- 設定項目 (Settings) ---
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR_PARAMS.polygonalApproxAccuracyRate = 0.05
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

CORNER_IDS = [0, 1, 2, 3] 
CORNER_INDEX_MAP = { 0: 2, 1: 3, 2: 0, 3: 1 } 

WARPED_SIZE = 800       # 出力画像のサイズ (800x800)

# --- ヘルパー関数 ---
def get_marker_point(target_id, detected_ids, detected_corners):
    """検出されたマーカーリストから、指定されたIDの「基準点」ピクセル座標を取得する"""
    if detected_ids is not None:
        for i, marker_id in enumerate(detected_ids.flatten()):
            if marker_id == target_id:
                if target_id in CORNER_INDEX_MAP:
                    corners = detected_corners[i][0]
                    corner_index = CORNER_INDEX_MAP[target_id]
                    point = corners[corner_index]
                    return point.astype(int)
    return None

# --- メイン処理 ---
def main():
    # 1. カメラの初期化 (環境に合わせて 0 または 1 に変更)
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"エラー: カメラ(Index: {camera_index})が開けません。")
        sys.exit()

    print(f"--- マーカー範囲撮影ツール (名前指定版) ---")
    print(f"解像度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(" [s] キー: 現在の範囲を撮影 -> ファイル名を入力して保存")
    print(" [l] キー: エリア検出をロック/解除")
    print(" [q] キー: 終了")

    dst_pts = np.float32([[0, 0], [WARPED_SIZE, 0], [WARPED_SIZE, WARPED_SIZE], [0, WARPED_SIZE]])
    
    M_matrix = None         
    M_locked = None         
    is_locked = False       

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. マーカー検出
        (corners, ids, _) = DETECTOR.detectMarkers(frame)
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        # 3. 4点の座標取得と行列計算
        src_pts = [get_marker_point(id, ids, corners) for id in CORNER_IDS]
        
        if all(pt is not None for pt in src_pts):
            src_pts_np = np.float32(src_pts)
            M_matrix = cv2.getPerspectiveTransform(src_pts_np, dst_pts)
            if not is_locked:
                cv2.polylines(frame, [src_pts_np.astype(int)], True, (0, 255, 0), 2)
        else:
            if not is_locked:
                M_matrix = None

        current_M = M_locked if is_locked else M_matrix

        # 4. 画面表示
        status_text = "Status: SEARCHING"
        color = (0, 0, 255) 

        if is_locked:
            status_text = "Status: LOCKED (Ready)"
            color = (0, 255, 0) 
            cv2.putText(frame, "LOCKED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        elif M_matrix is not None:
            status_text = "Status: DETECTED"
            color = (0, 255, 255) 

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Marker Area Capture", frame)

        # 5. キー操作
        key = cv2.waitKey(1) & 0xFF

        # [s] 保存 (Save) - ここを変更しました
        if key == ord('s'):
            if current_M is not None:
                # 射影変換を実行
                warped_img = cv2.warpPerspective(frame, current_M, (WARPED_SIZE, WARPED_SIZE))
                
                # ★ コンソールで入力を待つ
                print("\n" + "="*40)
                print("【撮影】 画像をキャプチャしました。")
                filename = input(">> 保存するファイル名を入力してください (例: tehon_01): ").strip()
                
                # 空エンターならデフォルト名、拡張子がなければ .png を付与
                if not filename:
                    timestamp = int(time.time())
                    filename = f"capture_{timestamp}.png"
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filename += ".png"
                
                try:
                    cv2.imwrite(filename, warped_img)
                    print(f"-> 保存完了: {filename}")
                except Exception as e:
                    print(f"-> 保存エラー: {e}")
                
                print("="*40 + "\n")
                
                # 直後に保存した画像を表示
                cv2.imshow("Last Captured", warped_img)
            else:
                print("エラー: 領域が認識されていません。[l]でロックするかマーカーを全て映してください。")

        # [l] ロック (Lock)
        if key == ord('l'):
            if is_locked:
                is_locked = False
                M_locked = None
                print("ロック解除")
            else:
                if M_matrix is not None:
                    is_locked = True
                    M_locked = M_matrix
                    print("領域をロックしました。")
                else:
                    print("エラー: マーカー認識不能のためロック不可。")

        # [q] 終了
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()