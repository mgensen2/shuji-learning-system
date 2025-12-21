import cv2
import cv2.aruco as aruco
import numpy as np
import sys
import time
import os
import glob

# --- 設定項目 (Settings) ---
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
DETECTOR_PARAMS = aruco.DetectorParameters()
DETECTOR_PARAMS.polygonalApproxAccuracyRate = 0.05
DETECTOR = aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

CORNER_IDS = [0, 1, 2, 3] 
CORNER_INDEX_MAP = { 0: 2, 1: 3, 2: 0, 3: 1 } 

WARPED_SIZE = 800       # 出力画像のサイズ (800x800)
FILE_PREFIX = "img_"    # 保存ファイル名の接頭辞

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

def setup_save_directory():
    """保存先フォルダの指定と作成、開始番号の取得を行う"""
    print("\n" + "="*50)
    folder_name = input(">> 保存先フォルダ名を入力してください (例: data01): ").strip()
    
    # 空入力の場合はデフォルトフォルダ
    if not folder_name:
        folder_name = "captured_images"
        print(f"   (空入力のため '{folder_name}' を使用します)")

    # フォルダ作成
    if not os.path.exists(folder_name):
        try:
            os.makedirs(folder_name)
            print(f"   -> 新規フォルダ作成: {folder_name}")
        except Exception as e:
            print(f"   -> フォルダ作成エラー: {e}")
            return None, 0
    else:
        print(f"   -> 既存フォルダを使用: {folder_name}")

    # 次の連番を取得 (既存ファイルをスキャン)
    # img_*.png のようなファイルを探して最大の番号+1を見つける
    search_pattern = os.path.join(folder_name, f"{FILE_PREFIX}*.png")
    existing_files = glob.glob(search_pattern)
    
    max_num = 0
    for f_path in existing_files:
        # ファイル名から番号部分を抽出
        base_name = os.path.basename(f_path) # img_001.png
        try:
            # "img_" と ".png" を除いて数値化
            num_part = base_name.replace(FILE_PREFIX, "").replace(".png", "")
            num = int(num_part)
            if num > max_num:
                max_num = num
        except ValueError:
            continue

    next_idx = max_num + 1
    print(f"   -> 次のファイル名: {FILE_PREFIX}{next_idx:03d}.png から開始します")
    print("="*50 + "\n")
    
    return folder_name, next_idx

# --- メイン処理 ---
def main():
    # 0. 保存設定の初期化
    current_folder, current_index = setup_save_directory()
    if current_folder is None:
        print("フォルダ設定に失敗したため終了します。")
        sys.exit()

    # 1. カメラの初期化
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    
    # 解像度設定 (カメラが対応していない場合は無視されます)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"エラー: カメラ(Index: {camera_index})が開けません。")
        sys.exit()

    print(f"--- マーカー範囲撮影ツール (連番保存版) ---")
    print(" [s] キー: 現在のフォルダに連番で即座に保存")
    print(" [n] キー: 保存フォルダを変更 (New Folder)")
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

        # 4. 画面表示情報
        status_text = "SEARCHING"
        color = (0, 0, 255) 
        folder_info = f"Folder: {current_folder} | Next: {current_index:03d}"

        if is_locked:
            status_text = "LOCKED"
            color = (0, 255, 0) 
            cv2.putText(frame, "LOCKED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        elif M_matrix is not None:
            status_text = "DETECTED"
            color = (0, 255, 255) 

        # ステータスと現在の保存先情報を画面に表示
        cv2.putText(frame, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, folder_info, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.imshow("Marker Area Capture", frame)

        # 5. キー操作
        key = cv2.waitKey(1) & 0xFF

        # [s] 保存 (Save) - 連番で即保存
        if key == ord('s'):
            if current_M is not None:
                # 射影変換を実行
                warped_img = cv2.warpPerspective(frame, current_M, (WARPED_SIZE, WARPED_SIZE))
                
                # ファイル名の生成
                filename = f"{FILE_PREFIX}{current_index:03d}.png"
                save_path = os.path.join(current_folder, filename)
                
                try:
                    cv2.imwrite(save_path, warped_img)
                    print(f"Saved: {save_path}")
                    
                    # 成功したらカウントアップ
                    current_index += 1
                    
                    # 保存した瞬間を少しフィードバック（画面を一瞬白くするなど簡易エフェクトの代わりにコンソール出力）
                    # 保存した画像を表示ウィンドウに出す
                    cv2.imshow("Last Captured", warped_img)
                    
                except Exception as e:
                    print(f"Save Error: {e}")
            else:
                print("エラー: 領域認識なし。保存できません。")

        # [n] フォルダ変更 (New Folder)
        if key == ord('n'):
            print("\n--- フォルダ変更モード ---")
            # OpenCVウィンドウがフリーズしないよう一旦閉じるか、ユーザーに入力を促す
            # input()はブロッキングするため、入力中は映像が止まります
            new_folder, new_index = setup_save_directory()
            if new_folder is not None:
                current_folder = new_folder
                current_index = new_index

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

        # [q] 終了
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()