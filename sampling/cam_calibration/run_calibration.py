import cv2
import numpy as np
import glob
import os

SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
DICTIONARY_NAME = cv2.aruco.DICT_4X4_100

# --- ボードの定義 (ステップ1と同一) ---
dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_NAME)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)

# --- 検出パラメータ ---
params = cv2.aruco.DetectorParameters()

# --- ★★★ 修正点 1: Detector オブジェクトを作成 ★★★ ---
detector = cv2.aruco.Detector(dictionary, params)
# --- 画像の読み込み (ステップ2で保存した場所) ---
IMG_DIR = "calibration_images"
images = glob.glob(os.path.join(IMG_DIR, '*.png'))

if not images:
    print(f"エラー: '{IMG_DIR}' に画像が見つかりません。")
    print("ステップ2 (2_collect_images.py) を実行して画像を収集してください。")
    exit()

print(f"{len(images)} 枚の画像を検出しました。キャリブレーションを開始します...")

all_corners = []
all_ids = []
img_size = None # 画像サイズを保持

# 各画像からマーカーとコーナーを検出
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"警告: {fname} を読み込めません。スキップします。")
        continue

    if img_size is None:
        img_size = img.shape[:2][::-1] # (width, height)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. ArUcoマーカーの検出
    corners, ids, rejected = detector.detectMarkers(gray)

    # 2. ChArUcoコーナーの検出（サブピクセル精度）
    if ids is not None and len(ids) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray,
            board
        )
        
        # 検出が成功した場合、データを保存
        if ret and charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
        else:
            print(f"警告: {fname} で十分なChArUcoコーナーを検出できませんでした。")
    else:
        print(f"警告: {fname} でArUcoマーカーを検出できませんでした。")

if not all_corners:
    print("エラー: どの画像からもコーナーを検出できませんでした。")
    exit()

print("コーナーの検出が完了。カメラパラメータを計算します...")

# --- キャリブレーションの実行 ---
try:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners,  # 検出したコーナーの画像座標
        all_ids,      # 対応するID
        board,        # ボードオブジェクト
        img_size,     # 画像サイズ
        None,         # cameraMatrix (初期値)
        None          # distCoeffs (初期値)
    )

    if ret:
        print("\n--- キャリブレーション成功 ---")
        
        print("\nカメラ行列 (Camera Matrix):")
        print(camera_matrix)
        
        print("\n歪み係数 (Distortion Coefficients):")
        print(dist_coeffs)
        
        print(f"\n再投影誤差 (Reprojection Error): {ret}")

        # --- 結果の保存 ---
        output_filename = "camera_calibration.npz"
        np.savez(output_filename,
                 cameraMatrix=camera_matrix,
                 distCoeffs=dist_coeffs,
                 rvecs=rvecs,
                 tvecs=tvecs,
                 reprojectionError=ret)
        print(f"\nキャリブレーション結果を '{output_filename}' に保存しました。")

    else:
        print("\n--- キャリブレーション失敗 ---")

except cv2.error as e:
    print(f"\n--- キャリブレーション中にエラーが発生しました ---")
    print(e)
    print("検出されたコーナーの数が少なすぎるか、画像の品質に問題がある可能性があります。")