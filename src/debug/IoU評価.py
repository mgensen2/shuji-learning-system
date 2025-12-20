import cv2
import numpy as np
import os
import glob
import pandas as pd

# --- 設定 ---
TARGET_IMG_PATH = 'target_sample.png'  # お手本画像
DATA_DIR = './student_data/'           # 被験者画像のフォルダ
OUTPUT_CSV = 'iou_results_centered.csv' # 結果保存名

def get_centered_image(img_bin):
    """
    二値画像(白文字・黒背景)を受け取り、
    文字の重心(または外接矩形の中心)が画像の中心に来るように平行移動する。
    """
    # 文字部分（白画素）の座標を取得
    coords = cv2.findNonZero(img_bin)
    if coords is None:
        return img_bin # 文字がない場合はそのまま

    # 外接矩形（バウンディングボックス）を取得
    x, y, w, h = cv2.boundingRect(coords)
    
    # 文字の中心座標
    center_x = x + w // 2
    center_y = y + h // 2
    
    # 画像の中心座標
    h_img, w_img = img_bin.shape
    img_center_x = w_img // 2
    img_center_y = h_img // 2
    
    # 移動量を計算
    shift_x = img_center_x - center_x
    shift_y = img_center_y - center_y
    
    # アフィン変換行列（平行移動）
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # 画像を移動させる
    shifted_img = cv2.warpAffine(img_bin, M, (w_img, h_img), borderValue=0)
    
    return shifted_img

def calculate_iou(img1_path, img2_path):
    # 1. 画像読み込み
    img1 = cv2.imread(img1_path, 0) # お手本
    img2 = cv2.imread(img2_path, 0) # 被験者

    if img1 is None or img2 is None:
        return None

    # 2. サイズ合わせ
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 3. 二値化（自動閾値決定）
    # ※ここでは「白背景・黒文字」を前提に、反転して「黒背景・白文字」にします
    _, bin1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bin2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- 【追加】 ここで位置合わせを行う ---
    bin1 = get_centered_image(bin1)
    bin2 = get_centered_image(bin2)
    # ------------------------------------

    # 4. IoU計算
    intersection = cv2.bitwise_and(bin1, bin2)
    union = cv2.bitwise_or(bin1, bin2)

    area_intersection = cv2.countNonZero(intersection)
    area_union = cv2.countNonZero(union)

    if area_union == 0:
        return 0.0
    
    return area_intersection / area_union

# --- メイン処理 ---
results = []
print("Starting IoU calculation with auto-centering...")

files = glob.glob(os.path.join(DATA_DIR, "*.*"))
for file_path in files:
    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    file_name = os.path.basename(file_path)
    # print(f"Processing: {file_name}") # 進捗が見たければコメントアウト解除
    
    score = calculate_iou(TARGET_IMG_PATH, file_path)
    
    if score is not None:
        results.append([file_name, score])

# 保存
df = pd.DataFrame(results, columns=['FileName', 'IoU'])
df.to_csv(OUTPUT_CSV, index=False)
print(f"完了しました。保存先: {OUTPUT_CSV}")