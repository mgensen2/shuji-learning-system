import cv2
import numpy as np
import os
import sys

# --- 設定項目 ---
SAMPLE_IMAGE_FILE = 'calligraphy_sample.png'  # お手本
USER_IMAGE_FILE   = 'calligraphy_image.png'   # 書いた文字
OUTPUT_RESULT_IMG = 'result_score_view.png'   # 結果画像

PROCESS_SIZE = 800  # 処理サイズ

def calculate_iou(img1, img2):
    """IoU (一致率) を計算"""
    intersection = cv2.bitwise_and(img1, img2)
    union = cv2.bitwise_or(img1, img2)
    count_inter = cv2.countNonZero(intersection)
    count_union = cv2.countNonZero(union)
    if count_union == 0: return 0.0
    return count_inter / count_union

def main():
    print("--- Step 1: 画像一致率の判定 ---")
    
    # 画像読み込み
    if not os.path.exists(SAMPLE_IMAGE_FILE) or not os.path.exists(USER_IMAGE_FILE):
        print("エラー: 画像ファイルが見つかりません。")
        return

    # グレースケール読み込み & リサイズ
    img_sample = cv2.resize(cv2.imread(SAMPLE_IMAGE_FILE, cv2.IMREAD_GRAYSCALE), (PROCESS_SIZE, PROCESS_SIZE))
    img_user   = cv2.resize(cv2.imread(USER_IMAGE_FILE, cv2.IMREAD_GRAYSCALE), (PROCESS_SIZE, PROCESS_SIZE))

    # 2値化 (白黒反転: 書いた部分を白(255)にする)
    _, bin_sample = cv2.threshold(img_sample, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, bin_user   = cv2.threshold(img_user, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # スコア計算
    score = calculate_iou(bin_sample, bin_user)
    
    print(f"\n★ 一致率 (IoUスコア): {score * 100:.1f}%")
    
    if score > 0.8: print("評価: 秀 (Excellent)")
    elif score > 0.6: print("評価: 優 (Good)")
    else: print("評価: 要練習 (Keep trying)")

    # 結果画像の作成 (可視化)
    # 青: お手本にあって自分にない (書き不足)
    # 赤: 自分にあってお手本にない (はみ出し)
    # 黒: 一致
    
    # ベースを白にする
    result_view = np.full((PROCESS_SIZE, PROCESS_SIZE, 3), 255, dtype=np.uint8)
    
    mask_match = cv2.bitwise_and(bin_sample, bin_user)
    mask_missing = cv2.bitwise_and(bin_sample, cv2.bitwise_not(bin_user))
    mask_extra = cv2.bitwise_and(cv2.bitwise_not(bin_sample), bin_user)
    
    result_view[mask_extra == 255] = [0, 0, 255]   # 赤 (BGR)
    result_view[mask_missing == 255] = [255, 0, 0] # 青
    result_view[mask_match == 255] = [0, 0, 0]     # 黒
    
    # お手本の輪郭を緑で描画
    contours, _ = cv2.findContours(bin_sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_view, contours, -1, (0, 200, 0), 1)

    cv2.imwrite(OUTPUT_RESULT_IMG, result_view)
    print(f"\n詳細画像を保存しました: {OUTPUT_RESULT_IMG}")

if __name__ == "__main__":
    main()