import cv2
import numpy as np
import pandas as pd
import os
import sys

# --- 設定項目 ---
# 画像ファイル
IMG_SAMPLE = 'calligraphy_sample.png'
IMG_USER   = 'calligraphy_image.png'

# CSVファイル
CSV_SAMPLE = 'sample_data.csv'
CSV_USER   = 'unpitsu_data_corrected.csv'

# パラメータ
WARPED_SIZE = 800
COORD_LIMIT = 200.0

# --- 共通関数 (省略なしで再定義) ---
def convert_from_custom_coords(x, y):
    norm_x_01 = (x / COORD_LIMIT) + 1.0
    norm_y_01 = y / -COORD_LIMIT
    px = norm_x_01 * WARPED_SIZE
    py = norm_y_01 * WARPED_SIZE
    return int(round(px)), int(round(py))

def render_csv_trace(df):
    canvas = np.zeros((WARPED_SIZE, WARPED_SIZE), dtype=np.uint8)
    last_pos = None
    is_drawing = False
    for _, row in df.iterrows():
        px, py = convert_from_custom_coords(row['x'], row['y'])
        thickness = int(row.get('pressure', 4) * 2) + 2
        if row['event_type'] == 'down':
            is_drawing = True
            last_pos = (px, py)
            cv2.circle(canvas, (px, py), thickness//2, 255, -1)
        elif row['event_type'] == 'move' and is_drawing and last_pos:
            cv2.line(canvas, last_pos, (px, py), 255, thickness)
            last_pos = (px, py)
        elif row['event_type'] == 'up':
            is_drawing = False
            last_pos = None
    return canvas

def calculate_iou(img1, img2):
    intersection = cv2.bitwise_and(img1, img2)
    union = cv2.bitwise_or(img1, img2)
    if cv2.countNonZero(union) == 0: return 0.0
    return cv2.countNonZero(intersection) / cv2.countNonZero(union)

def load_and_binarize_image(filepath):
    if not os.path.exists(filepath): return None
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (WARPED_SIZE, WARPED_SIZE))
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary

def main():
    print("\n========== 書道 総合判定 (画像 + CSV) ==========")
    
    # 1. 画像評価パート
    print("1. 画像の一致率を計算中...")
    bin_img_sample = load_and_binarize_image(IMG_SAMPLE)
    bin_img_user   = load_and_binarize_image(IMG_USER)
    
    score_img = 0.0
    if bin_img_sample is not None and bin_img_user is not None:
        score_img = calculate_iou(bin_img_sample, bin_img_user)
        print(f"   画像スコア: {score_img * 100:.1f}点")
    else:
        print("   (エラー: 画像ファイルが不足しているためスキップ)")

    # 2. CSV評価パート
    print("2. CSVデータの一致率を計算中...")
    score_csv = 0.0
    if os.path.exists(CSV_SAMPLE) and os.path.exists(CSV_USER):
        df_sample = pd.read_csv(CSV_SAMPLE)
        df_user   = pd.read_csv(CSV_USER)
        
        trace_sample = render_csv_trace(df_sample)
        trace_user   = render_csv_trace(df_user)
        
        score_csv = calculate_iou(trace_sample, trace_user)
        print(f"   データスコア: {score_csv * 100:.1f}点")
    else:
        print("   (エラー: CSVファイルが不足しているためスキップ)")

    # 3. 総合評価
    # 重み付け: 画像(結果)とCSV(過程)を半々で評価
    total_score = (score_img + score_csv) / 2.0
    
    print("\n------------------------------")
    print(f"★ 総合評価スコア: {total_score * 100:.1f} 点")
    print("------------------------------")
    
    if total_score > 0.85: grade = "免許皆伝 (Master)"
    elif total_score > 0.7: grade = "大変よくできました (Excellent)"
    elif total_score > 0.5: grade = "よくできました (Good)"
    else: grade = "がんばりましょう (Keep practicing)"
    
    print(f"判定: {grade}")
    print("\nヒント:")
    if score_img > score_csv + 0.1:
        print("・「形」は綺麗ですが、「筆の運び（書き順や勢い）」がお手本と少し違います。")
    elif score_csv > score_img + 0.1:
        print("・「筆の運び」は正しいですが、最終的な「形（太さや位置）」が少しズレています。")
    else:
        print("・形と筆運びのバランスが良いです。")

if __name__ == "__main__":
    main()