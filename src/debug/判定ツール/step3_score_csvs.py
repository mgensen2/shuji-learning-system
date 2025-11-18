import cv2
import numpy as np
import pandas as pd
import os
import sys

# --- 設定項目 ---
SAMPLE_CSV_FILE = 'sample_data.csv'        # お手本のCSVデータ
USER_CSV_FILE   = 'unpitsu_data_corrected.csv' # 自分のCSVデータ（補正後の使用を推奨）
OUTPUT_COMPARE_IMG = 'result_csv_compare.png'  # 比較結果画像

# 座標変換パラメータ
WARPED_SIZE = 800
COORD_LIMIT = 200.0

def convert_from_custom_coords(x, y):
    """CSV座標 -> ピクセル座標"""
    norm_x_01 = (x / COORD_LIMIT) + 1.0
    norm_y_01 = y / -COORD_LIMIT
    px = norm_x_01 * WARPED_SIZE
    py = norm_y_01 * WARPED_SIZE
    return int(round(px)), int(round(py))

def render_csv_trace(df):
    """CSVデータから軌跡画像(2値)を生成"""
    canvas = np.zeros((WARPED_SIZE, WARPED_SIZE), dtype=np.uint8)
    last_pos = None
    is_drawing = False
    
    for _, row in df.iterrows():
        px, py = convert_from_custom_coords(row['x'], row['y'])
        # 筆圧を太さに反映 (筆圧0でも最低1pxで描画して形を見る)
        pressure = row['pressure'] if 'pressure' in row else 4
        thickness = int(pressure * 2) + 2 
        
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
    """IoU (一致率) 計算"""
    intersection = cv2.bitwise_and(img1, img2)
    union = cv2.bitwise_or(img1, img2)
    count_inter = cv2.countNonZero(intersection)
    count_union = cv2.countNonZero(union)
    if count_union == 0: return 0.0
    return count_inter / count_union

def main():
    print("--- Step 3: CSV筆跡一致率の判定 ---")
    
    if not os.path.exists(SAMPLE_CSV_FILE):
        print(f"エラー: お手本CSV ({SAMPLE_CSV_FILE}) が見つかりません。")
        return
    if not os.path.exists(USER_CSV_FILE):
        print(f"エラー: ユーザーCSV ({USER_CSV_FILE}) が見つかりません。")
        return

    # 1. CSV読み込み
    print("データを読み込んで軌跡を生成中...")
    df_sample = pd.read_csv(SAMPLE_CSV_FILE)
    df_user   = pd.read_csv(USER_CSV_FILE)

    # 2. 画像化 (ラスタライズ)
    img_sample = render_csv_trace(df_sample)
    img_user   = render_csv_trace(df_user)

    # 3. 一致率計算
    score = calculate_iou(img_sample, img_user)
    print(f"\n★ CSV一致率 (Trace IoU): {score * 100:.1f}%")

    # 4. 比較画像の保存
    # 赤: 自分の軌跡, 緑: お手本の軌跡, 黄: 重なっている部分
    result_view = np.zeros((WARPED_SIZE, WARPED_SIZE, 3), dtype=np.uint8)
    
    # 重ね合わせロジック
    # Bチャンネル(青)は使わない
    # Gチャンネル(緑) = お手本
    result_view[:, :, 1] = img_sample 
    # Rチャンネル(赤) = 自分
    result_view[:, :, 2] = img_user   
    
    # 両方ある場所(R+G=Yellow)は一致、片方だけなら赤か緑に見える
    
    cv2.imwrite(OUTPUT_COMPARE_IMG, result_view)
    print(f"比較画像を保存しました: {OUTPUT_COMPARE_IMG}")
    print("  [緑] お手本のみ")
    print("  [赤] 自分のみ")
    print("  [黄] 一致した部分")

if __name__ == "__main__":
    main()