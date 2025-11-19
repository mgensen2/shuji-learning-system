import cv2
import numpy as np
import pandas as pd
import os
import sys

# --- 設定項目 ---
USER_IMAGE_FILE = 'calligraphy_image.png'       # 書いた文字の画像 (正解位置)
INPUT_CSV_FILE  = '10_full.csv'       # 記録されたCSV (ズレている可能性あり)
OUTPUT_CSV_FILE = 'unpitsu_data_corrected.csv'  # 補正後のCSV
DEBUG_IMAGE     = 'correction_debug.png'        # 補正確認用画像

# 座標変換パラメータ (unpitsu_recorderと同じにする)
WARPED_SIZE = 800
GRID_SIZE = 8
COORD_LIMIT = 200.0
CELL_SIZE_PX = WARPED_SIZE // GRID_SIZE

# --- 関数群 ---
def convert_from_custom_coords(x, y):
    """CSV座標 -> ピクセル座標"""
    norm_x_01 = (x / COORD_LIMIT) + 1.0
    norm_y_01 = y / -COORD_LIMIT
    px = norm_x_01 * WARPED_SIZE
    py = norm_y_01 * WARPED_SIZE
    return int(round(px)), int(round(py))

def convert_to_custom_coords(px, py):
    """ピクセル座標 -> CSV座標"""
    norm_x_01 = px / WARPED_SIZE
    norm_y_01 = py / WARPED_SIZE
    x = (norm_x_01 - 1.0) * COORD_LIMIT
    y = norm_y_01 * -COORD_LIMIT
    return x, y

def get_cell_id(px, py):
    """ピクセル座標 -> セルID"""
    cx = int(px // CELL_SIZE_PX)
    cy = int(py // CELL_SIZE_PX)
    cx = max(0, min(cx, GRID_SIZE - 1))
    cy = max(0, min(cy, GRID_SIZE - 1))
    return (cy * GRID_SIZE) + cx

def render_csv_trace(df):
    """CSVの軌跡を画像化する"""
    canvas = np.zeros((WARPED_SIZE, WARPED_SIZE), dtype=np.uint8)
    last_pos = None
    is_drawing = False
    for _, row in df.iterrows():
        px, py = convert_from_custom_coords(row['x'], row['y'])
        thickness = int(row['pressure'] * 2) + 1
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

def main():
    print("--- Step 2: CSVデータの座標補正 ---")

    if not os.path.exists(USER_IMAGE_FILE) or not os.path.exists(INPUT_CSV_FILE):
        print("エラー: 画像またはCSVファイルが見つかりません。")
        return

    # 1. 実際の画像を読み込み (正解データ)
    img_real = cv2.imread(USER_IMAGE_FILE, cv2.IMREAD_GRAYSCALE)
    img_real = cv2.resize(img_real, (WARPED_SIZE, WARPED_SIZE))
    _, bin_real = cv2.threshold(img_real, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 2. CSVを読み込み、軌跡画像を生成 (補正前)
    print("CSVデータを解析中...")
    df = pd.read_csv(INPUT_CSV_FILE)
    img_csv_trace = render_csv_trace(df)

    # 3. テンプレートマッチングでズレ (dx, dy) を検出
    #    CSVの軌跡画像が、実際の画像の「どこ」にあるかを探す
    print("最適な重なり位置を計算中...")
    
    # 検索範囲を絞るために重心を使ってクロップしても良いが、
    # ここではシンプルにそのままマッチング（全体探索）を行う
    # 計算負荷を下げるため、少し解像度を落としてマッチングする場合もあるが、今回はそのまま実施
    
    res = cv2.matchTemplate(bin_real, img_csv_trace, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # matchTemplateは「左上の位置」を返すため、ズレを計算する
    # img_csv_trace と bin_real は同じサイズなので、ズレがなければ (0,0) がベストになるはず
    # ここでは「CSVの画像」全体をテンプレートとして使うとサイズが同じでマッチングできないため、
    # 「CSVの描画部分」を切り出してマッチングするか、あるいは
    # 重心位置の差分を使う簡易的な方法を採用する（こちらの方が高速で安定しやすい）

    # --- 重心ベースの補正アルゴリズム ---
    M_real = cv2.moments(bin_real)
    M_csv  = cv2.moments(img_csv_trace)

    if M_real["m00"] == 0 or M_csv["m00"] == 0:
        print("エラー: 描画部分が見つかりません（真っ白か真っ黒です）。補正を中止します。")
        return

    cx_real = int(M_real["m10"] / M_real["m00"])
    cy_real = int(M_real["m01"] / M_real["m00"])
    cx_csv  = int(M_csv["m10"] / M_csv["m00"])
    cy_csv  = int(M_csv["m01"] / M_csv["m00"])

    offset_x = cx_real - cx_csv
    offset_y = cy_real - cy_csv
    
    print(f"検出されたズレ: X={offset_x}px, Y={offset_y}px")

    # 4. CSVデータを補正して保存
    print("座標とセルIDを修正中...")
    df_corrected = df.copy()
    new_rows = []
    for _, row in df_corrected.iterrows():
        px, py = convert_from_custom_coords(row['x'], row['y'])
        
        # 補正
        px_new = px + offset_x
        py_new = py + offset_y
        
        # 再変換
        x_new, y_new = convert_to_custom_coords(px_new, py_new)
        cell_id_new = get_cell_id(px_new, py_new)
        
        row['x'] = round(x_new, 3)
        row['y'] = round(y_new, 3)
        row['cell_id'] = cell_id_new
        new_rows.append(row)

    df_corrected = pd.DataFrame(new_rows)
    df_corrected.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"補正済みCSVを保存しました: {OUTPUT_CSV_FILE}")

    # 5. 確認画像の保存
    # 実際の画像(白背景)に、補正後の軌跡(緑線)を重ねる
    debug_img = cv2.cvtColor(img_real, cv2.COLOR_GRAY2BGR) # 元画像
    # 補正後の軌跡を描画
    img_csv_corrected = render_csv_trace(df_corrected)
    contours, _ = cv2.findContours(img_csv_corrected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2) # 緑で描画
    
    cv2.imwrite(DEBUG_IMAGE, debug_img)
    print(f"確認画像を保存しました: {DEBUG_IMAGE}")

if __name__ == "__main__":
    main()