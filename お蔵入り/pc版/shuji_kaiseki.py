import cv2
import numpy as np
import csv
import os
import sys

# --- 設定項目 ---

# 1. 分析したい習字の画像ファイル名
INPUT_IMAGE_FILE = 'calligraphy_image.png'

# 2. 結果を出力するCSVファイル名
OUTPUT_CSV_FILE = 'speaker_hits.csv'

# 3. 仮想グリッドのサイズ (例: 8x8 グリッド)
GRID_SIZE = 8

# 4. 画像を分析する際の内部解像度
ANALYSIS_SIZE = 800

# 5. 解析結果の可視化画像ファイル名
OUTPUT_ANALYSIS_IMAGE = 'analysis_result.png'

# 6. CSVに出力する座標系の定義
COORDINATE_MIN = -200.0
COORDINATE_MAX = 0.0

# 7. 当たり判定のしきい値 (★ 追加)
#    (例: 0.5 = 50%以上黒ピクセルならヒット)
#    (例: 0.01 = 1%以上黒ピクセルならヒット)
#    (0にすると、1ピクセルでもあればヒット ※以前の動作)
HIT_THRESHOLD_PERCENT = 0.4

# -----------------

def create_placeholder_image_if_not_exists(filepath):
    """
    入力画像が存在しない場合に、ダミーの画像を生成する
    """
    if not os.path.exists(filepath):
        print(f"--- {filepath} が見つかりません。 ---")
        print("代わりにダミーのサンプル画像を生成します。")
        print("この 'calligraphy_image.png' をご自身の画像に置き換えて、再度実行してください。")
        
        # 800x800 の白い背景画像を作成
        placeholder_img = np.full((ANALYSIS_SIZE, ANALYSIS_SIZE, 3), 255, dtype=np.uint8)
        
        # ダミーの「書」を描画
        cv2.putText(placeholder_img, "Sumi", (100, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 30, cv2.LINE_AA)
        
        cv2.line(placeholder_img, (400, 400), (600, 700), (0,0,0), 20)
        
        try:
            cv2.imwrite(filepath, placeholder_img)
            print(f"--- サンプル画像 {filepath} を生成しました。 ---")
        except Exception as e:
            print(f"エラー: サンプル画像の保存に失敗しました: {e}")
            sys.exit()

def analyze_image_hits(filepath):
    """
    画像を読み込み、8x8グリッドで黒ピクセルの当たり判定を行う
    """
    print(f"--- 画像ファイル {filepath} を読み込んでいます... ---")
    
    # 画像を読み込む
    img = cv2.imread(filepath)
    if img is None:
        print(f"エラー: 画像ファイル {filepath} を読み込めませんでした。")
        return

    # 1. グレースケールに変換
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 2値化（白黒に変換）
    try:
        _, binary_img_inv = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        binary_img = cv2.bitwise_not(binary_img_inv)
    except Exception as e:
        print(f"エラー: 2値化処理(OTSU)に失敗しました。画像が真っ白または真っ黒である可能性があります。: {e}")
        mean_val = np.mean(gray_img)
        if mean_val < 128:
            _, binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
        else:
            _, binary_img = cv2.threshold(gray_img, 254, 255, cv2.THRESH_BINARY) 
            
    print("--- 2値化処理 完了 ---")

    # 3. 標準サイズにリサイズ
    resized_img = cv2.resize(binary_img, (ANALYSIS_SIZE, ANALYSIS_SIZE), interpolation=cv2.INTER_NEAREST)

    # (★ 変更) 4. グリッドセルごとに当たり判定
    CELL_SIZE_PX = ANALYSIS_SIZE // GRID_SIZE
    hit_cells = set() 

    print(f"--- {GRID_SIZE}x{GRID_SIZE} グリッド ({GRID_SIZE*GRID_SIZE}セル) の当たり判定を開始... ---")
    print(f"--- 当たり判定しきい値: {HIT_THRESHOLD_PERCENT * 100:.0f}% ---")

    # (★ 追加) セル内の総ピクセル数を計算
    total_pixels_in_cell = CELL_SIZE_PX * CELL_SIZE_PX
    
    # (★ 追加) ヒット判定に必要な黒ピクセル数を計算
    # (例: 10000ピクセル * 0.5 = 5000ピクセル)
    hit_threshold_count = total_pixels_in_cell * HIT_THRESHOLD_PERCENT

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            cell_id = (y * GRID_SIZE) + x
            
            # このセルのピクセル領域 (ROI) を切り出す
            y_start = y * CELL_SIZE_PX
            y_end = (y + 1) * CELL_SIZE_PX
            x_start = x * CELL_SIZE_PX
            x_end = (x + 1) * CELL_SIZE_PX
            
            cell_roi = resized_img[y_start:y_end, x_start:x_end]
            
            # (★ 変更) 当たり判定ロジック
            # 変更前:
            # if np.min(cell_roi) == 0:
            
            # 変更後:
            # この領域 (cell_roi) に、黒ピクセル (値 = 0) が
            # しきい値の割合以上存在するかどうかをチェックする

            # 黒ピクセル(値=0)の数をカウント
            # (注: np.count_nonzero は 0 以外の数を数えるので、
            #  (cell_roi == 0) の真偽配列に対して実行する)
            black_pixel_count = np.count_nonzero(cell_roi == 0)
            
            # 黒ピクセル数がしきい値以上ならヒット
            if black_pixel_count >= hit_threshold_count:
                hit_cells.add(cell_id)

    # 5. 結果をソート
    sorted_hits = sorted(list(hit_cells))
    
    print(f"--- 当たり判定 完了 ---")
    print(f"ヒットした「スピーカ」(セル) の総数: {len(sorted_hits)}")
    print(f"ヒットしたセル ID: {sorted_hits}")

    # 5.5. 解析結果の可視化画像を生成・保存
    print(f"--- 解析結果の可視化画像を生成中... ---")
    try:
        debug_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
        
        # グリッド線
        for i in range(1, GRID_SIZE):
            pos = i * CELL_SIZE_PX
            cv2.line(debug_img, (pos, 0), (pos, ANALYSIS_SIZE), (128, 128, 128), 1)
            cv2.line(debug_img, (0, pos), (ANALYSIS_SIZE, pos), (128, 128, 128), 1)

        # ヒットしたセルに赤枠
        for cell_id in sorted_hits:
            grid_x = cell_id % GRID_SIZE
            grid_y = cell_id // GRID_SIZE
            
            x_start = grid_x * CELL_SIZE_PX
            y_start = grid_y * CELL_SIZE_PX
            x_end = (grid_x + 1) * CELL_SIZE_PX - 1
            y_end = (grid_y + 1) * CELL_SIZE_PX - 1
            
            cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        
        cv2.imwrite(OUTPUT_ANALYSIS_IMAGE, debug_img)
        print(f"--- 可視化画像を {OUTPUT_ANALYSIS_IMAGE} に保存しました。 ---")
    except Exception as e:
        print(f"エラー: 可視化画像の保存に失敗しました: {e}")

    # 6. 座標変換とCSVファイルへの保存
    try:
        analysis_range = float(ANALYSIS_SIZE - 1) 
        coord_range = COORDINATE_MAX - COORDINATE_MIN 

        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['cell_id', 'grid_x', 'grid_y', 'plotter_x_start', 'plotter_y_start'])
            
            for cell_id in sorted_hits:
                grid_x = cell_id % GRID_SIZE
                grid_y = cell_id // GRID_SIZE
                
                pixel_x_start = grid_x * CELL_SIZE_PX
                pixel_y_start = grid_y * CELL_SIZE_PX
                
                # X軸: (0..799) -> (-200..0)
                plotter_x = (pixel_x_start / analysis_range) * coord_range + COORDINATE_MIN
                
                # Y軸: (0..799) -> (0..-200)
                plotter_y = (pixel_y_start / analysis_range) * (COORDINATE_MIN - COORDINATE_MAX) + COORDINATE_MAX
                
                writer.writerow([
                    cell_id, 
                    grid_x, 
                    grid_y, 
                    f"{plotter_x:.3f}", 
                    f"{plotter_y:.3f}"
                ])
        
        print(f"--- 結果を {OUTPUT_CSV_FILE} に保存しました。 ---")
    except Exception as e:
        print(f"エラー: CSVファイルの保存に失敗しました: {e}")

# --- メイン実行 ---
def main():
    # 1. ダミー画像がなければ生成する
    create_placeholder_image_if_not_exists(INPUT_IMAGE_FILE)
    
    # 2. 分析を実行する
    analyze_image_hits(INPUT_IMAGE_FILE)

if __name__ == "__main__":
    main()