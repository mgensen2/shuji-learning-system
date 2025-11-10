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

# 4. 画像を分析する際の内部解像度 (大きいほど精度が上がるが、処理が重くなる)
#    (unpitsu_recorder_hybrid.py と同じ値に設定)
ANALYSIS_SIZE = 800

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
    #    Otsuの自動しきい値設定を使用し、紙(白)を 255、墨(黒)を 0 にする
    #    (cv2.THRESH_BINARY)
    try:
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    except Exception as e:
        print(f"エラー: 2値化処理に失敗しました。画像が真っ白または真っ黒である可能性があります。: {e}")
        # もし画像が真っ黒なら、THRESH_OTSUは失敗することがある
        # 代替として、平均値をしきい値にする
        mean_val = np.mean(gray_img)
        if mean_val < 128: # ほぼ黒い画像
            _, binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY) # 0以外を白
        else: # ほぼ白い画像
            _, binary_img = cv2.threshold(gray_img, 254, 255, cv2.THRESH_BINARY) # 255以外を黒
            
    print("--- 2値化処理 完了 ---")

    # 3. 標準サイズにリサイズ
    #    INTER_NEAREST は、白黒の境界をクッキリ保ったままリサイズする
    resized_img = cv2.resize(binary_img, (ANALYSIS_SIZE, ANALYSIS_SIZE), interpolation=cv2.INTER_NEAREST)

    # 4. グリッドセルごとに当たり判定
    CELL_SIZE_PX = ANALYSIS_SIZE // GRID_SIZE
    hit_cells = set() # ヒットしたセルのIDを重複なく格納

    print(f"--- {GRID_SIZE}x{GRID_SIZE} グリッド ({GRID_SIZE*GRID_SIZE}セル) の当たり判定を開始... ---")

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            cell_id = (y * GRID_SIZE) + x
            
            # このセルのピクセル領域 (ROI) を切り出す
            y_start = y * CELL_SIZE_PX
            y_end = (y + 1) * CELL_SIZE_PX
            x_start = x * CELL_SIZE_PX
            x_end = (x + 1) * CELL_SIZE_PX
            
            cell_roi = resized_img[y_start:y_end, x_start:x_end]
            
            # 当たり判定：
            # この領域 (cell_roi) に、黒ピクセル (値 = 0) が
            # 1つでも存在するかどうかをチェックする
            
            # np.min() を使うと高速
            # もしセルの最小値が 0 なら、そのセルには黒が
            if np.min(cell_roi) == 0:
                hit_cells.add(cell_id)

    # 5. 結果をソート
    sorted_hits = sorted(list(hit_cells))
    
    print(f"--- 当たり判定 完了 ---")
    print(f"ヒットした「スピーカ」(セル) の総数: {len(sorted_hits)}")
    print(f"ヒットしたセル ID: {sorted_hits}")

    # 6. CSVファイルに保存
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['cell_id', 'grid_x', 'grid_y'])
            
            for cell_id in sorted_hits:
                grid_x = cell_id % GRID_SIZE
                grid_y = cell_id // GRID_SIZE
                writer.writerow([cell_id, grid_x, grid_y])
        
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