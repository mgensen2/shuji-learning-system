import pandas as pd
import numpy as np
import os

INPUT_CSV = "unpitsu_data_full.csv"
OUTPUT_CSV = "normalized_data.csv"

def normalize_csv(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    # 1. 読み込み (ヘッダーあり)
    df = pd.read_csv(input_path, header=0)
    
    # 2. ノイズ除去等の処理 (例: pen_state=1 のデータ数が少なすぎるストロークを削除など)
    # ここでは単純な正規化のみ例示します
    
    # 座標の正規化 (0.0 ~ 1.0) - pen_state関係なく全体で行うか、描画範囲で行うか
    # ここでは全体を 0-1 に正規化して保存する例
    min_x, max_x = df['x'].min(), df['x'].max()
    min_y, max_y = df['y'].min(), df['y'].max()
    
    if max_x != min_x:
        df['x'] = (df['x'] - min_x) / (max_x - min_x)
    if max_y != min_y:
        df['y'] = (df['y'] - min_y) / (max_y - min_y)

    # 3. 保存 (ヘッダーあり indexなし)
    # pen_state列もそのまま保存されます
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Normalized data saved to: {output_path}")

if __name__ == "__main__":
    normalize_csv(INPUT_CSV, OUTPUT_CSV)