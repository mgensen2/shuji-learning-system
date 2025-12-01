import cv2
import numpy as np
import pandas as pd
import sys
import os

# --- 設定項目 (Settings) ---

# 運筆記録ツール (unpitsu_recorder_hybrid.py) と同じ設定値を使用
COORD_LIMIT = 200.0
WARPED_SIZE = 800
GRID_SIZE = 8 # ★ グリッドサイズを追加 (Add grid size)

# --- 関数 (Functions) ---

def convert_from_custom_coords(x, y):
    """
    (-COORD_LIMIT, 0)-(0, -COORD_LIMIT) の座標を
    (0,0)-(WARPED_SIZE, WARPED_SIZE) のピクセル座標に逆変換する
    """
    # 逆変換ロジック
    # (Inverse transformation logic)
    
    # converted_y = norm_y_01 * -COORD_LIMIT
    norm_y_01 = y / -COORD_LIMIT
    
    # converted_x = (norm_x_01 - 1.0) * COORD_LIMIT
    norm_x_01 = (x / COORD_LIMIT) + 1.0
    
    # ピクセル座標に変換
    # (Convert to pixel coordinates)
    px = norm_x_01 * WARPED_SIZE
    py = norm_y_01 * WARPED_SIZE
    
    return int(round(px)), int(round(py))

def get_color_for_stroke(stroke_id):
    """
    ストロークIDに基づいて色を決定する（デバッグ・視覚化のため）
    """
    """Determines color based on stroke ID (for debugging/visualization)"""
    
    # 事前定義された色のリスト (List of predefined colors)
    colors = [
        (0, 0, 200),   # 赤 (Red)
        (0, 128, 0),   # 緑 (Green)
        (200, 0, 0),   # 青 (Blue)
        (0, 165, 255), # オレンジ (Orange)
        (130, 0, 75),  # 紫 (Purple)
        (128, 128, 0), # オリーブ (Olive)
        (0, 0, 0),     # 黒 (Black) - 7番目以降は黒
    ]
    
    # ストロークIDがリストの範囲内ならその色を、超えたら黒を返す
    # (Return the color if within the list range, otherwise return black)
    if stroke_id - 1 < len(colors):
        return colors[stroke_id - 1]
    else:
        return (0, 0, 0) # デフォルトは黒 (Default is black)

def get_thickness_for_pressure(pressure):
    """
    筆圧 (0-8) を線の太さ (ピクセル) に変換する
    """
    """Converts pressure (0-8) to line thickness (pixels)"""
    
    # 0 (空中) の場合は 1ピクセル (デバッグ用)
    if pressure == 0:
        return 1
    
    # 筆圧に応じて太さをスケーリング (例: 1 -> 2px, 8 -> 17px)
    # (Scale thickness based on pressure (e.g., 1 -> 2px, 8 -> 17px))
    return int(round(pressure * 2)) + 1

# ★★★ 新規追加 ★★★
def draw_grid(image):
    """
    画像に 8x8 のグリッド線を描画する
    """
    """Draws an 8x8 grid on the image"""
    grid_color = (220, 220, 220) # 薄いグレー (Light gray)
    grid_thickness = 1
    
    cell_size = WARPED_SIZE // GRID_SIZE
    
    # 7本の内側の線を描画 (Draw 7 inner lines)
    for i in range(1, GRID_SIZE):
        # 縦線 (Vertical lines)
        cv2.line(image, (i * cell_size, 0), (i * cell_size, WARPED_SIZE), grid_color, grid_thickness)
        # 横線 (Horizontal lines)
        cv2.line(image, (0, i * cell_size), (WARPED_SIZE, i * cell_size), grid_color, grid_thickness)
    
    # (オプション) 外枠を黒で描画 (Optional: Draw outer border in black)
    cv2.rectangle(image, (0, 0), (WARPED_SIZE - 1, WARPED_SIZE - 1), (0, 0, 0), 1)

# --- メイン処理 (Main Process) ---
def main():
    # 1. 読み込むCSVファイル名をユーザーに入力させる
    # (Prompt user for the CSV filename to read)
    csv_filename = input("読み込む _full.csv ファイル名を入力してください: ").strip()

    if not os.path.exists(csv_filename):
        print(f"エラー: ファイル '{csv_filename}' が見つかりません。")
        sys.exit()

    # 2. CSVを pandas で読み込む
    # (Read CSV with pandas)
    try:
        print(f"--- {csv_filename} を読み込んでいます... ---")
        df = pd.read_csv(csv_filename)
        # 必要な列が存在するかチェック (Check if necessary columns exist)
        if not {'event_type', 'x', 'y', 'pressure', 'stroke_id'}.issubset(df.columns):
            print("エラー: CSVファイルに必要な列 (event_type, x, y, pressure, stroke_id) がありません。")
            sys.exit()
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}")
        sys.exit()

    # 3. 描画用の白いキャンバスを作成
    # (Create a white canvas for drawing)
    canvas = np.full((WARPED_SIZE, WARPED_SIZE, 3), 255, dtype=np.uint8)

    # ★★★ 新規追加 ★★★
    # 4. キャンバスにグリッド線を描画
    # (Draw grid lines on the canvas)
    draw_grid(canvas)

    print(f"--- {len(df)} 件のデータを処理して画像を描画します... ---")

    # 5. データを1行ずつ処理して線を描画 (旧ステップ4)
    # (Process data row by row and draw lines (formerly step 4))
    last_pos = None
    is_drawing = False
    current_color = (0, 0, 0)

    for index, row in df.iterrows():
        # 座標をピクセルに逆変換
        # (Inverse transform coordinates to pixels)
        px, py = convert_from_custom_coords(row['x'], row['y'])
        current_pos = (px, py)
        
        event_type = row['event_type']
        
        if event_type == 'down':
            is_drawing = True
            last_pos = current_pos
            # ストロークIDに基づいて色を取得
            # (Get color based on stroke ID)
            current_color = get_color_for_stroke(row['stroke_id'])
            # 線の太さを取得
            # (Get line thickness)
            thickness = get_thickness_for_pressure(row['pressure'])
            # 開始点に点を打つ (Draw a dot at the start point)
            cv2.circle(canvas, current_pos, thickness // 2, current_color, -1)

        elif event_type == 'move':
            if is_drawing and last_pos is not None:
                # 線の太さを取得
                # (Get line thickness)
                thickness = get_thickness_for_pressure(row['pressure'])
                
                # 前回の点から今回の点まで線を引く
                # (Draw a line from the last point to the current point)
                cv2.line(canvas, last_pos, current_pos, current_color, thickness, cv2.LINE_AA)
            
            last_pos = current_pos

        elif event_type == 'up':
            is_drawing = False
            last_pos = None

    # 6. 結果を画像ファイルとして保存 (旧ステップ5)
    # (Save the result as an image file (formerly step 5))
    
    # 出力ファイル名を生成 (Generate output filename)
    output_filename = csv_filename.replace("_full.csv", "_replay.png").replace(".csv", "_replay.png")
    if output_filename == csv_filename: # 拡張子が変わらなかった場合 (If extension didn't change)
        output_filename = csv_filename + "_replay.png"
        
    try:
        cv2.imwrite(output_filename, canvas)
        print(f"--- 成功: 軌跡画像を {output_filename} として保存しました。 ---")
    except Exception as e:
        print(f"エラー: 画像の保存に失敗しました: {e}")

if __name__ == "__main__":
    main()