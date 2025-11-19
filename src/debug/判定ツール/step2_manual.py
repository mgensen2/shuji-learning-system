import cv2
import numpy as np
import pandas as pd
import os
import sys

# --- 設定項目 ---
# 自動補正後のファイルを読み込む
INPUT_CSV_FILE  = 'unpitsu_data_corrected.csv'
# 手動補正後の保存ファイル名
OUTPUT_CSV_FILE = 'unpitsu_data_final.csv'
# 背景画像
USER_IMAGE_FILE = 'square-image.jpg'

# パラメータ
WARPED_SIZE = 800
COORD_LIMIT = 200.0
GRID_SIZE = 8
CELL_SIZE_PX = WARPED_SIZE // GRID_SIZE

# --- 関数群 ---
def convert_from_custom_coords_float(x, y):
    """CSV座標 -> ピクセル座標 (小数のまま返す: 高精度維持のため)"""
    norm_x_01 = (x / COORD_LIMIT) + 1.0
    norm_y_01 = y / -COORD_LIMIT
    px = norm_x_01 * WARPED_SIZE
    py = norm_y_01 * WARPED_SIZE
    return px, py  # intに丸めず、floatのまま返す

def convert_to_custom_coords(px, py):
    """ピクセル座標 -> CSV座標"""
    norm_x_01 = px / WARPED_SIZE
    norm_y_01 = py / WARPED_SIZE
    x = (norm_x_01 - 1.0) * COORD_LIMIT
    y = norm_y_01 * -COORD_LIMIT
    return x, y

def get_cell_id(px, py):
    cx = int(px // CELL_SIZE_PX)
    cy = int(py // CELL_SIZE_PX)
    cx = max(0, min(cx, GRID_SIZE - 1))
    cy = max(0, min(cy, GRID_SIZE - 1))
    return (cy * GRID_SIZE) + cx

def render_preview(bg_img, points_float, scale, offset_x, offset_y, pressures):
    """プレビュー画面を描画 (小数の座標を使って滑らかに計算)"""
    display_img = bg_img.copy()
    
    center_x = WARPED_SIZE // 2
    center_y = WARPED_SIZE // 2
    
    # 描画用の一時リスト
    draw_points = []
    
    for i, (px, py) in enumerate(points_float):
        # 1. 中心基準で拡大縮小 (小数計算)
        px_s = (px - center_x) * scale + center_x
        py_s = (py - center_y) * scale + center_y
        
        # 2. 平行移動
        px_final = px_s + offset_x
        py_final = py_s + offset_y
        
        # 描画のためにここで初めて整数に丸める
        draw_points.append((int(round(px_final)), int(round(py_final))))
    
    # 線を描画
    for i in range(len(draw_points) - 1):
        if pressures[i] > 0:
            pt1 = draw_points[i]
            pt2 = draw_points[i+1]
            cv2.line(display_img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA) # AA(アンチエイリアス)で綺麗に
            
    info_text = f"Move: Arrows | Scale: z/x ({scale:.2f}) | Save: Enter | Quit: Esc"
    cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return display_img

def main():
    print("--- Step 2.5: 手動微調整ツール (v2: 高精度版) ---")
    
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"エラー: {INPUT_CSV_FILE} が見つかりません。")
        return
    if not os.path.exists(USER_IMAGE_FILE):
        print(f"エラー: {USER_IMAGE_FILE} が見つかりません。")
        return

    # 1. データ読み込み
    df = pd.read_csv(INPUT_CSV_FILE)
    
    # ピクセル座標を「小数のまま」キャッシュする
    raw_points_float = []
    pressures = []
    for _, row in df.iterrows():
        px, py = convert_from_custom_coords_float(row['x'], row['y'])
        raw_points_float.append((px, py))
        pressures.append(row.get('pressure', 4))
    
    # 背景画像読み込み
    bg_img_raw = cv2.imread(USER_IMAGE_FILE)
    if bg_img_raw is None:
        print("エラー: 画像読み込み失敗")
        return
    bg_img = cv2.resize(bg_img_raw, (WARPED_SIZE, WARPED_SIZE))
    
    # 2. 調整ループ
    scale = 1.0
    offset_x = 0.0 # 小数単位で移動できるように変更
    offset_y = 0.0
    
    print("ウィンドウが立ち上がります。")
    print("  [↑↓←→]: 移動 (Shift同時押しで大きく移動)")
    print("  [z/x]: 縮小/拡大")
    print("  [Enter]: 保存")
    
    while True:
        view = render_preview(bg_img, raw_points_float, scale, offset_x, offset_y, pressures)
        cv2.imshow("Manual Adjustment v2", view)
        
        key = cv2.waitKey(10)
        
        # 終了系
        if key == 27: # Esc
            print("保存せずに終了しました。")
            break
        if key == 13: # Enter
            print("調整を適用して保存します...")
            
            df_final = df.copy()
            new_rows = []
            center_x = WARPED_SIZE // 2
            center_y = WARPED_SIZE // 2
            
            for i, row in df_final.iterrows():
                px, py = raw_points_float[i] # 元の滑らかな座標
                
                # 画面と同じ変換を「小数のまま」適用
                px_s = (px - center_x) * scale + center_x
                py_s = (py - center_y) * scale + center_y
                px_final = px_s + offset_x
                py_final = py_s + offset_y
                
                # 高精度のままCSV座標へ変換
                cx_new, cy_new = convert_to_custom_coords(px_final, py_final)
                cell_id_new = get_cell_id(px_final, py_final)
                
                row['x'] = round(cx_new, 3)
                row['y'] = round(cy_new, 3)
                row['cell_id'] = cell_id_new
                new_rows.append(row)
                
            df_final = pd.DataFrame(new_rows)
            df_final.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
            print(f"保存完了: {OUTPUT_CSV_FILE}")
            break
            
        # 移動系 (Shiftキー判定は環境依存が強いためシンプルに)
        move_step = 1.0 # 細かく動かせるように
        big_step = 10.0
        
        # 特殊キー対応
        code = key & 0xFF
        
        # WASD または 矢印キー (Shift機能は実装が複雑なため、キー割り当てで対応)
        # w/a/s/d = 細かく移動
        # W/A/S/D (Shift+w...) = 大きく移動 (OpenCVのwaitKeyでは大文字小文字で判別可能)
        
        if code == ord('w'): offset_y -= move_step
        if code == ord('s'): offset_y += move_step
        if code == ord('a'): offset_x -= move_step
        if code == ord('d'): offset_x += move_step
        
        if code == ord('W'): offset_y -= big_step
        if code == ord('S'): offset_y += big_step
        if code == ord('A'): offset_x -= big_step
        if code == ord('D'): offset_x += big_step

        # 矢印キー (通常移動)
        if code == 0: offset_y -= move_step # Up
        if code == 1: offset_y += move_step # Down
        if code == 2: offset_x -= move_step # Left
        if code == 3: offset_x += move_step # Right
        
        # Linux/Mac/Windows互換用の矢印キーコード
        if key == 65362 or key == 2490368: offset_y -= move_step
        if key == 65364 or key == 2621440: offset_y += move_step
        if key == 65361 or key == 2424832: offset_x -= move_step
        if key == 65363 or key == 2555904: offset_x += move_step
        
        # 拡大縮小
        if code == ord('z'): scale -= 0.005 # より細かく
        if code == ord('x'): scale += 0.005

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()