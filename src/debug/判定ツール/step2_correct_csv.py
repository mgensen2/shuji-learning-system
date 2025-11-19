import cv2
import numpy as np
import pandas as pd
import os
import sys

# --- 設定項目 ---
USER_IMAGE_FILE = 'square-image.jpg'        # 書いた文字の画像
INPUT_CSV_FILE  = '10_full.csv'             # 記録されたCSV
OUTPUT_CSV_FILE = 'unpitsu_data_final.csv'  # 最終保存ファイル名

# パラメータ
WARPED_SIZE = 800
COORD_LIMIT = 200.0
GRID_SIZE = 8
CELL_SIZE_PX = WARPED_SIZE // GRID_SIZE
OUTLIER_MARGIN = 20.0  # 外れ値除去マージン(mm)

# --- 関数群 ---
def convert_from_custom_coords_float(x, y):
    """CSV座標 -> ピクセル座標 (小数のまま)"""
    norm_x_01 = (x / COORD_LIMIT) + 1.0
    norm_y_01 = y / -COORD_LIMIT
    px = norm_x_01 * WARPED_SIZE
    py = norm_y_01 * WARPED_SIZE
    return px, py

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

def get_centroid(binary_img):
    M = cv2.moments(binary_img)
    if M["m00"] == 0: return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)

def remove_outliers(df):
    """座標やセルIDに基づいて外れ値を除去"""
    min_val = -COORD_LIMIT - OUTLIER_MARGIN
    max_val = 0.0 + OUTLIER_MARGIN
    mask_coord = (df['x'] >= min_val) & (df['x'] <= max_val) & \
                 (df['y'] >= min_val) & (df['y'] <= max_val)
    mask_cell = (df['cell_id'] >= 0) & (df['cell_id'] < GRID_SIZE * GRID_SIZE)
    df_clean = df[mask_coord & mask_cell].copy()
    removed = len(df) - len(df_clean)
    if removed > 0: print(f"  [Info] {removed} 個の外れ値を除去しました。")
    return df_clean

def render_trace_simple(points_float, pressures):
    """重心計算用の簡易描画"""
    canvas = np.zeros((WARPED_SIZE, WARPED_SIZE), dtype=np.uint8)
    if not points_float: return canvas
    pts_int = [(int(round(px)), int(round(py))) for px, py in points_float]
    for i in range(len(pts_int) - 1):
        if pressures[i] > 0:
            thickness = int(pressures[i] * 3) + 2
            cv2.line(canvas, pts_int[i], pts_int[i+1], 255, thickness)
    return canvas

def smooth_trajectory(points, window_size):
    """移動平均フィルタで軌跡を滑らかにする"""
    if len(points) < window_size: return points
    points_np = np.array(points)
    df_temp = pd.DataFrame(points_np, columns=['x', 'y'])
    # center=Trueで位置ズレを防ぐ
    df_smoothed = df_temp.rolling(window=window_size, center=True, min_periods=1).mean()
    return df_smoothed.to_numpy().tolist()

def render_preview(bg_img, points_grouped, scale, offset_x, offset_y, enable_smoothing, smoothing_window, mode_text):
    """プレビュー画面描画"""
    display_img = bg_img.copy()
    center_x = WARPED_SIZE // 2
    center_y = WARPED_SIZE // 2
    SHIFT_BITS = 4
    SHIFT_FACTOR = 1 << SHIFT_BITS

    for stroke_data in points_grouped:
        raw_pts = stroke_data['points']
        pressures = stroke_data['pressures']
        
        # スムージング適用 (ONの場合)
        if enable_smoothing:
            pts_to_draw = smooth_trajectory(raw_pts, window_size=smoothing_window)
        else:
            pts_to_draw = raw_pts

        # 座標変換 & 固定小数点化
        pts_fixed = []
        for px, py in pts_to_draw:
            px_s = (px - center_x) * scale + center_x + offset_x
            py_s = (py - center_y) * scale + center_y + offset_y
            px_fixed = int(round(px_s * SHIFT_FACTOR))
            py_fixed = int(round(py_s * SHIFT_FACTOR))
            pts_fixed.append((px_fixed, py_fixed))
        
        # 線を描画
        for i in range(len(pts_fixed) - 1):
            if pressures[i] > 0:
                pt1 = pts_fixed[i]
                pt2 = pts_fixed[i+1]
                cv2.line(display_img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA, shift=SHIFT_BITS)

    # ステータス表示
    cv2.rectangle(display_img, (0, 0), (WARPED_SIZE, 60), (0,0,0), -1)
    cv2.putText(display_img, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    status_smooth = f"ON (Win:{smoothing_window})" if enable_smoothing else "OFF"
    sub_text = f"Scale:{scale:.2f} | Smooth:{status_smooth}"
    cv2.putText(display_img, sub_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    return display_img

def main():
    print("--- Step 2: 統合補正ツール (外れ値除去 + 自動補正 + スムージング) ---")
    
    if not os.path.exists(INPUT_CSV_FILE) or not os.path.exists(USER_IMAGE_FILE):
        print("エラー: ファイルが見つかりません。")
        return

    # 1. 画像読み込み
    bg_img_raw = cv2.imread(USER_IMAGE_FILE)
    bg_img = cv2.resize(bg_img_raw, (WARPED_SIZE, WARPED_SIZE))
    
    # 重心計算用画像処理
    img_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    _, bin_real = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    bin_real_clean = cv2.morphologyEx(bin_real, cv2.MORPH_OPEN, kernel, iterations=2)
    
    center_real = get_centroid(bin_real_clean)
    if center_real is None:
        center_real = (WARPED_SIZE/2, WARPED_SIZE/2)
        print("警告: 画像の中心が見つかりません。仮の中心を使用します。")

    # 2. CSV読み込み & 外れ値除去
    df_raw = pd.read_csv(INPUT_CSV_FILE)
    df = remove_outliers(df_raw)

    # 座標リスト化
    raw_points_all = []
    pressures_all = []
    for _, row in df.iterrows():
        px, py = convert_from_custom_coords_float(row['x'], row['y'])
        raw_points_all.append((px, py))
        pressures_all.append(row.get('pressure', 4))

    # 3. 自動補正 (重心合わせ)
    print("自動補正中...")
    img_csv_temp = render_trace_simple(raw_points_all, pressures_all)
    center_csv = get_centroid(img_csv_temp)
    
    auto_offset_x = 0.0
    auto_offset_y = 0.0
    if center_csv:
        auto_offset_x = center_real[0] - center_csv[0]
        auto_offset_y = center_real[1] - center_csv[1]
        print(f"  自動ズレ補正: X={auto_offset_x:.1f}, Y={auto_offset_y:.1f}")
    
    # 自動補正を適用してグループ化
    points_grouped = []
    if 'stroke_id' not in df.columns: df['stroke_id'] = 0
    stroke_ids = df['stroke_id'].unique()
    
    for sid in stroke_ids:
        df_stroke = df[df['stroke_id'] == sid]
        pts = []
        prs = []
        for _, row in df_stroke.iterrows():
            px, py = convert_from_custom_coords_float(row['x'], row['y'])
            px += auto_offset_x
            py += auto_offset_y
            pts.append((px, py))
            prs.append(row.get('pressure', 4))
        if pts:
            points_grouped.append({'points': pts, 'pressures': prs, 'stroke_id': sid})

    # 4. 手動調整 & スムージング設定ループ
    scale = 1.0
    offset_x = 0.0
    offset_y = 0.0
    
    # ★ スムージング設定
    enable_smoothing = True   # デフォルトON
    smoothing_window = 5      # デフォルト強度 (3〜9推奨)
    
    manual_mode = False
    
    print("\n=== 操作方法 ===")
    print(" [Enter] : 保存して終了")
    print(" [M]     : 手動微調整モードへ")
    print(" [Esc]   : 終了")
    print(" --- 手動モード時 ---")
    print(" [矢印]  : 移動")
    print(" [z / x] : 縮小 / 拡大")
    print(" [s]     : スムージング ON/OFF 切替")
    print(" [1 / 2] : 滑らかさ調整 (1:弱く / 2:強く)")
    
    while True:
        if not manual_mode:
            mode_msg = "Auto Corrected. [Enter]:Save / [M]:Manual"
        else:
            mode_msg = "Manual Mode. [Arrows]:Move [z/x]:Scale [1/2]:Smooth"
            
        view = render_preview(bg_img, points_grouped, scale, offset_x, offset_y, 
                              enable_smoothing, smoothing_window, mode_msg)
        cv2.imshow("Integrated Tool Final", view)
        
        key = cv2.waitKey(10)
        code = key & 0xFF
        
        # 終了
        if code == 27: break
        if code == 13: # Enter (Save)
            print("保存中...")
            new_rows = []
            center_x = WARPED_SIZE // 2
            center_y = WARPED_SIZE // 2
            
            for stroke_data in points_grouped:
                raw_pts = stroke_data['points']
                sid = stroke_data['stroke_id']
                prs = stroke_data['pressures']
                
                # ★ 保存時にスムージングを適用
                if enable_smoothing:
                    pts_save = smooth_trajectory(raw_pts, window_size=smoothing_window)
                else:
                    pts_save = raw_pts
                
                for i, (px, py) in enumerate(pts_save):
                    # 手動補正適用
                    px_final = (px - center_x) * scale + center_x + offset_x
                    py_final = (py - center_y) * scale + center_y + offset_y
                    
                    cx_new, cy_new = convert_to_custom_coords(px_final, py_final)
                    cell_id_new = get_cell_id(px_final, py_final)
                    
                    row = {
                        'stroke_id': sid,
                        'x': round(cx_new, 3),
                        'y': round(cy_new, 3),
                        'pressure': prs[i],
                        'cell_id': cell_id_new,
                        'event_type': 'move'
                    }
                    if i==0: row['event_type'] = 'down'
                    if i==len(pts_save)-1: row['event_type'] = 'up'
                    new_rows.append(row)
            
            df_final = pd.DataFrame(new_rows)
            df_final.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
            print(f"--- 保存完了: {OUTPUT_CSV_FILE} ---")
            break

        # モード切替
        if not manual_mode and (code == ord('m') or code == ord('M')):
            manual_mode = True
            continue

        if manual_mode:
            # 移動
            step = 1.0
            if code == ord('w') or code == 0: offset_y -= step
            if code == ord('s') or code == 1: offset_y += step
            if code == ord('a') or code == 2: offset_x -= step
            if code == ord('d') or code == 3: offset_x += step
            
            # Linux/Mac/Win 矢印キー
            if key in [65362, 2490368]: offset_y -= step
            if key in [65364, 2621440]: offset_y += step
            if key in [65361, 2424832]: offset_x -= step
            if key in [65363, 2555904]: offset_x += step
            
            # 拡大縮小
            if code == ord('z'): scale -= 0.002
            if code == ord('x'): scale += 0.002
            
            # スムージング操作
            if code == ord('s'): 
                enable_smoothing = not enable_smoothing
            
            # [1]キー: 滑らかさを弱める (窓を小さく)
            if code == ord('1'):
                smoothing_window = max(1, smoothing_window - 2)
            
            # [2]キー: 滑らかさを強める (窓を大きく)
            if code == ord('2'):
                smoothing_window = min(21, smoothing_window + 2)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()