import cv2
import numpy as np
import pandas as pd
import os

# --- 設定 ---
SAMPLE_CSV = "sample.csv"
USER_CSV = "unpitsu_data_full.csv"
# -----------

CANVAS_SIZE = 800
PADDING = 50

def load_csv(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    return pd.read_csv(filepath, header=0)

def get_bounds(df1, df2):
    all_x = pd.concat([df1['x'], df2['x']])
    all_y = pd.concat([df1['y'], df2['y']])
    return (all_x.min(), all_x.max()), (all_y.min(), all_y.max())

def render_trace(df, x_range, y_range, color=(255, 255, 255)):
    """pen_stateを用いて軌跡を描画"""
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    min_x, max_x = x_range
    min_y, max_y = y_range
    
    range_x = max(max_x - min_x, 1)
    range_y = max(max_y - min_y, 1)
    scale = min((CANVAS_SIZE - 2*PADDING) / range_x, (CANVAS_SIZE - 2*PADDING) / range_y)
    
    def to_px(x, y):
        return int((x - min_x) * scale + PADDING), int((y - min_y) * scale + PADDING)

    # ストロークごとに描画
    for _, group in df.groupby('stroke_id'):
        # pen_state == 1 の点のみを抽出
        points_df = group[group['pen_state'] == 1]
        if len(points_df) > 1:
            pts = [to_px(r['x'], r['y']) for _, r in points_df.iterrows()]
            cv2.polylines(canvas, [np.array(pts)], False, color, 3, cv2.LINE_AA)
            
    return canvas

def calc_iou(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, bin1 = cv2.threshold(gray1, 1, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
    
    inter = cv2.bitwise_and(bin1, bin2)
    union = cv2.bitwise_or(bin1, bin2)
    
    c_inter = cv2.countNonZero(inter)
    c_union = cv2.countNonZero(union)
    return c_inter / c_union if c_union > 0 else 0

def main():
    df_s = load_csv(SAMPLE_CSV)
    df_u = load_csv(USER_CSV)
    
    if df_s is None or df_u is None: return

    xr, yr = get_bounds(df_s, df_u)
    
    img_s = render_trace(df_s, xr, yr, (0, 255, 0)) # Green
    img_u = render_trace(df_u, xr, yr, (0, 0, 255)) # Red
    
    iou = calc_iou(img_s, img_u)
    print(f"Image Score (IoU): {iou*100:.2f}")
    
    merged = cv2.addWeighted(img_s, 1, img_u, 1, 0)
    cv2.imwrite("result_step1_iou.png", merged)

if __name__ == "__main__":
    main()