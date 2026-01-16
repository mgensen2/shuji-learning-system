import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import pandas as pd
import re
import math

# --- 設定 ---
TARGET_DILATE = 13      # お手本(線画)は太くして許容範囲を作る
SUBJECT_DILATE = 1      # 手書き(筆)は太らせない
EDGE_CUT_RATIO = 0.08   # ★追加: 上下左右の端から何%を強制削除するか (0.08 = 8%)

# --- 日本語パス対応 画像読み込み ---
def imread_jp(filename, flags=cv2.IMREAD_GRAYSCALE):
    try:
        n = np.fromfile(filename, np.uint8)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        return None

# --- 影やノイズを除去する関数 ---
def remove_shadows_and_noise(img_bin):
    h, w = img_bin.shape
    
    # --- ★追加: 画像の縁を強制的に黒く塗りつぶしてノイズ削除 ---
    # 元の画像を書き換えないようコピーして処理
    proc_img = img_bin.copy()
    
    cut_w = int(w * EDGE_CUT_RATIO)
    cut_h = int(h * EDGE_CUT_RATIO)
    
    if cut_h > 0:
        proc_img[:cut_h, :] = 0        # 上端を削除
        proc_img[-cut_h:, :] = 0       # 下端を削除
    if cut_w > 0:
        proc_img[:, :cut_w] = 0        # 左端を削除
        proc_img[:, -cut_w:] = 0       # 右端を削除

    # 1. 連結成分分析 (縁をカットした画像で判定)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(proc_img, connectivity=8)
    
    new_img = np.zeros_like(proc_img)
    
    # 影判定エリア（さらに内側の影用）
    margin_w = w * 0.15 
    margin_h = h * 0.15
    
    for i in range(1, num_labels): # ラベル0は背景なのでスキップ
        x, y, w, h, area = stats[i]
        
        # 端に接している塊（影の可能性大）を除外
        # 上下左右の端から数ピクセル以内にあるか
        is_touching_edge = (x <= 1) or (y <= 1) or (x + w >= img_w - 1) or (y + h >= img_h - 1)
        
        # ただし、画面中央を大きく占めるような「巨大な文字」が端に触れている場合は残したいので
        # アスペクト比（縦横比）で「細長い影」だけを消すロジックにする
        aspect_ratio = h / w if w > 0 else 0
        
        # 「端に接している」かつ「極端に細長い（影っぽい）」なら無視
        if is_touching_edge and (aspect_ratio > 5.0 or aspect_ratio < 0.2):
            continue
            
        # ノイズ除去（小さすぎるゴミは無視）
        if area < 50: 
            continue

        new_img[labels == i] = 255
        
    return new_img

# --- 画像の正規化 ＆ ズレ量計算 ---
def normalize_and_process(img_bin, canvas_size=(300, 300), dilate_iter=0):
    """
    戻り値: (正規化後の画像, X方向のズレ, Y方向のズレ)
    """
    coords = cv2.findNonZero(img_bin)
    
    if coords is None: 
        return np.zeros(canvas_size, dtype=np.uint8), 0, 0
    
    # 1. 元画像での文字の位置（バウンディングボックス）を取得
    x, y, w, h = cv2.boundingRect(coords)
    
    # --- ズレ量の計算 ---
    img_h, img_w = img_bin.shape
    img_center_x = img_w / 2
    img_center_y = img_h / 2
    
    char_center_x = x + (w / 2)
    char_center_y = y + (h / 2)
    
    shift_x = char_center_x - img_center_x
    shift_y = char_center_y - img_center_y
    # -------------------

    char_roi = img_bin[y:y+h, x:x+w]
    h_roi, w_roi = char_roi.shape
    target_w, target_h = canvas_size
    
    scale = min(target_w / w_roi, target_h / h_roi) * 0.9
    new_w, new_h = int(max(1, w_roi * scale)), int(max(1, h_roi * scale))
    
    resized_char = cv2.resize(char_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    start_x, start_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_char
    
    if dilate_iter > 0:
        kernel = np.ones((3,3), np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=dilate_iter)
        
    return canvas, shift_x, shift_y

# --- IoU計算 & デバッグ画像保存 ---
def calc_iou_and_save(img_target, img_subject, save_path, shift_dist):
    intersection = cv2.bitwise_and(img_target, img_subject)
    union = cv2.bitwise_or(img_target, img_subject)
    
    area_inter = cv2.countNonZero(intersection)
    area_union = cv2.countNonZero(union)
    
    iou = area_inter / area_union if area_union > 0 else 0.0
    
    debug_img = np.zeros((img_target.shape[0], img_target.shape[1], 3), dtype=np.uint8)
    debug_img[:, :, 1] = img_target # G
    debug_img[:, :, 2] = img_subject # R
    
    # 画像にIoUとズレ距離を描画
    cv2.putText(debug_img, f"IoU: {iou:.3f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Shift: {shift_dist:.1f}px", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    try:
        ext = os.path.splitext(save_path)[1]
        result, n = cv2.imencode(ext, debug_img)
        if result:
            with open(save_path, mode='w+b') as f:
                n.tofile(f)
    except Exception as e:
        print(f"Failed to save debug image: {e}")
        
    return iou

# --- GUIアプリ ---
class IoUAutoMatchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IoU & ズレ量計算 (端カット版)")
        self.root.geometry("650x550")
        
        self.targets = {} 
        self.csv_path = tk.StringVar()
        self.root_dir = tk.StringVar()
        
        # Step 1
        f1 = tk.LabelFrame(root, text="Step 1: お手本画像 (cho, kou, ko)", padx=5, pady=5)
        f1.pack(fill="x", padx=10, pady=5)
        tk.Button(f1, text="画像を選択", command=self.load_targets).pack(side="left")
        self.lbl_targets = tk.Label(f1, text="未登録")
        self.lbl_targets.pack(side="left", padx=10)

        # Step 2
        f2 = tk.LabelFrame(root, text="Step 2: CSVファイル", padx=5, pady=5)
        f2.pack(fill="x", padx=10, pady=5)
        tk.Button(f2, text="CSVを選択", command=self.load_csv).pack(side="left")
        tk.Label(f2, textvariable=self.csv_path).pack(side="left", padx=10)

        # Step 3
        f3 = tk.LabelFrame(root, text="Step 3: データフォルダ", padx=5, pady=5)
        f3.pack(fill="x", padx=10, pady=5)
        tk.Button(f3, text="フォルダを選択", command=self.load_dir).pack(side="left")
        tk.Label(f3, textvariable=self.root_dir).pack(side="left", padx=10)

        # Step 4
        tk.Button(root, text="実行 & CSV保存", command=self.run, bg="#4CAF50", fg="white", height=2).pack(fill="x", padx=10, pady=20)

    def load_targets(self):
        paths = filedialog.askopenfilenames(filetypes=[("Image", "*.png;*.jpg")])
        if not paths: return
        
        self.targets = {}
        loaded_names = []
        
        for p in paths:
            fname = os.path.basename(p).lower()
            key = None
            if 'cho' in fname: key = '刁'
            elif 'kou' in fname: key = '爻'
            elif 'ko' in fname: key = '乎'
            
            if key:
                img = imread_jp(p)
                if img is not None:
                    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    # お手本はカットせずそのまま使用（あるいは必要ならカットしても良いが、通常お手本は綺麗なので不要）
                    processed_target, _, _ = normalize_and_process(bin_img, dilate_iter=TARGET_DILATE)
                    self.targets[key] = processed_target
                    loaded_names.append(f"{fname}->{key}")
        
        self.lbl_targets.config(text=f"登録済み: {', '.join(loaded_names)}")

    def load_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if p: self.csv_path.set(p)

    def load_dir(self):
        d = filedialog.askdirectory()
        if d: self.root_dir.set(d)

    def run(self):
        if not self.targets or not self.csv_path.get() or not self.root_dir.get():
            messagebox.showerror("エラー", "全ての項目を設定してください")
            return
            
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not save_path: return

        debug_dir = os.path.join(os.path.dirname(save_path), "debug_images")
        if not os.path.exists(debug_dir): os.makedirs(debug_dir)

        try:
            df_sheet = pd.read_csv(self.csv_path.get())
        except:
            df_sheet = pd.read_csv(self.csv_path.get(), encoding='shift_jis')

        if '被験者ID(記入用)' in df_sheet.columns:
            df_sheet['被験者ID(記入用)'] = df_sheet['被験者ID(記入用)'].astype(str).str.replace('.0', '', regex=False)

        results = []
        count = 0
        
        for root, dirs, files in os.walk(self.root_dir.get()):
            img_files = [f for f in files if f.lower().endswith(('.png', '.jpg'))]
            if not img_files: continue

            sub_id = os.path.basename(root)
            print(f"Folder: {sub_id}")

            for f in img_files:
                num_match = re.search(r'\d+', f)
                if not num_match: continue
                num = int(num_match.group())

                try:
                    target_id_int = int(sub_id)
                    df_sheet['id_temp_int'] = pd.to_numeric(df_sheet['被験者ID(記入用)'], errors='coerce')
                    row = df_sheet[(df_sheet['id_temp_int'] == target_id_int) & (df_sheet['通し番号'] == num)]
                except:
                    row = df_sheet[(df_sheet['被験者ID(記入用)'] == sub_id) & (df_sheet['通し番号'] == num)]

                if row.empty: continue

                correct_char_str = str(row.iloc[0]['文字'])
                target_key = None
                if '刁' in correct_char_str: target_key = '刁'
                elif '爻' in correct_char_str: target_key = '爻'
                elif '乎' in correct_char_str: target_key = '乎'

                if not target_key or target_key not in self.targets: continue

                f_path = os.path.join(root, f)
                img = imread_jp(f_path)
                if img is None: continue

                _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # ★ここで端カット処理が走ります
                clean_bin = remove_shadows_and_noise(bin_img)
                
                # ★重要2: 手書き画像はほとんど太らせない (SUBJECT_DILATE)
                norm_sub = normalize_and_process(clean_bin, dilate_iter=SUBJECT_DILATE)
                
                dbg_name = f"{sub_id}_{num}_{target_key}.png"
                dbg_path = os.path.join(debug_dir, dbg_name)
                
                iou = calc_iou_and_save(self.targets[target_key], norm_sub, dbg_path, shift_dist)
                
                results.append({
                    "Subject_ID": sub_id,
                    "FileName": f,
                    "Num": num,
                    "Correct_Char": target_key,
                    "Condition": row.iloc[0]['条件'],
                    "IoU": iou,
                    "Shift_X": shift_x,
                    "Shift_Y": shift_y,
                    "Shift_Dist": shift_dist
                })
                count += 1
                print(f"Processed: {sub_id}-{num} IoU={iou:.3f}, Shift={shift_dist:.1f}")

        if results:
            df_result = pd.DataFrame(results)
            df_result.to_csv(save_path, index=False, encoding='utf-8-sig')
            
            summary = df_result.groupby("Condition")[["IoU", "Shift_Dist"]].mean()
            print("\n=== 条件ごとの平均値 ===")
            print(summary)
            
            summary_str = summary.to_string()
            messagebox.showinfo("完了", f"{count}件完了\n\n【条件別平均】\n{summary_str}\n\n保存先: {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = IoUAutoMatchApp(root)
    root.mainloop()