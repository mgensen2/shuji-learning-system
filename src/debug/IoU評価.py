import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import pandas as pd
import re

# --- 設定: 線を太らせる回数 ---
DILATE_ITER = 7  # 文字を太くしてIoUを安定させる（3~5推奨）

# --- 日本語パス対応 画像読み込み ---
def imread_jp(filename, flags=cv2.IMREAD_GRAYSCALE):
    try:
        n = np.fromfile(filename, np.uint8)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        # print(f"Error reading {filename}: {e}")
        return None

# --- 画像の前処理（正規化・太らせる） ---
def normalize_and_process(img_bin, canvas_size=(300, 300), dilate_iter=0):
    coords = cv2.findNonZero(img_bin)
    if coords is None: return np.zeros(canvas_size, dtype=np.uint8)
    
    x, y, w, h = cv2.boundingRect(coords)
    char_roi = img_bin[y:y+h, x:x+w]
    
    h_roi, w_roi = char_roi.shape
    target_w, target_h = canvas_size
    # 余白を少し持たせる (0.9倍)
    scale = min(target_w / w_roi, target_h / h_roi) * 0.9
    new_w, new_h = int(max(1, w_roi * scale)), int(max(1, h_roi * scale))
    
    resized_char = cv2.resize(char_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    start_x, start_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_char
    
    if dilate_iter > 0:
        kernel = np.ones((3,3), np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=dilate_iter)
        
    return canvas

# --- IoU計算 & デバッグ画像保存 ---
def calc_iou_and_save(img_target, img_subject, save_path):
    intersection = cv2.bitwise_and(img_target, img_subject)
    union = cv2.bitwise_or(img_target, img_subject)
    
    area_inter = cv2.countNonZero(intersection)
    area_union = cv2.countNonZero(union)
    
    iou = area_inter / area_union if area_union > 0 else 0.0
    
    # デバッグ画像作成（カラー）
    # Green: Target, Red: Subject
    debug_img = np.zeros((img_target.shape[0], img_target.shape[1], 3), dtype=np.uint8)
    debug_img[:, :, 1] = img_target
    debug_img[:, :, 2] = img_subject
    
    # 視認性向上のため明るく
    debug_img = cv2.convertScaleAbs(debug_img, alpha=1.5, beta=0)
    
    cv2.putText(debug_img, f"IoU: {iou:.3f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 保存
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
        self.root.title("IoU計算 (CSV自動連携・ゼロ埋め対応版)")
        self.root.geometry("650x500")
        
        # 変数
        self.targets = {} 
        self.csv_path = tk.StringVar()
        self.root_dir = tk.StringVar()
        
        # Step 1: お手本
        f1 = tk.LabelFrame(root, text="Step 1: お手本画像 (ファイル名に cho, kou, ko を含むもの)", padx=5, pady=5)
        f1.pack(fill="x", padx=10, pady=5)
        tk.Button(f1, text="画像を選択 (複数可)", command=self.load_targets).pack(side="left")
        self.lbl_targets = tk.Label(f1, text="未登録")
        self.lbl_targets.pack(side="left", padx=10)

        # Step 2: CSV
        f2 = tk.LabelFrame(root, text="Step 2: 実験進行チェックシート (CSV)", padx=5, pady=5)
        f2.pack(fill="x", padx=10, pady=5)
        tk.Button(f2, text="CSVを選択", command=self.load_csv).pack(side="left")
        tk.Label(f2, textvariable=self.csv_path).pack(side="left", padx=10)

        # Step 3: フォルダ
        f3 = tk.LabelFrame(root, text="Step 3: データ親フォルダ (中に002, 003...)", padx=5, pady=5)
        f3.pack(fill="x", padx=10, pady=5)
        tk.Button(f3, text="フォルダを選択", command=self.load_dir).pack(side="left")
        tk.Label(f3, textvariable=self.root_dir).pack(side="left", padx=10)

        # Step 4: 実行
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
                    self.targets[key] = normalize_and_process(bin_img, dilate_iter=DILATE_ITER)
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

        # デバッグフォルダ作成
        debug_dir = os.path.join(os.path.dirname(save_path), "debug_images")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        # CSV読み込み
        try:
            df_sheet = pd.read_csv(self.csv_path.get())
        except:
            df_sheet = pd.read_csv(self.csv_path.get(), encoding='shift_jis')

        # ID列の文字列化（.0削除対応）
        if '被験者ID(記入用)' in df_sheet.columns:
            df_sheet['被験者ID(記入用)'] = df_sheet['被験者ID(記入用)'].astype(str).str.replace('.0', '', regex=False)
        else:
            messagebox.showerror("エラー", "CSVに「被験者ID(記入用)」列が見つかりません。")
            return

        results = []
        count = 0
        
        # フォルダ巡回
        for root, dirs, files in os.walk(self.root_dir.get()):
            img_files = [f for f in files if f.lower().endswith(('.png', '.jpg'))]
            if not img_files: continue

            sub_id = os.path.basename(root) # フォルダ名 "004"
            print(f"Folder: {sub_id}")

            for f in img_files:
                # 通し番号抽出 (1.png -> 1)
                num_match = re.search(r'\d+', f)
                if not num_match: continue
                num = int(num_match.group())

                # ★柔軟なIDマッチング（数字として比較）
                try:
                    target_id_int = int(sub_id)
                    # CSV側でIDを数値化して検索
                    df_sheet['id_temp_int'] = pd.to_numeric(df_sheet['被験者ID(記入用)'], errors='coerce')
                    row = df_sheet[
                        (df_sheet['id_temp_int'] == target_id_int) & 
                        (df_sheet['通し番号'] == num)
                    ]
                except:
                    # 数値変換できない場合は文字列一致で検索
                    row = df_sheet[
                        (df_sheet['被験者ID(記入用)'] == sub_id) & 
                        (df_sheet['通し番号'] == num)
                    ]

                if row.empty: continue

                # 正解文字
                correct_char_str = str(row.iloc[0]['文字'])
                target_key = None
                if '刁' in correct_char_str: target_key = '刁'
                elif '爻' in correct_char_str: target_key = '爻'
                elif '乎' in correct_char_str: target_key = '乎'

                if not target_key or target_key not in self.targets: continue

                # 計算
                f_path = os.path.join(root, f)
                img = imread_jp(f_path)
                if img is None: continue

                _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                norm_sub = normalize_and_process(bin_img, dilate_iter=DILATE_ITER)
                
                dbg_name = f"{sub_id}_{num}_{target_key}.png"
                dbg_path = os.path.join(debug_dir, dbg_name)
                
                iou = calc_iou_and_save(self.targets[target_key], norm_sub, dbg_path)
                
                results.append({
                    "Subject_ID": sub_id,
                    "FileName": f,
                    "Num": num,
                    "Correct_Char": target_key,
                    "Condition": row.iloc[0]['条件'],
                    "IoU": iou
                })
                count += 1
                print(f"Processed: {sub_id}-{num} ({target_key}) IoU={iou:.3f}")

        if results:
            pd.DataFrame(results).to_csv(save_path, index=False, encoding='utf-8-sig')
            messagebox.showinfo("完了", f"{count}件の処理が完了しました。\n保存先: {save_path}")
        else:
            messagebox.showwarning("警告", "処理対象が1件もありませんでした。CSVのIDとフォルダ名を確認してください。")

if __name__ == "__main__":
    root = tk.Tk()
    app = IoUAutoMatchApp(root)  # アプリをインスタンス化
    root.mainloop()              # rootでループを開始（これが正解）