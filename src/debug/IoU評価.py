import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
import pandas as pd

# --- 【重要】日本語パス対応の画像読み込み関数 ---
def imread_jp(filename, flags=cv2.IMREAD_GRAYSCALE):
    """
    日本語（全角文字）を含むパスから画像を読み込むための関数
    np.fromfile でバイナリとして読み込み、cv2.imdecode で画像変換する
    """
    try:
        n = np.fromfile(filename, np.uint8)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(f"Read Error: {e}")
        return None

# --- 画像処理ロジック (サイズ正規化 & 重心合わせ) ---

def normalize_char_shape(img_bin, canvas_size=(300, 300)):
    """
    文字部分だけを切り出し、アスペクト比を維持して
    キャンバスサイズいっぱいにリサイズ・中央配置する関数
    """
    coords = cv2.findNonZero(img_bin)
    if coords is None:
        return np.zeros(canvas_size, dtype=np.uint8)
        
    x, y, w, h = cv2.boundingRect(coords)
    char_roi = img_bin[y:y+h, x:x+w]
    
    h_roi, w_roi = char_roi.shape
    target_w, target_h = canvas_size
    
    # 拡大率の計算（余白0.9倍）
    scale = min(target_w / w_roi, target_h / h_roi) * 0.9
    
    new_w = int(max(1, w_roi * scale)) # 0にならないよう対策
    new_h = int(max(1, h_roi * scale))
    
    resized_char = cv2.resize(char_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    
    # 範囲外エラーを防ぐクリッピング
    end_x = min(start_x + new_w, target_w)
    end_y = min(start_y + new_h, target_h)
    
    # 貼り付けサイズの調整（1pxのズレ対策）
    paste_w = end_x - start_x
    paste_h = end_y - start_y
    
    canvas[start_y:end_y, start_x:end_x] = resized_char[:paste_h, :paste_w]
    
    return canvas

def calculate_iou_normalized(target_path, subject_path):
    """正規化処理付きのIoU計算"""
    
    # --- 【変更点】日本語対応の読み込み関数を使用 ---
    img_target = imread_jp(target_path, cv2.IMREAD_GRAYSCALE)
    img_subject = imread_jp(subject_path, cv2.IMREAD_GRAYSCALE)
    # ---------------------------------------------

    if img_target is None:
        print(f"Error: お手本画像が読めません: {target_path}")
        return None
    if img_subject is None:
        print(f"Error: 被験者画像が読めません: {subject_path}")
        return None

    # 二値化 (Otsu) -> 白文字(255), 黒背景(0) に統一
    # ※白背景・黒文字画像の場合は反転(INV)が必要
    _, bin_target = cv2.threshold(img_target, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bin_subject = cv2.threshold(img_subject, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # サイズと位置の正規化
    norm_target = normalize_char_shape(bin_target)
    norm_subject = normalize_char_shape(bin_subject)

    # IoU計算
    intersection = cv2.bitwise_and(norm_target, norm_subject)
    union = cv2.bitwise_or(norm_target, norm_subject)

    area_inter = cv2.countNonZero(intersection)
    area_union = cv2.countNonZero(union)

    if area_union == 0:
        return 0.0
    
    return area_inter / area_union

# --- GUIアプリケーション ---
class IoUApp:
    def __init__(self, root):
        self.root = root
        self.root.title("書字IoU評価ツール (日本語パス対応版)")
        self.root.geometry("1000x700")

        self.targets = {} 
        self.file_map = [] 

        # --- UI Layout ---
        
        # Step 1: お手本
        frame_top = tk.LabelFrame(root, text="Step 1: お手本画像の登録 (例: 「花」「鳥」「風」の正解画像)", padx=10, pady=10)
        frame_top.pack(fill="x", padx=10, pady=5)
        
        tk.Button(frame_top, text="画像を追加...", command=self.add_target).pack(side="left")
        self.lbl_targets = tk.Label(frame_top, text="登録数: 0", fg="blue")
        self.lbl_targets.pack(side="left", padx=15)
        
        # Step 2: フォルダ
        frame_mid = tk.LabelFrame(root, text="Step 2: 被験者データ読み込み (フォルダ選択)", padx=10, pady=10)
        frame_mid.pack(fill="x", padx=10, pady=5)
        
        tk.Button(frame_mid, text="フォルダを選択", command=self.load_directory).pack(side="left")
        self.lbl_dir = tk.Label(frame_mid, text="未選択")
        self.lbl_dir.pack(side="left", padx=15)

        # Step 3: リスト
        frame_list = tk.Frame(root)
        frame_list.pack(fill="both", expand=True, padx=10, pady=5)
        
        h_frame = tk.Frame(frame_list)
        h_frame.pack(fill="x")
        tk.Label(h_frame, text="ファイル名", width=40, anchor="w", font=("bold")).pack(side="left")
        tk.Label(h_frame, text="比較対象のお手本", width=30, anchor="w", font=("bold")).pack(side="left")

        canvas = tk.Canvas(frame_list, bg="white")
        scrollbar = tk.Scrollbar(frame_list, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg="white")

        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Step 4: 実行
        frame_btm = tk.Frame(root, padx=10, pady=10)
        frame_btm.pack(fill="x")
        
        tk.Button(frame_btm, text="Step 4: IoU計算実行 & CSV保存", command=self.run_calc, 
                  bg="#4CAF50", fg="white", font=("bold", 12), height=2).pack(fill="x")

    def add_target(self):
        paths = filedialog.askopenfilenames(title="お手本画像を選択(複数可)", filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if not paths: return
        
        for p in paths:
            # 拡張子なしのファイル名をラベルにする
            name = os.path.splitext(os.path.basename(p))[0]
            self.targets[name] = p
            
        self.lbl_targets.config(text=f"登録数: {len(self.targets)} ( {', '.join(self.targets.keys())} )")
        if self.file_map: self.refresh_combos()

    def load_directory(self):
        d = filedialog.askdirectory()
        if not d: return
        self.lbl_dir.config(text=d)
        
        files = sorted([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for w in self.scroll_frame.winfo_children(): w.destroy()
        self.file_map = []
        
        for f in files:
            row_frame = tk.Frame(self.scroll_frame, bg="white")
            row_frame.pack(fill="x", pady=2)
            
            tk.Label(row_frame, text=f, width=45, anchor="w", bg="white").pack(side="left")
            
            var = tk.StringVar()
            # 簡易自動マッチング（ファイル名にお手本ラベルが含まれているか）
            matched = next((k for k in self.targets if k in f), "")
            var.set(matched)
            
            cb = ttk.Combobox(row_frame, textvariable=var, values=list(self.targets.keys()), state="readonly", width=25)
            cb.pack(side="left")
            
            self.file_map.append({"name": f, "path": os.path.join(d, f), "var": var})

    def refresh_combos(self):
        opts = list(self.targets.keys())
        for item in self.scroll_frame.winfo_children():
            for child in item.winfo_children():
                if isinstance(child, ttk.Combobox):
                    child['values'] = opts

    def run_calc(self):
        if not self.targets:
            messagebox.showerror("エラー", "お手本画像がありません")
            return
            
        save_path = filedialog.asksaveasfilename(title="保存先", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not save_path: return
        
        results = []
        total = len(self.file_map)
        
        for i, item in enumerate(self.file_map):
            t_label = item["var"].get()
            if not t_label:
                results.append([item["name"], "", "Skipped"])
                continue
                
            t_path = self.targets[t_label]
            iou = calculate_iou_normalized(t_path, item["path"])
            
            val = f"{iou:.4f}" if iou is not None else "Error"
            results.append([item["name"], t_label, val])
            
            if i % 10 == 0:
                print(f"Processing {i}/{total}: {item['name']}")
        
        try:
            df = pd.DataFrame(results, columns=["FileName", "Target_Label", "IoU_Score"])
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("完了", "計算が完了しました！\nCSVファイルを確認してください。")
        except Exception as e:
            messagebox.showerror("保存エラー", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = IoUApp(root)
    root.mainloop()