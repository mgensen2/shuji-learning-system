import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 設定: 太らせる回数を分ける ---
TARGET_DILATE = 12   # お手本(線画)はガッツリ太くして「許容範囲」を作る
SUBJECT_DILATE = 1   # 手書き(筆)はもともと太いので、穴埋め程度にする

# グラフの表示順（聴覚→触覚→提案）とラベル名
CONDITION_ORDER = ['C', 'B', 'A']
CONDITION_LABELS = {
    'A': 'A:提案手法\n(両方)', 
    'B': 'B:触覚のみ', 
    'C': 'C:聴覚のみ'
}

def main():
    # 1. データ読み込み
    if not os.path.exists(INPUT_CSV):
        print(f"エラー: {INPUT_CSV} が見つかりません。")
        return
    try:
        n = np.fromfile(filename, np.uint8)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        return None

# --- ★追加: 影やノイズを除去する関数 ---
def remove_shadows_and_noise(img_bin):
    """
    影除去の最終版:
    1. 「重心」が端にあるものだけを影とみなす（文字が端に触れていても重心が中央なら消さない）
    2. 離れ文字（父、小など）のパーツは全て残す
    """
    # 連結成分分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)
    
    img_h, img_w = img_bin.shape
    new_img = np.zeros_like(img_bin)
    
    # 影判定の基準ゾーン（画面端から10%以内を「影ゾーン」とする）
    margin_w = img_w * 0.10 
    margin_h = img_h * 0.10
    
    # ラベル0は背景なのでスキップ
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i] # ★ここが重要：塊の「重心」座標
        
        # --- A. 影判定 (重心が端っこにある & 形が細長い) ---
        
        # 左端 or 右端にある縦長の影
        # 「重心(cx)が端のゾーンにある」 かつ 「縦に長い(高さが画面の20%以上)」
        is_side_shadow = (cx < margin_w or cx > img_w - margin_w) and (h > img_h * 0.2)
        
        # 上端 or 下端にある横長の影
        # 「重心(cy)が端のゾーンにある」 かつ 「横に長い(幅が画面の20%以上)」
        is_top_bottom_shadow = (cy < margin_h or cy > img_h - margin_h) and (w > img_w * 0.2)
        
        if is_side_shadow or is_top_bottom_shadow:
            continue # これは影なので描画しない（削除）

        # --- B. ノイズ除去 ---
        # ゴマ粒のようなノイズは消す（文字の点画はこれより大きいはず）
        if area < 20: 
            continue

        # --- C. 採用 ---
        # 影でもゴミでもないものは、全てキャンバスに描く
        # これにより、離れているパーツも全て保持される
        new_img[labels == i] = 255
        
    return new_img
# --- 画像の正規化（修正版） ---
def normalize_and_process(img_bin, canvas_size=(300, 300), dilate_iter=0):
    coords = cv2.findNonZero(img_bin)
    if coords is None: return np.zeros(canvas_size, dtype=np.uint8)
    
    x, y, w, h = cv2.boundingRect(coords)
    char_roi = img_bin[y:y+h, x:x+w]
    
    h_roi, w_roi = char_roi.shape
    target_w, target_h = canvas_size
    
    # スケール計算（余白0.9倍）
    scale = min(target_w / w_roi, target_h / h_roi) * 0.9
    new_w, new_h = int(max(1, w_roi * scale)), int(max(1, h_roi * scale))
    
    resized_char = cv2.resize(char_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    start_x, start_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_char
    
    # 膨張処理（太らせる）
    if dilate_iter > 0:
        kernel = np.ones((3,3), np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=dilate_iter)
        
    # ラベルの置換（改行を入れて幅を抑える）
    df['Condition_Name'] = df['Condition'].map(CONDITION_LABELS)
    
    # フォント設定
    plt.rcParams['font.family'] = FONT_NAME

    # --- Level 1: 全体比較 (棒グラフ) ---
    print("\n【Level 1】全体比較")
    # 集計
    summary1 = df.groupby('Condition')['IoU'].agg(['mean', 'std', 'sem', 'count'])
    print(summary1)
    summary1.to_csv('L1_Overall_Summary.csv', encoding='utf-8-sig')
    
    # グラフ
    plt.figure(figsize=(8, 6), constrained_layout=True) # 重なり防止レイアウト
    sns.barplot(data=df, x='Condition', y='IoU', order=CONDITION_ORDER, 
                palette='viridis', capsize=.1, errorbar='se')
    
    # 緑:お手本(Target), 赤:手書き(Subject), 黄:重なり
    debug_img = np.zeros((img_target.shape[0], img_target.shape[1], 3), dtype=np.uint8)
    debug_img[:, :, 1] = img_target # G
    debug_img[:, :, 2] = img_subject # R
    
    plt.figure(figsize=(10, 6), constrained_layout=True)
    sns.barplot(data=df, x='Correct_Char', y='IoU', hue='Condition', 
                hue_order=CONDITION_ORDER, palette='viridis', capsize=.1)
    
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
        self.root.title("IoU計算 (影除去 & 太さ調整版)")
        self.root.geometry("650x500")
        
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
                    # ★重要: お手本は「線」なので、ガッツリ太らせる (TARGET_DILATE)
                    # お手本は綺麗なので remove_shadows は不要
                    self.targets[key] = normalize_and_process(bin_img, dilate_iter=TARGET_DILATE)
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
                
                # ★重要1: 手書き画像から「影」を除去する
                clean_bin = remove_shadows_and_noise(bin_img)
                
                # ★重要2: 手書き画像はほとんど太らせない (SUBJECT_DILATE)
                norm_sub = normalize_and_process(clean_bin, dilate_iter=SUBJECT_DILATE)
                
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
            messagebox.showinfo("完了", f"{count}件完了\n{save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = IoUAutoMatchApp(root)
    root.mainloop()
