import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
import math
import copy

# --- 1. 永字八法レシピ (音の設定のみ) ---
# Format: "Val1 Val2 Val3 Val4"
EIGHT_STROKES = {
    "動的 (F/距離連動)": {
        "Speaker": "DYNAMIC" 
    },
    "側 (点)": {
        "Speaker": "150 100 250 2"
    },
    "勒 (横画)": {
        "Speaker": "130 300 250 1"
    },
    "努 (縦画)": {
        "Speaker": "100 400 255 1"
    },
    "趯 (跳ね)": {
        "Speaker": "200 150 250 3"
    },
    "策 (短横画)": {
        "Speaker": "140 200 250 1"
    },
    "掠 (左はらい)": {
        "Speaker": "120 350 240 1"
    },
    "啄 (短いはらい)": {
        "Speaker": "180 100 250 2"
    },
    "磔 (右はらい)": {
        "Speaker": "110 500 255 1"
    }
}

DEFAULT_SPEAKER_PARAMS = "130 300 250 2"

class PlotterCsvEnhancer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("プロッタ用 音響エンハンサー (動的F/距離対応)")
        self.geometry("1100x700")

        self.CANVAS_WIDTH = 550
        self.CANVAS_HEIGHT = 550
        self.COORD_LIMIT = 200.0
        
        self.rows = []
        self.headers = []

        self.create_widgets()

    def create_widgets(self):
        left_frame = tk.Frame(self, padx=10, pady=10, width=400)
        left_frame.pack(side="left", fill="y")
        
        right_frame = tk.Frame(self, padx=10, pady=10)
        right_frame.pack(side="right", fill="both", expand=True)

        # --- 1. ファイル操作 ---
        file_frame = tk.LabelFrame(left_frame, text="1. ファイル操作", padx=5, pady=5)
        file_frame.pack(fill="x", pady=5)
        
        btn_frame = tk.Frame(file_frame)
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="CSVを開く", command=self.load_csv, bg="#ddd").pack(side="left", padx=5)
        tk.Button(btn_frame, text="CSV保存", command=self.save_csv, bg="#ddd").pack(side="left", padx=5)
        
        tk.Button(file_frame, text="TXT保存 (プロッタ用)", command=self.save_txt_for_plotter, bg="#FF9800", fg="white").pack(fill="x", padx=5, pady=5)
        
        self.lbl_status = tk.Label(file_frame, text="未読み込み")
        self.lbl_status.pack(anchor="w", padx=5)

        # --- 2. データリスト ---
        list_frame = tk.LabelFrame(left_frame, text="2. データ選択 (A1/G1行を選択)", padx=5, pady=5)
        list_frame.pack(fill="both", expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        self.listbox = tk.Listbox(list_frame, selectmode="extended", yscrollcommand=scrollbar.set, font=("Consolas", 9))
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        # --- 3. 音響設定 ---
        convert_frame = tk.LabelFrame(left_frame, text="3. 音響設定", padx=5, pady=5)
        convert_frame.pack(fill="x", pady=5)

        tk.Label(convert_frame, text="技法/モード:").pack(side="left")
        self.stroke_var = tk.StringVar(self)
        self.stroke_var.set("動的 (F/距離連動)") 
        
        menu_items = ["動的 (F/距離連動)"] + [k for k in EIGHT_STROKES.keys() if k != "動的 (F/距離連動)"]
        stroke_menu = tk.OptionMenu(convert_frame, self.stroke_var, *menu_items)
        stroke_menu.pack(side="left", padx=5)

        tk.Button(convert_frame, text="適用", command=self.apply_enhancement, bg="#4CAF50", fg="white").pack(side="left", padx=10)

        # --- 4. キャンバス ---
        self.canvas = tk.Canvas(right_frame, bg="white", width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)
        self.canvas.pack(fill="both", expand=True)
        
        legend_frame = tk.Frame(right_frame)
        legend_frame.pack(fill="x")
        tk.Label(legend_frame, text="凡例: 線色=速度(青:遅→赤:速), 線太さ=筆圧(Z)").pack()

    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath: return

        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                self.headers = reader.fieldnames
                self.rows = [row for row in reader]
            
            for row in self.rows:
                if 'Command' not in row and 'event_type' in row:
                    evt = row['event_type']
                    if evt == 'down': row['Command'] = 'A0'
                    elif evt == 'move': row['Command'] = 'A1'
                    elif evt == 'up': row['Command'] = 'A0'
                    else: row['Command'] = 'A0'
                
                for k in ['X', 'Y', 'Z', 'F', 'Delay_ms', 'Cell_ID', 'pressure', 'x', 'y']:
                    val = None
                    if k in row: val = row[k]
                    elif k.lower() in row: val = row[k.lower()]
                    elif k.upper() in row: val = row[k.upper()]
                    
                    if val is not None and val != '':
                        try:
                            target_key = k
                            if k in ['x', 'X']: target_key = 'X'
                            if k in ['y', 'Y']: target_key = 'Y'
                            if k in ['z', 'Z', 'pressure']: target_key = 'Z'
                            if k in ['cell_id', 'Cell_ID']: target_key = 'Cell_ID'
                            
                            row[target_key] = float(val)
                        except: pass
                
                if 'SpeakerParams' not in row:
                    row['SpeakerParams'] = DEFAULT_SPEAKER_PARAMS

            self.lbl_status.config(text=f"{len(self.rows)}行")
            self.update_listbox()
            self.draw_trajectory()
            messagebox.showinfo("成功", "CSVを読み込みました。")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def save_csv(self):
        if not self.rows: return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not filepath: return

        try:
            output_headers = ['Command', 'X', 'Y', 'Z', 'F', 'Delay_ms', 'Cell_ID', 'SpeakerParams']
            
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=output_headers, extrasaction='ignore')
                writer.writeheader()
                for row in self.rows:
                    out_row = {}
                    for k in output_headers:
                        val = row.get(k, '')
                        if isinstance(val, float):
                            if k in ['X', 'Y', 'Z']: val = f"{val:.2f}"
                            else: val = f"{int(val)}"
                        out_row[k] = val
                    writer.writerow(out_row)
            messagebox.showinfo("成功", "CSVを保存しました。")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def save_txt_for_plotter(self):
        if not self.rows: return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not filepath: return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for row in self.rows:
                    cmd = row.get('Command', 'A0')
                    g_code = cmd
                    if cmd == 'A0': g_code = 'G0'
                    if cmd == 'A1': g_code = 'G1'
                    
                    if g_code not in ['G0', 'G1', 'D1']:
                        continue

                    x = row.get('X', 0.0)
                    y = row.get('Y', 0.0)
                    z = row.get('Z', 0.0)
                    f_val = row.get('F', '')
                    delay_ms = row.get('Delay_ms', '')
                    cell_id = int(row.get('Cell_ID', 0))
                    
                    speaker_params = row.get('SpeakerParams', DEFAULT_SPEAKER_PARAMS)

                    spk_delay = int(delay_ms) if delay_ms != '' and delay_ms > 0 else 100
                    spk_cmd = f"A2 {cell_id} {spk_delay} {speaker_params}"

                    if g_code == 'G0':
                        line_g0 = f"G0 X{x:.2f} Y{y:.2f} Z{z:.2f}"
                        line_s = f"S {spk_cmd}"
                        f.write(line_g0 + "\n")
                        f.write(line_s + "\n")
                        
                    elif g_code == 'G1':
                        line_g1 = f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f}"
                        if f_val != '':
                            line_g1 += f" F{int(f_val)}"
                        
                        line_g1 += f"\t{spk_cmd}"
                        f.write(line_g1 + "\n")
                        
                    elif g_code == 'D1':
                        if delay_ms != '':
                            f.write(f"D1 {int(delay_ms)}\n")

            messagebox.showinfo("成功", f"プロッタ用TXTファイルを保存しました。\n{filepath}")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def calculate_dynamic_params(self, f_val, distance):
        f_num = float(f_val) if f_val != '' else 1000.0
        val1 = 100 + (f_num / 20.0)
        val1 = max(50, min(500, val1))

        val2 = 100 + (distance * 4.0)
        val2 = max(50, min(800, val2))

        val3 = 250
        val4 = 2

        return f"{int(val1)} {int(val2)} {int(val3)} {int(val4)}"

    def apply_enhancement(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "適用したい行を選択してください。")
            return

        stroke_name = self.stroke_var.get()
        is_dynamic = (stroke_name == "動的 (F/距離連動)")
        
        fixed_params = DEFAULT_SPEAKER_PARAMS
        if not is_dynamic:
            fixed_params = EIGHT_STROKES[stroke_name]["Speaker"]
        
        indices = list(selection)
        target_indices = [i for i in indices if self.rows[i].get('Command') in ['A1', 'G1']]
        
        if not target_indices:
            messagebox.showwarning("警告", "選択範囲に描画コマンド(A1/G1)が含まれていません。")
            return

        for idx in target_indices:
            if is_dynamic:
                prev_x = 0.0
                prev_y = 0.0
                if idx > 0:
                    prev_x = float(self.rows[idx-1].get('X', 0))
                    prev_y = float(self.rows[idx-1].get('Y', 0))
                
                curr_x = float(self.rows[idx].get('X', 0))
                curr_y = float(self.rows[idx].get('Y', 0))
                
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                f_val = self.rows[idx].get('F', '')
                
                self.rows[idx]['SpeakerParams'] = self.calculate_dynamic_params(f_val, distance)
            else:
                self.rows[idx]['SpeakerParams'] = fixed_params

        self.update_listbox()
        for i in indices:
            self.listbox.selection_set(i)
        messagebox.showinfo("完了", f"{stroke_name} の音設定を適用しました。")

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, row in enumerate(self.rows):
            cmd = row.get('Command', '')
            x = row.get('X', 0)
            y = row.get('Y', 0)
            z = row.get('Z', 0)
            f = row.get('F', '')
            cid = row.get('Cell_ID', '')
            spk = row.get('SpeakerParams', DEFAULT_SPEAKER_PARAMS)
            
            txt = f"{i+1:03d}: {cmd} | Cell:{cid} | Z:{z:.1f} F:{f} | Spk:{spk}"
            self.listbox.insert(tk.END, txt)

    def draw_trajectory(self):
        self.canvas.delete("all")
        
        scale = self.CANVAS_WIDTH / (self.COORD_LIMIT * 2.2)
        center_x = self.CANVAS_WIDTH / 2
        center_y = self.CANVAS_HEIGHT / 2

        def to_canvas(cx, cy):
            x = (cx + 100) * scale + center_x 
            # ★ Y座標の方向を反転 (下方向がマイナスのため)
            y = (-cy) * scale + center_y + (100 * scale)
            return x, y

        prev_x, prev_y = None, None
        
        for row in self.rows:
            cmd = row.get('Command')
            if cmd not in ['A0', 'A1', 'D1', 'G0', 'G1']: continue
            
            raw_x = float(row.get('X', 0))
            raw_y = float(row.get('Y', 0))
            cx, cy = to_canvas(raw_x, raw_y)
            
            if cmd in ['A0', 'G0']:
                prev_x, prev_y = cx, cy
                
            elif cmd in ['A1', 'G1']:
                if prev_x is not None:
                    z = float(row.get('Z', 0))
                    f = row.get('F', 1000)
                    if f == '': f = 1000
                    f = float(f)
                    
                    width = max(1, z)
                    norm_f = min(1.0, max(0.0, (f - 500) / 3500))
                    r = int(255 * norm_f)
                    b = int(255 * (1 - norm_f))
                    color = f"#{r:02x}00{b:02x}"
                    
                    self.canvas.create_line(prev_x, prev_y, cx, cy, width=width, fill=color, capstyle=tk.ROUND)
                
                prev_x, prev_y = cx, cy
            
            elif cmd == 'D1':
                r = 3
                self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="black")
                prev_x, prev_y = cx, cy

if __name__ == "__main__":
    app = PlotterCsvEnhancer()
    app.mainloop()