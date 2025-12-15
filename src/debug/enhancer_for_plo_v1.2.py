import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
import math
import copy

# --- 1. 永字八法レシピ ---
# PDFの6枚目の表に基づき、音響パラメータ(Speaker)の開始値と終了値を設定
# Format: [Val1, Val2, Val3, Val4]
# Index:  [data8, data9, data10, data11]
# Val1: Freq/Filter, Val2: Duration/Mod, Val3: Volume/Mode, Val4: WaveType/Magnification

EIGHT_STROKES = {
    "側 (点)": {
        "Type": "A2",
        "Start_Spk": [180, 50, 220, 2],
        "End_Spk":   [180, 50, 220, 2],
        "Special": {}
    },
    "勒 (横画)": {
        "Type": "A2",
        "Start_Spk": [130, 300, 240, 1],
        "End_Spk":   [130, 300, 240, 1],
        "Special": {"END_A3": True}
    },
    "努 (縦画)": {
        "Type": "A2",
        "Start_Spk": [80, 400, 255, 1],
        "End_Spk":   [80, 400, 255, 1],
        "Special": {"END_A3": True}
    },
    "趯 (跳ね)": {
        "Type": "A2",
        "Start_Spk": [80, 150, 255, 3],
        "End_Spk":   [200, 150, 100, 3],
        "Special": {"END_A3": True}
    },
    "策 (短横画)": {
        "Type": "A2",
        "Start_Spk": [160, 150, 150, 1],
        "End_Spk":   [160, 150, 250, 1],
        "Special": {}
    },
    "掠 (左はらい)": {
        "Type": "A2",
        "Start_Spk": [120, 300, 180, 1],
        "End_Spk":   [120, 300, 0, 1],
        "Special": {"END_A3": True}
    },
    "啄 (短いはらい)": {
        "Type": "A2",
        "Start_Spk": [170, 80, 200, 2],
        "End_Spk":   [170, 80, 200, 2],
        "Special": {}
    },
    "磔 (右はらい)": {
        "Type": "A2",
        "Start_Spk": [90, 500, 100, 1],
        "End_Spk":   [90, 500, 255, 1],
        "Special": {"SPLIT_D1": True, "END_A3": True}
    },
    "反捺 (長点)": {
        "Type": "A2",
        "Start_Spk": [100, 400, 100, 1],
        "End_Spk":   [100, 400, 220, 1],
        "Special": {"END_A3": True}
    }
}

DEFAULT_VAL1 = 130 # data8
DEFAULT_VAL2 = 300 # data9
DEFAULT_VAL3 = 2   # data10 (固定)
DEFAULT_VAL4 = 100 # data11 (Min 100)
DEFAULT_SPEAKER_PARAMS = f"{DEFAULT_VAL1} {DEFAULT_VAL2} {DEFAULT_VAL3} {DEFAULT_VAL4}"

class PlotterCsvEnhancer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("プロッタ用 永字八法エンハンサー (Z:4-8 / data8,9,11>=100 / data10=2)")
        self.geometry("1200x800")

        self.CANVAS_WIDTH = 600
        self.CANVAS_HEIGHT = 600
        self.COORD_LIMIT = 200.0 
        
        self.rows = []
        self.headers = []
        self.line_ids = {} 

        self.create_widgets()

    def create_widgets(self):
        left_frame = tk.Frame(self, padx=10, pady=10, width=450)
        left_frame.pack(side="left", fill="y")
        
        right_frame = tk.Frame(self, padx=10, pady=10)
        right_frame.pack(side="right", fill="both", expand=True)

        # --- 1. ファイル操作 ---
        file_frame = tk.LabelFrame(left_frame, text="1. ファイル操作", padx=5, pady=5)
        file_frame.pack(fill="x", pady=5)
        
        btn_frame = tk.Frame(file_frame)
        btn_frame.pack(fill="x")
        
        # CSV読込ボタン
        tk.Button(btn_frame, text="CSVを開く", command=self.load_csv, bg="#ddd").pack(side="left", padx=5)
        
        # ★ TXT保存ボタン (メイン)
        tk.Button(btn_frame, text="保存 (TXT形式)", command=self.save_txt_for_plotter, bg="#FF9800", fg="white").pack(side="left", padx=5)
        
        self.lbl_status = tk.Label(file_frame, text="未読み込み")
        self.lbl_status.pack(anchor="w", padx=5)

        # --- 2. データリスト ---
        list_frame = tk.LabelFrame(left_frame, text="2. データ選択 (ストローク単位)", padx=5, pady=5)
        list_frame.pack(fill="both", expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        self.listbox = tk.Listbox(list_frame, selectmode="extended", yscrollcommand=scrollbar.set, font=("Consolas", 10))
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        self.listbox.bind('<<ListboxSelect>>', self.on_select_list)

        # --- 3. 技法適用 ---
        convert_frame = tk.LabelFrame(left_frame, text="3. 永字八法 適用", padx=5, pady=5)
        convert_frame.pack(fill="x", pady=5)

        tk.Label(convert_frame, text="技法:").pack(side="left")
        self.stroke_var = tk.StringVar(self)
        self.stroke_var.set(list(EIGHT_STROKES.keys())[0])
        stroke_menu = tk.OptionMenu(convert_frame, self.stroke_var, *EIGHT_STROKES.keys())
        stroke_menu.pack(side="left", padx=5)

        # 動的補正の有無
        self.use_dynamic_var = tk.BooleanVar(value=True)
        tk.Checkbutton(convert_frame, text="F/距離で音を動的補正", variable=self.use_dynamic_var).pack(anchor="w", padx=5)

        tk.Button(convert_frame, text="適用", command=self.apply_enhancement, bg="#4CAF50", fg="white").pack(side="left", padx=10)

        # --- 4. キャンバス ---
        self.canvas = tk.Canvas(right_frame, bg="white", width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)
        self.canvas.pack(fill="both", expand=True)
        
        legend_frame = tk.Frame(right_frame)
        legend_frame.pack(fill="x")
        tk.Label(legend_frame, text="凡例: 線色=速度(青→赤), 選択中=水色強調, 座標:右上が原点(0,0), 左下が(-200,-200)").pack()

    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath: return

        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                self.headers = reader.fieldnames
                self.rows = [row for row in reader]
            
            for row in self.rows:
                # コマンド正規化
                if 'Command' not in row and 'event_type' in row:
                    evt = row['event_type']
                    row['Command'] = 'A1' if evt == 'move' else 'A0'
                
                # 数値変換
                for k in ['X', 'Y', 'Z', 'F', 'Delay_ms', 'Cell_ID']:
                    val = None
                    for key_candidate in [k, k.lower(), k.upper()]:
                        if key_candidate in row:
                            val = row[key_candidate]
                            break
                    
                    if val is not None and val != '':
                        try:
                            f_val = float(val)
                            
                            # ★★★ 筆圧(Z)の調整: +2.0下駄履き & 4.0-8.0クリップ ★★★
                            if k == 'Z':
                                if f_val > 0.001:
                                    shifted_val = f_val + 2.0
                                    f_val = max(4.0, min(8.0, shifted_val))
                            
                            row[k] = f_val
                        except: pass
                
                if 'Z_orig' not in row:
                    row['Z_orig'] = row.get('Z', 0.0)
                
                if 'SpeakerParams' not in row:
                    row['SpeakerParams'] = DEFAULT_SPEAKER_PARAMS
                
                if 'SpkCmdType' not in row:
                    row['SpkCmdType'] = 'A2'

            self.lbl_status.config(text=f"{len(self.rows)}行")
            self.draw_trajectory()
            self.update_listbox()
            messagebox.showinfo("成功", "CSVを読み込みました。\n(Z値を4.0〜8.0に調整済)")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    # ★ F値からDelay(ms)を計算する関数
    def calculate_delay_from_feed(self, f_val):
        if f_val is None or f_val == '' or float(f_val) == 0:
            return 100 # デフォルト
        
        f = float(f_val)
        # Delay = 1,500,000 / F
        delay = 1500000.0 / f
        return int(delay)

    # ★ TXT形式で保存するメソッド
    def save_txt_for_plotter(self):
        if not self.rows: return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not filepath: return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for i, row in enumerate(self.rows):
                    cmd = row.get('Command', 'A0')
                    
                    x = row.get('X', 0.0)
                    y = row.get('Y', 0.0)
                    z = row.get('Z', 0.0)
                    f_val = row.get('F', '')
                    
                    cell_id = int(row.get('Cell_ID', 0))
                    
                    speaker_params = row.get('SpeakerParams', DEFAULT_SPEAKER_PARAMS)
                    spk_type = row.get('SpkCmdType', 'A2')

                    # F値からDelayを計算 (A2用)
                    spk_delay = self.calculate_delay_from_feed(f_val)
                    
                    # G1行用のスピーカコマンド: A2 Cell Delay Params...
                    spk_cmd = f"{spk_type} {cell_id} {spk_delay} {speaker_params}"

                    if cmd in ['A0', 'G0']:
                        # --- G0 (移動) 行 ---
                        next_cell_id = 0
                        next_delay = 0
                        next_val1 = DEFAULT_VAL1

                        found_next = False
                        for j in range(i + 1, len(self.rows)):
                            next_row = self.rows[j]
                            next_cmd = next_row.get('Command', '')
                            if next_cmd in ['A1', 'G1']:
                                next_cell_id = int(next_row.get('Cell_ID', 0))
                                
                                next_f = next_row.get('F', '')
                                next_delay = self.calculate_delay_from_feed(next_f)
                                
                                next_params = next_row.get('SpeakerParams', DEFAULT_SPEAKER_PARAMS)
                                try:
                                    # next_val1 (data8) も最低100のチェックを通しておく（念のため）
                                    v1_candidate = int(next_params.split()[0])
                                    next_val1 = max(100, v1_candidate)
                                except: pass
                                found_next = True
                                break
                        
                        line_g0 = f"G0 X{x:.2f} Y{y:.2f} Z{z:.2f}"
                        if found_next:
                             line_g0 += f" A0 {next_cell_id} {next_delay} {next_val1}"
                        f.write(line_g0 + "\n")
                        
                    elif cmd in ['A1', 'G1']: 
                        # --- G1 (描画) 行 ---
                        line = f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f}"
                        if f_val != '':
                            line += f" F{int(f_val)}"
                        line += f"\t{spk_cmd}" 
                        f.write(line + "\n")
                        
                    elif cmd == 'D1':
                        d_ms = row.get('Delay_ms', '')
                        if d_ms != '':
                            f.write(f"D1 {int(d_ms)}\n")
                    
                    elif cmd == 'A3':
                        f.write("A3\n")

            messagebox.showinfo("成功", f"プロッタ用TXTファイルを保存しました。\n{filepath}")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def _interpolate(self, start_val, end_val, progress):
        return start_val + (end_val - start_val) * progress

    def calculate_modulated_params(self, base_params_str, f_val, distance):
        try:
            base_vals = [float(x) for x in base_params_str.split()]
            if len(base_vals) < 4: return base_params_str
        except:
            return base_params_str

        f_num = float(f_val) if f_val != '' else 1000.0
        
        # 補正ロジック (Val1, Val2のみ動かす)
        freq_mod = (f_num - 1000.0) * 0.05
        new_val1 = base_vals[0] + freq_mod
        # ★ data8 (Val1) Min 100
        new_val1 = max(100, min(800, new_val1)) 

        dist_mod = distance * 2.0
        new_val2 = base_vals[1] + dist_mod
        # ★ data9 (Val2) Min 100
        new_val2 = max(100, min(1000, new_val2)) 

        # Val3, Val4 はベース値を維持 (Base値はすでに apply_enhancement で min100/fixed処理済み)
        new_val3 = base_vals[2]
        new_val4 = base_vals[3]

        return f"{int(new_val1)} {int(new_val2)} {int(new_val3)} {int(new_val4)}"

    def apply_enhancement(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "変換したい範囲を選択してください。")
            return

        stroke_name = self.stroke_var.get()
        recipe = EIGHT_STROKES[stroke_name]
        use_dynamic = self.use_dynamic_var.get()
        
        indices = sorted(list(selection))
        target_indices = [i for i in indices if self.rows[i].get('Command') in ['A1', 'G1']]
        
        if not target_indices:
            messagebox.showwarning("警告", "選択範囲に描画コマンド(A1/G1)が含まれていません。")
            return

        start_row_num = target_indices[0] + 1
        end_row_num = target_indices[-1] + 1
        
        # 経過時間の計算
        total_duration = 0
        durations = []
        for i in target_indices:
            f_val = self.rows[i].get('F', '')
            d = self.calculate_delay_from_feed(f_val)
            durations.append(d)
            total_duration += d
        
        if total_duration == 0: total_duration = 1

        split_d1_inserted = False
        current_time = 0
        
        processed_rows = [] 
        processed_rows.extend(self.rows[:indices[0]])

        for i, original_idx in enumerate(indices):
            row = self.rows[original_idx]
            cmd = row.get('Command')
            
            if cmd not in ['A1', 'G1']:
                processed_rows.append(row)
                continue
            
            t_idx = target_indices.index(original_idx)
            duration = durations[t_idx]
            progress = (current_time + (duration / 2)) / total_duration
            
            # 特殊処理: 磔 (SPLIT_D1)
            if recipe["Special"].get("SPLIT_D1") and not split_d1_inserted:
                if progress >= 0.60:
                    d1_row = {
                        'Command': 'D1',
                        'Delay_ms': 500, 
                        'X': row.get('X'), 'Y': row.get('Y'),
                        'Z': 0, 'F': '', 'Cell_ID': row.get('Cell_ID'),
                        'SpeakerParams': DEFAULT_SPEAKER_PARAMS
                    }
                    processed_rows.append(d1_row)
                    split_d1_inserted = True
            
            start_spk = recipe["Start_Spk"]
            end_spk = recipe["End_Spk"]
            
            current_spk = []
            for k in range(4):
                val = self._interpolate(start_spk[k], end_spk[k], progress)
                
                # ★★★ 修正箇所: 値の制限と固定 ★★★
                if k == 0: # data[8]
                    val = max(100, val)
                elif k == 1: # data[9]
                    val = max(100, val)
                elif k == 2: # data[10] -> 2に固定
                    val = 2
                elif k == 3: # data[11] -> 以前のロジック(補間)を使用、ただしMin 100
                    val = max(100, val)
                
                current_spk.append(int(val))
            
            base_spk_str = f"{current_spk[0]} {current_spk[1]} {current_spk[2]} {current_spk[3]}"
            row['SpkCmdType'] = recipe["Type"] 

            if use_dynamic:
                prev_x = 0.0
                prev_y = 0.0
                if original_idx > 0:
                    prev_x = float(self.rows[original_idx-1].get('X', 0))
                    prev_y = float(self.rows[original_idx-1].get('Y', 0))
                
                curr_x = float(row.get('X', 0))
                curr_y = float(row.get('Y', 0))
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                distance = math.sqrt(dx*dx + dy*dy)
                f_val = row.get('F', '')

                row['SpeakerParams'] = self.calculate_modulated_params(base_spk_str, f_val, distance)
            else:
                row['SpeakerParams'] = base_spk_str
            
            processed_rows.append(row)
            current_time += duration

        if recipe["Special"].get("END_A3"):
            a3_row = {
                'Command': 'A3',
                'X': processed_rows[-1].get('X'),
                'Y': processed_rows[-1].get('Y'),
                'Z': 0, 'F': '', 'Delay_ms': '',
                'Cell_ID': processed_rows[-1].get('Cell_ID'),
                'SpeakerParams': DEFAULT_SPEAKER_PARAMS
            }
            processed_rows.append(a3_row)

        processed_rows.extend(self.rows[indices[-1]+1:])
        self.rows = processed_rows
        
        self.draw_trajectory()
        self.update_listbox()
        
        msg = f"{stroke_name} を適用しました。\n(data8,9,11>=100, data10=2)"
        messagebox.showinfo("完了", msg)

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, row in enumerate(self.rows):
            cmd = row.get('Command', '')
            x = row.get('X', 0)
            y = row.get('Y', 0)
            z = row.get('Z', 0)
            f = row.get('F', '')
            f_str = f"{int(f)}" if f!='' and f is not None else ""
            z_str = f"{float(z):.1f}"
            
            spk_type = row.get('SpkCmdType', '')
            spk_params = row.get('SpeakerParams', '')
            
            prefix = ""
            if cmd == 'D1': prefix = "[待機] "
            if cmd == 'A3': prefix = "[停止] "
            
            txt = f"{i+1:03d}: {prefix}{cmd} | Z:{z_str} F:{f_str} | {spk_type} {spk_params}"
            self.listbox.insert(tk.END, txt)
            
            if cmd in ['D1', 'A3']:
                self.listbox.itemconfig(tk.END, {'bg': '#ffffe0'}) 

    def on_select_list(self, event):
        selected_indices = self.listbox.curselection()
        if not selected_indices: return
        self.canvas.dtag("selected", "selected") 
        self.draw_trajectory(selected_indices)

    def draw_trajectory(self, selected_indices=None):
        self.canvas.delete("all")
        self.line_ids = {} 
        
        scale_x = self.CANVAS_WIDTH / abs(self.COORD_LIMIT)
        scale_y = self.CANVAS_HEIGHT / abs(self.COORD_LIMIT)
        
        scale_x *= 0.9
        scale_y *= 0.9
        offset_x = self.CANVAS_WIDTH * 0.05
        offset_y = self.CANVAS_HEIGHT * 0.05

        def to_canvas(cx, cy):
            x = (cx + 200) * scale_x + offset_x
            y = (-cy) * scale_y + offset_y
            return x, y

        prev_x, prev_y = None, None
        
        self.canvas.create_line(0, offset_y, self.CANVAS_WIDTH, offset_y, fill="#ddd") 
        self.canvas.create_line(self.CANVAS_WIDTH-offset_x, 0, self.CANVAS_WIDTH-offset_x, self.CANVAS_HEIGHT, fill="#ddd") 

        for i, row in enumerate(self.rows):
            cmd = row.get('Command')
            if cmd not in ['A0', 'A1', 'D1', 'G0', 'G1', 'A3']: continue
            
            raw_x = float(row.get('X', 0))
            raw_y = float(row.get('Y', 0))
            x, y = to_canvas(raw_x, raw_y)
            
            is_selected = selected_indices and (i in selected_indices)
            
            if cmd in ['A0', 'G0']:
                prev_x, prev_y = x, y
                
            elif cmd in ['A1', 'G1']:
                if prev_x is not None:
                    z = float(row.get('Z', 0))
                    f = row.get('F', 1000)
                    if f == '': f = 1000
                    f = float(f)
                    
                    width = max(1, z * 1.5)
                    if is_selected:
                        width += 2 
                    
                    if is_selected:
                        color = "#00FFFF" 
                    else:
                        norm_f = min(1.0, max(0.0, (f - 300) / 3700))
                        r = int(255 * norm_f)
                        b = int(255 * (1 - norm_f))
                        color = f"#{r:02x}00{b:02x}"
                    
                    line_id = self.canvas.create_line(prev_x, prev_y, x, y, 
                                        width=width, fill=color, capstyle=tk.ROUND)
                    self.line_ids[i] = line_id 
                
                prev_x, prev_y = x, y
            
            elif cmd == 'D1':
                r = 3
                fill_c = "red" if is_selected else "black"
                self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=fill_c, outline=fill_c)
                prev_x, prev_y = x, y
                
            elif cmd == 'A3':
                r = 4
                fill_c = "#00FFFF" if is_selected else "red"
                self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=fill_c, outline=fill_c)
                prev_x, prev_y = x, y
            
            else:
                prev_x, prev_y = x, y

if __name__ == "__main__":
    app = PlotterCsvEnhancer()
    app.mainloop()