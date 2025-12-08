import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
import math
import copy

# --- 1. 永字八法レシピ ---
# PDFの6枚目の表に基づき、音響パラメータ(Speaker)の開始値と終了値を設定
# Format: [Val1, Val2, Val3, Val4]
# Val1: Freq/Filter, Val2: Duration/Mod, Val3: Volume, Val4: WaveType

EIGHT_STROKES = {
    "側 (点)": {
        "Type": "A2",
        # 変化なし: 高域強調、短く鋭い
        "Start_Spk": [180, 50, 220, 2],
        "End_Spk":   [180, 50, 220, 2],
        "Special": {}
    },
    "勒 (横画)": {
        "Type": "A2",
        # 変化小: 全体的に音量大
        "Start_Spk": [130, 300, 240, 1],
        "End_Spk":   [130, 300, 240, 1],
        "Special": {"END_A3": True}
    },
    "努 (縦画)": {
        "Type": "A2",
        # 変化小: ローパス、音量最大
        "Start_Spk": [80, 400, 255, 1],
        "End_Spk":   [80, 400, 255, 1],
        "Special": {"END_A3": True}
    },
    "趯 (跳ね)": {
        "Type": "A2",
        # 変化大: 溜め(ローパス/大) -> 跳ね(ハイパス/小)
        "Start_Spk": [80, 150, 255, 3],
        "End_Spk":   [200, 150, 100, 3],
        "Special": {"END_A3": True}
    },
    "策 (短横画)": {
        "Type": "A2",
        # 変化あり: 音量 小 -> 大
        "Start_Spk": [160, 150, 150, 1],
        "End_Spk":   [160, 150, 250, 1],
        "Special": {}
    },
    "掠 (左はらい)": {
        "Type": "A2",
        # 変化大: 音量 中 -> 小 (徐々に小さく)
        "Start_Spk": [120, 300, 180, 1],
        "End_Spk":   [120, 300, 0, 1],
        "Special": {"END_A3": True}
    },
    "啄 (短いはらい)": {
        "Type": "A2",
        # 変化なし: 鋭い
        "Start_Spk": [170, 80, 200, 2],
        "End_Spk":   [170, 80, 200, 2],
        "Special": {}
    },
    "磔 (右はらい)": {
        "Type": "A2",
        # 変化大: 音量 小 -> 大 (だんだん大きく)
        "Start_Spk": [90, 500, 100, 1],
        "End_Spk":   [90, 500, 255, 1],
        "Special": {"SPLIT_D1": True, "END_A3": True}
    },
    "反捺 (長点)": {
        "Type": "A2",
        # 変化あり: 音量 小 -> 大
        "Start_Spk": [100, 400, 100, 1],
        "End_Spk":   [100, 400, 220, 1],
        "Special": {"END_A3": True}
    }
}

DEFAULT_VAL1 = 130
DEFAULT_VAL3 = 250
DEFAULT_VAL4 = 2
DEFAULT_SPEAKER_PARAMS = f"{DEFAULT_VAL1} 300 {DEFAULT_VAL3} {DEFAULT_VAL4}"

class PlotterCsvEnhancer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("プロッタ用 音響エンハンサー (Z/F維持・音響補間版)")
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
        tk.Button(btn_frame, text="CSVを開く", command=self.load_csv, bg="#ddd").pack(side="left", padx=5)
        tk.Button(btn_frame, text="CSV保存", command=self.save_csv, bg="#ddd").pack(side="left", padx=5)
        
        tk.Button(file_frame, text="TXT保存 (プロッタ用)", command=self.save_txt_for_plotter, bg="#FF9800", fg="white").pack(fill="x", padx=5, pady=5)
        
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
                if 'Command' not in row and 'event_type' in row:
                    evt = row['event_type']
                    row['Command'] = 'A1' if evt == 'move' else 'A0'
                
                for k in ['X', 'Y', 'Z', 'F', 'Delay_ms', 'Cell_ID']:
                    val = None
                    for key_candidate in [k, k.lower(), k.upper()]:
                        if key_candidate in row:
                            val = row[key_candidate]
                            break
                    
                    if val is not None and val != '':
                        try:
                            row[k] = float(val)
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
            messagebox.showinfo("成功", "CSVを読み込みました。")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def save_csv(self):
        if not self.rows: return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not filepath: return

        try:
            output_headers = ['Command', 'X', 'Y', 'Z', 'F', 'Delay_ms', 'Cell_ID', 'SpeakerParams', 'SpkCmdType', 'Z_orig']
            
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=output_headers, extrasaction='ignore')
                writer.writeheader()
                for row in self.rows:
                    out_row = {}
                    for k in output_headers:
                        val = row.get(k, '')
                        if isinstance(val, float):
                            if k in ['X', 'Y', 'Z', 'Z_orig']: val = f"{val:.2f}"
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
                    
                    x = row.get('X', 0.0)
                    y = row.get('Y', 0.0)
                    z = row.get('Z', 0.0)
                    f_val = row.get('F', '')
                    delay_ms = row.get('Delay_ms', '')
                    cell_id = int(row.get('Cell_ID', 0))
                    
                    speaker_params = row.get('SpeakerParams', DEFAULT_SPEAKER_PARAMS)
                    spk_type = row.get('SpkCmdType', 'A2')

                    spk_delay = int(delay_ms) if delay_ms != '' and delay_ms > 0 else 100
                    spk_cmd = f"{spk_type} {cell_id} {spk_delay} {speaker_params}"

                    if cmd in ['A0', 'G0']:
                        f.write(f"G0 X{x:.2f} Y{y:.2f} Z{z:.2f}\n")
                        f.write(f"S {spk_cmd}\n") 
                        
                    elif cmd in ['A1', 'G1']: 
                        line = f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f}"
                        if f_val != '':
                            line += f" F{int(f_val)}"
                        line += f"\t{spk_cmd}" 
                        f.write(line + "\n")
                        
                    elif cmd == 'D1':
                        if delay_ms != '':
                            f.write(f"D1 {int(delay_ms)}\n")
                    
                    elif cmd == 'A3':
                        f.write("A3\n")

            messagebox.showinfo("成功", f"プロッタ用TXTファイルを保存しました。\n{filepath}")
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def _interpolate(self, start_val, end_val, progress):
        return start_val + (end_val - start_val) * progress

    def apply_enhancement(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "変換したい範囲を選択してください。")
            return

        stroke_name = self.stroke_var.get()
        recipe = EIGHT_STROKES[stroke_name]
        
        indices = sorted(list(selection))
        target_indices = [i for i in indices if self.rows[i].get('Command') in ['A1', 'G1']]
        
        if not target_indices:
            messagebox.showwarning("警告", "選択範囲に描画コマンド(A1/G1)が含まれていません。")
            return

        # 適用範囲の記録 (完了メッセージ用)
        start_row_num = target_indices[0] + 1
        end_row_num = target_indices[-1] + 1
        applied_count = len(target_indices)

        total_duration = 0
        durations = []
        for i in target_indices:
            d = self.rows[i].get('Delay_ms', 0)
            if d == '': d = 0
            d = float(d)
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
            
            # --- 音響パラメータ補間 ---
            start_spk = recipe["Start_Spk"]
            end_spk = recipe["End_Spk"]
            
            # Val1 ~ Val4 それぞれを補間
            current_spk = []
            for k in range(4):
                val = self._interpolate(start_spk[k], end_spk[k], progress)
                current_spk.append(int(val))
            
            # データ更新
            row['SpkCmdType'] = recipe["Type"] 
            row['SpeakerParams'] = f"{current_spk[0]} {current_spk[1]} {current_spk[2]} {current_spk[3]}"
            
            # Z値、F値は変更しない (元の値を維持)
            
            processed_rows.append(row)
            
            current_time += duration

        # 特殊処理: END_A3
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
        
        msg = f"{stroke_name} を適用しました。\n"
        msg += f"範囲: 行 {start_row_num} 〜 {end_row_num} (計{applied_count}行)\n"
        msg += "(Z/F値維持, 音響パラメータ補間, 特殊コマンド挿入)"
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