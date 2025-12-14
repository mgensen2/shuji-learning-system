import tkinter as tk
from tkinter import filedialog, messagebox, ttk # ttkをインポート
import csv
import math
import re
import copy

# --- 1. 永字八法コマンド変換レシピ (V_adj=2, Ratio >= 100) ---
EIGHT_STROKES = {
    "側 (点)": {
        "Type": "A2", "V_adj": 2, "Ratio": 150, 
        "Start": {"V_diff": 15.0, "F": 2000, "W": 200},
        "End":   {"V_diff": 15.0, "F": 2000, "W": 200},
        "Special": {}
    },
    "勒 (横画)": {
        "Type": "A2", "V_adj": 2, "Ratio": 100,
        "Start": {"V_diff": -5.0, "F": 500, "W": 300},
        "End":   {"V_diff": -5.0, "F": 500, "W": 300},
        "Special": {"END_A3": True}
    },
    "努 (縦画)": {
        "Type": "A2", "V_adj": 2, "Ratio": 120,
        "Start": {"V_diff": 10.0, "F": 300, "W": 500},
        "End":   {"V_diff": 10.0, "F": 300, "W": 500},
        "Special": {"END_A3": True}
    },
    "趯 (跳ね)": {
        "Type": "A2", "V_adj": 2, 
        "Ratio_Start": 150, "Ratio_End": 300, 
        "Start": {"V_diff": 10.0, "F": 2000, "W": 1000},
        "End":   {"V_diff": 20.0, "F": 4000, "W": 1500},
        "Special": {"END_A3": True}
    },
    "策 (短横画)": {
        "Type": "A2", "V_adj": 2, "Ratio": 100, 
        "Start": {"V_diff": -10.0, "F": 1500, "W": 500},
        "End":   {"V_diff": -10.0, "F": 1500, "W": 500},
        "Special": {}
    },
    "掠 (左はらい)": {
        "Type": "A2", "V_adj": 2, 
        "Ratio_Start": 300, "Ratio_End": 100, 
        "Start": {"V_diff": 10.0, "F": 700, "W": 1200},
        "End":   {"V_diff": -15.0, "F": 1500, "W": 1200},
        "Special": {"END_A3": True}
    },
    "啄 (短いはらい)": {
        "Type": "A2", "V_adj": 2, "Ratio": 180, 
        "Start": {"V_diff": 15.0, "F": 2500, "W": 400},
        "End":   {"V_diff": 15.0, "F": 2500, "W": 400},
        "Special": {}
    },
    "磔 (右はらい)": {
        "Type": "A2", "V_adj": 2, 
        "Ratio_Start": 100, "Ratio_End": 250, 
        "Start": {"V_diff": -10.0, "F": 600, "W": 300},
        "End":   {"V_diff": 20.0, "F": 1500, "W": 1000},
        "Special": {"SPLIT_D1": True, "END_A3": True}
    }
}

# その他の定数
DEFAULT_VAL3 = 14 
DEFAULT_VAL4 = 0 
DEFAULT_SPEAKER_PARAMS = f"0 0 0 0 0 0" 
MIN_FREQ_W = 100 # FとWの最小値を100に保証する

class PlotterCsvEnhancer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("プロッタ用 永字八法エンハンサー (最終確定版/ttk)")
        self.geometry("1200x800")

        self.CANVAS_WIDTH = 600
        self.CANVAS_HEIGHT = 600
        self.COORD_LIMIT = 200.0
        self.MAX_LINE_WIDTH = 8
        self.MIN_LINE_WIDTH = 2
        self.MAX_FREQ = 4000
        self.MIN_FREQ = 300
        
        self.rows = []
        self.headers = []
        self.line_ids = {} 

        self.create_widgets()

    def create_widgets(self):
        # ttk.Frameを使用
        left_frame = ttk.Frame(self, padding="10", width=450)
        left_frame.pack(side="left", fill="y")
        
        right_frame = ttk.Frame(self, padding="10")
        right_frame.pack(side="right", fill="both", expand=True)

        # 1. ファイル操作 (ttk.LabelFrameを使用)
        file_frame = ttk.LabelFrame(left_frame, text="1. ファイル操作", padding="5")
        file_frame.pack(fill="x", pady=5)
        
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill="x")
        # ttk.Buttonを使用
        ttk.Button(btn_frame, text="CSVを開く", command=self.load_csv).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="CSV保存", command=self.save_csv).pack(side="left", padx=5)
        
        ttk.Button(file_frame, text="TXT保存 (プロッタ用)", command=self.save_txt_for_plotter, style="Accent.TButton").pack(fill="x", padx=5, pady=5)
        
        # ttk.Labelを使用
        self.lbl_status = ttk.Label(file_frame, text="未読み込み")
        self.lbl_status.pack(anchor="w", padx=5)

        # 2. データリスト (ttk.LabelFrameを使用)
        list_frame = ttk.LabelFrame(left_frame, text="2. データ選択 (ストローク単位)", padding="5")
        list_frame.pack(fill="both", expand=True, pady=5)
        
        # ttk.Scrollbarを使用
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        # Listboxはtk.Listboxを維持
        self.listbox = tk.Listbox(list_frame, selectmode="extended", yscrollcommand=scrollbar.set, font=("Consolas", 10), height=15)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        self.listbox.bind('<<ListboxSelect>>', self.on_select_list)

        # 3. 技法適用 (ttk.LabelFrameを使用)
        convert_frame = ttk.LabelFrame(left_frame, text="3. 永字八法 適用", padding="5")
        convert_frame.pack(fill="x", pady=5)

        ttk.Label(convert_frame, text="技法:").pack(side="left")
        self.stroke_var = tk.StringVar(self)
        
        # ttk.Comboboxを使用 (OptionMenuの代替、ルックアンドフィールが良い)
        self.stroke_var.set(list(EIGHT_STROKES.keys())[0])
        self.stroke_combobox = ttk.Combobox(convert_frame, textvariable=self.stroke_var, values=list(EIGHT_STROKES.keys()), state="readonly", width=15)
        self.stroke_combobox.pack(side="left", padx=5)

        ttk.Button(convert_frame, text="適用", command=self.apply_enhancement, style="Accent.TButton").pack(side="left", padx=10)

        # 4. キャンバス
        self.canvas = tk.Canvas(right_frame, bg="white", width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(fill="both", expand=True)
        
        legend_frame = ttk.Frame(right_frame)
        legend_frame.pack(fill="x")
        ttk.Label(legend_frame, text="凡例: 線色=周波数(青(低F)→赤(高F)), 線幅=Z値(ペン圧), 座標:右上が原点(0,0)").pack()
        
        # ttkのスタイル設定 (アクセントカラーの定義)
        style = ttk.Style(self)
        style.configure("Accent.TButton", foreground='white', background='#4CAF50')


    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath: return

        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                self.headers = reader.fieldnames
                self.rows = [row for row in reader]
            
            for row in self.rows:
                for k in ['X', 'Y', 'Z', 'F', 'Delay_ms', 'Cell_ID']:
                    val = None
                    for key_candidate in [k, k.lower(), k.upper()]:
                        if key_candidate in row:
                            val = row[key_candidate]
                            break
                    if val is not None and val != '':
                        try: row[k] = float(val)
                        except: pass
                
                if 'Z_orig' not in row: row['Z_orig'] = row.get('Z', 0.0)
                if 'SpeakerParams' not in row: row['SpeakerParams'] = DEFAULT_SPEAKER_PARAMS
                if 'SpkCmdType' not in row: row['SpkCmdType'] = 'A2'
                if 'Command' not in row: row['Command'] = row.get('mode', 'G1')
                
            self.lbl_status.config(text=f"ファイル読み込み完了 ({len(self.rows)}行)")
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
                            elif k in ['F', 'Delay_ms']: val = f"{int(val)}"
                            else: val = str(val)
                        out_row[k] = val
                    writer.writerow(out_row)
            messagebox.showinfo("成功", "CSVを保存しました。")
        except Exception as e:
            messagebox.showerror("エラー", str(e))
            
    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, row in enumerate(self.rows):
            cmd = row.get('Command', '')
            z = row.get('Z', 0)
            f = row.get('F', '')
            f_str = f"{int(f)}" if f!='' and f is not None else ""
            z_str = f"{float(z):.1f}"
            
            spk_type = row.get('SpkCmdType', '')
            spk_params = row.get('SpeakerParams', '')
            
            prefix = ""
            if cmd == 'D1': prefix = "[待機] "
            if cmd == 'A3': prefix = "[停止] "
            
            txt = f"{i+1:03d}: {prefix}{cmd} | Z:{z_str} F:{f_str} | {spk_type} ({spk_params})"
            self.listbox.insert(tk.END, txt)
            
            if cmd in ['D1', 'A3']:
                self.listbox.itemconfig(tk.END, {'bg': '#ffffe0'}) 

    def on_select_list(self, event):
        selected_indices = self.listbox.curselection()
        if not selected_indices: return
        self.draw_trajectory(selected_indices)

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

        start_row_num = target_indices[0] + 1
        end_row_num = target_indices[-1] + 1
        
        total_duration = 0
        for i in target_indices:
            d = self.rows[i].get('Delay_ms', 0)
            if d == '': d = 0
            total_duration += float(d)
        
        if total_duration == 0: total_duration = 1

        processed_rows = [] 
        processed_rows.extend(self.rows[:indices[0]])

        split_d1_inserted = False
        current_time = 0
        
        is_split_stroke = recipe["Special"].get("SPLIT_D1", False)
        
        # Ratio/F/Wの開始・終了値を設定
        is_ratio_interpolated = "Ratio_Start" in recipe
        if is_ratio_interpolated:
            Ratio_start = recipe['Ratio_Start']
            Ratio_end = recipe['Ratio_End']
        else:
            Ratio_fixed = recipe['Ratio']
            Ratio_start = Ratio_fixed
            Ratio_end = Ratio_fixed
            
        V_start, V_end = recipe["Start"]["V_diff"], recipe["End"]["V_diff"]
        F_start, F_end = recipe["Start"]["F"], recipe["End"]["F"]
        W_start, W_end = recipe["Start"]["W"], recipe["End"]["W"]
        
        V_adj = 2 # data[10]は常に2に固定

        for original_idx in target_indices:
            row = self.rows[original_idx]
            duration = row.get('Delay_ms', 0)
            duration = float(duration)
            
            # --- 磔（D1挿入）の判定と処理 ---
            if is_split_stroke and not split_d1_inserted:
                if (current_time + duration / 2) / total_duration >= 0.6: # D1_pos=0.6
                    # D1の挿入
                    d1_row = copy.deepcopy(row)
                    d1_row['Command'] = 'D1'
                    d1_row['Delay_ms'] = 300 
                    d1_row['Z'] = 0.0
                    d1_row['F'] = ''
                    d1_row['SpkCmdType'] = ''
                    d1_row['SpeakerParams'] = ''
                    processed_rows.append(d1_row)
                    split_d1_inserted = True
                    
                    # フェーズ2の開始値は、レシピのEnd値（F, W, Ratio）を開始として使用
                    F_start, F_end = recipe["End"]["F"], recipe["End"]["F"]
                    W_start, W_end = recipe["End"]["W"], recipe["End"]["W"]
                    
                    if is_ratio_interpolated:
                        Ratio_start = recipe["Ratio_End"]
                        
                    current_time = 0 
                    total_duration = total_duration * 0.4 

            # --- 補間計算 ---
            progress = (current_time + duration / 2) / total_duration
            progress = min(1.0, max(0.0, progress)) 

            # バンドパスパラメータの補間
            f_interp = self._interpolate(F_start, F_end, progress)
            w_interp = self._interpolate(W_start, W_end, progress)
            
            # Ratioの補間
            ratio_interp = self._interpolate(Ratio_start, Ratio_end, progress)
            
            # FとWが100未満にならないよう保証 (最小値保証)
            f_interp = max(MIN_FREQ_W, f_interp)
            w_interp = max(MIN_FREQ_W, w_interp)
            
            # 16bit データの分割
            f_high = int(f_interp) >> 8
            f_low = int(f_interp) & 0xFF
            w_high = int(w_interp) >> 8
            w_low = int(w_interp) & 0xFF
            
            # SpeakerParams の構築 (data[6]〜data[11])
            speaker_params = (
                f"{f_high} {f_low} {w_high} {w_low} "
                f"{V_adj} {int(ratio_interp)}" # V_adj=2, 補間されたRatioを使用
            )
            
            # 行データの更新 (ZとFは元の値を維持)
            row['SpkCmdType'] = 'A2'
            row['SpeakerParams'] = speaker_params
            
            processed_rows.append(row)
            current_time += duration

        # 適用範囲後の行
        processed_rows.extend(self.rows[target_indices[-1]+1:])
        
        # --- A3 (停止) の挿入 ---
        if recipe["Special"].get("END_A3"):
            last_row = processed_rows[-1]
            a3_row = copy.deepcopy(last_row)
            a3_row['Command'] = 'A3'
            a3_row['Z'] = 0.0 
            a3_row['F'] = ''
            a3_row['Delay_ms'] = ''
            a3_row['SpkCmdType'] = 'A3' # モード17
            a3_row['SpeakerParams'] = speaker_params 
            processed_rows.append(a3_row)
            
        self.rows = processed_rows
        
        self.draw_trajectory()
        self.update_listbox()
        
        msg = f"{stroke_name} を適用しました。\nZ値とF値は元の教師データを維持しています。"
        messagebox.showinfo("完了", msg)

    # --- TXTファイル出力メソッド (スピーカアレイ準拠) ---
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
                    
                    speaker_params = row.get('SpeakerParams', "") 
                    spk_type_str = row.get('SpkCmdType', 'A0') 

                    spk_delay = int(delay_ms) if delay_ms != '' and delay_ms > 0 else 100
                    
                    spk_cmd_line = ""
                    
                    if spk_type_str == 'A2':
                        # モード14 (バンドパス再生)
                        base_params = "0 14" # data[2]=0, data[3]=14
                        spk_cmd = f"1 14 {base_params} {cell_id} 0 {speaker_params}"
                        spk_cmd_line = f"S {spk_cmd}"
                        
                    elif spk_type_str == 'A3':
                        # モード17 (バンドパス停止)
                        spk_cmd = f"1 17 0 17 {cell_id} 0 {speaker_params}" 
                        spk_cmd_line = f"S {spk_cmd}"
                        f.write(f"{spk_cmd_line}\n")
                        f.write(f"D1 10\n") 
                        continue 
                    
                    # --- G/A コマンドの出力 (Z, F値は元のCSVデータをそのまま使用) ---
                    if cmd in ['A0', 'G0']:
                        f.write(f"G0 X{x:.2f} Y{y:.2f} Z{z:.2f}\n")
                        if spk_cmd_line:
                            f.write(f"{spk_cmd_line}\n") 
                        
                    elif cmd in ['A1', 'G1']: 
                        line = f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f}"
                        if f_val != '':
                            line += f" F{int(f_val)}"
                        
                        if spk_cmd_line:
                            f.write(line + "\n")
                            f.write(f"{spk_cmd_line}\n") 
                        else:
                            f.write(line + "\n")
                        
                    elif cmd == 'D1':
                        if delay_ms != '':
                            f.write(f"D1 {int(delay_ms)}\n")

            messagebox.showinfo("成功", f"プロッタ用TXTファイルを保存しました。")
        except Exception as e:
            messagebox.showerror("エラー", str(e))
            
    # --- 軌跡描画メソッド ---
    def get_color_from_freq(self, freq):
        """周波数に基づいて色のRBG値を計算 (低F=青, 高F=赤)"""
        norm_freq = (freq - self.MIN_FREQ) / (self.MAX_FREQ - self.MIN_FREQ)
        norm_freq = max(0, min(1, norm_freq)) 

        r = int(255 * norm_freq)       
        g = int(255 * (1 - abs(norm_freq - 0.5) * 2)) 
        b = int(255 * (1 - norm_freq)) 
        
        return f'#{r:02x}{g:02x}{b:02x}'


    def draw_trajectory(self, selected_indices=None):
        self.canvas.delete("all")
        self.line_ids = {} 
        
        scale_x = self.CANVAS_WIDTH / abs(self.COORD_LIMIT) * 0.9
        scale_y = self.CANVAS_HEIGHT / abs(self.COORD_LIMIT) * 0.9
        offset_x = self.CANVAS_WIDTH * 0.05
        offset_y = self.CANVAS_HEIGHT * 0.05

        def to_canvas(cx, cy):
            x = (cx + self.COORD_LIMIT) * scale_x + offset_x
            y = (self.COORD_LIMIT - (-cy)) * scale_y + offset_y 
            return x, y

        prev_x, prev_y = None, None
        
        self.canvas.create_line(offset_x, 0, offset_x, self.CANVAS_HEIGHT, fill="#ddd") 
        self.canvas.create_line(0, self.CANVAS_HEIGHT - offset_y, self.CANVAS_WIDTH, self.CANVAS_HEIGHT - offset_y, fill="#ddd") 

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
                    f = float(row.get('F', 1000))
                    
                    width = max(self.MIN_LINE_WIDTH, min(self.MAX_LINE_WIDTH, self.MIN_LINE_WIDTH + z * 0.75))
                    if is_selected:
                        width += 2 
                    
                    if is_selected:
                        color = "#00FFFF" 
                    else:
                        color = self.get_color_from_freq(f)
                    
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

        if prev_x is not None:
            self.canvas.create_oval(prev_x-2, prev_y-2, prev_x+2, prev_y+2, fill="black")

if __name__ == "__main__":
    app = PlotterCsvEnhancer()
    app.mainloop()