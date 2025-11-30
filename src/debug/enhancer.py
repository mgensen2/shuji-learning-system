import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv
import math
import copy

# --- 1. 永字八法レシピ ---
EIGHT_STROKES = {
    "側 (点)": {
        "Type": "A2",
        "Start": {"V_diff": 15.0, "F": 2000, "W": 200},
        "End":   {"V_diff": 15.0, "F": 2000, "W": 200},
        "Special": {}
    },
    "勒 (横画)": {
        "Type": "B2",
        "Start": {"V_diff": -5.0, "F": 500, "W": 300},
        "End":   {"V_diff": -5.0, "F": 500, "W": 300},
        "Special": {"END_A3": True}
    },
    "努 (縦画)": {
        "Type": "B2",
        "Start": {"V_diff": 5.0, "F": 300, "W": 500},
        "End":   {"V_diff": 5.0, "F": 300, "W": 500},
        "Special": {"END_A3": True}
    },
    "趯 (跳ね)": {
        "Type": "B2",
        "Start": {"V_diff": 10.0, "F": 2000, "W": 1000},
        "End":   {"V_diff": 20.0, "F": 4000, "W": 1500},
        "Special": {"END_A3": True}
    },
    "策 (短横画)": {
        "Type": "A2",
        "Start": {"V_diff": -10.0, "F": 1500, "W": 500},
        "End":   {"V_diff": -10.0, "F": 1500, "W": 500},
        "Special": {}
    },
    "掠 (左はらい)": {
        "Type": "A2", 
        "Start": {"V_diff": 10.0, "F": 700, "W": 1200},
        "End":   {"V_diff": -15.0, "F": 1500, "W": 1200},
        "Special": {"END_A3": True}
    },
    "啄 (短いはらい)": {
        "Type": "A2",
        "Start": {"V_diff": 15.0, "F": 2500, "W": 400},
        "End":   {"V_diff": 15.0, "F": 2500, "W": 400},
        "Special": {}
    },
    "磔 (右はらい)": {
        "Type": "A2", 
        "Start": {"V_diff": -10.0, "F": 600, "W": 300},
        "End":   {"V_diff": 20.0, "F": 1500, "W": 1000},
        "Special": {"SPLIT_D1": True, "END_A3": True}
    }
}

DEFAULT_VAL1 = 130
DEFAULT_VAL3 = 250
DEFAULT_VAL4 = 2

class PlotterCsvEnhancer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("プロッタ用 永字八法エンハンサー (選択範囲強調表示)")
        self.geometry("1200x800")

        self.CANVAS_WIDTH = 600
        self.CANVAS_HEIGHT = 600
        self.COORD_LIMIT = 200.0 # -200 ~ 0
        
        self.rows = []
        self.headers = []
        
        # 行インデックス -> CanvasアイテムID のマッピング
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
        
        # ★ 選択イベントのバインド
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
        # 座標説明を更新
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
                    row['SpeakerParams'] = f"{DEFAULT_VAL1} 300 {DEFAULT_VAL3} {DEFAULT_VAL4}" 
                
                if 'SpkCmdType' not in row:
                    row['SpkCmdType'] = 'A2'

            self.lbl_status.config(text=f"{len(self.rows)}行")
            self.draw_trajectory() # 先に描画してIDマップを作る
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
                    
                    speaker_params = row.get('SpeakerParams', f"{DEFAULT_VAL1} 100 {DEFAULT_VAL3} {DEFAULT_VAL4}")
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
            
            start_p = recipe["Start"]
            end_p = recipe["End"]
            
            v_diff = self._interpolate(start_p["V_diff"], end_p["V_diff"], progress)
            z_orig = float(row.get('Z_orig', 0))
            new_z = max(0.0, z_orig + v_diff) 
            
            new_w = self._interpolate(start_p["W"], end_p["W"], progress)
            
            row['Z'] = new_z
            # F値は維持
            row['SpkCmdType'] = recipe["Type"] 
            
            row['SpeakerParams'] = f"{DEFAULT_VAL1} {int(new_w)} {DEFAULT_VAL3} {DEFAULT_VAL4}"
            
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
        
        messagebox.showinfo("完了", f"{stroke_name} を適用しました。")

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
        """リスト選択時にキャンバス上の対応する線を強調表示する"""
        selected_indices = self.listbox.curselection()
        if not selected_indices: return

        # 全ての線を元の色に戻す
        # (効率化のため、前回選択されていたものだけ戻すロジックも可だが、
        #  ここではシンプルにdraw_trajectoryを呼ぶか、タグで管理する)
        
        # 今回はシンプルに、選択されたIDの線だけ色を変える
        # self.line_ids には {row_index: canvas_item_id} が入っている
        
        # まず全リセット (描画処理自体が軽いので再描画でも良いが、色変更のみ行う)
        # 描画時の色を保持していないと戻せないので、draw_trajectoryを再実行してリセットするのが確実
        # ただし頻繁な再描画は重いので、タグを使って制御する
        
        self.canvas.dtag("selected", "selected") # 既存の選択タグを消す
        
        # ハイライト色
        highlight_color = "#00FFFF" # 水色
        
        for idx in self.line_ids:
            # 元の色に戻す処理が必要だが、元の色は動的に決まっているため
            # ここでは「全再描画」し、その直後に「選択部分だけ上書き」する戦略を取る
            # あるいは draw_trajectory 内で選択状態をチェックして描画する
            pass

        # 効率のため、draw_trajectoryに選択インデックスを渡して再描画させる
        self.draw_trajectory(selected_indices)

    def draw_trajectory(self, selected_indices=None):
        self.canvas.delete("all")
        self.line_ids = {} # マッピング初期化
        
        # 座標系: 右上が原点(0,0), 左下が(-200, -200)
        # Canvas: 左上が(0,0), 右下が(W,H)
        # X: -200 -> 0 (0 -> W)
        # Y: -200 -> 0 (H -> 0) ※YはCanvasと逆
        
        scale_x = self.CANVAS_WIDTH / abs(self.COORD_LIMIT)
        scale_y = self.CANVAS_HEIGHT / abs(self.COORD_LIMIT)
        
        # 少し余白を持たせる
        scale_x *= 0.9
        scale_y *= 0.9
        offset_x = self.CANVAS_WIDTH * 0.05
        offset_y = self.CANVAS_HEIGHT * 0.05

        def to_canvas(cx, cy):
            # cx: -200 ~ 0
            # cy: -200 ~ 0
            # x = (cx - (-200)) * scale = (cx + 200) * scale
            x = (cx + 200) * scale_x + offset_x
            # y = 0 -> 0, -200 -> H
            # y = (-cy) * scale
            y = (-cy) * scale_y + offset_y
            
            # 原点(0,0)が右上に来るように反転が必要なら調整
            # 現在のロジック:
            # cx=-200 -> x=0 (左)
            # cx=0    -> x=W (右)  OK
            # cy=-200 -> y=H (下)
            # cy=0    -> y=0 (上)  OK
            
            # 微調整: 右上(0,0) をキャンバスの右上に合わせる
            # 上記式で cx=0, cy=0 は (W, 0) に近い場所になるはず
            
            return x, y

        prev_x, prev_y = None, None
        
        # グリッド線 (オプション)
        self.canvas.create_line(0, offset_y, self.CANVAS_WIDTH, offset_y, fill="#ddd") # Y=0
        self.canvas.create_line(self.CANVAS_WIDTH-offset_x, 0, self.CANVAS_WIDTH-offset_x, self.CANVAS_HEIGHT, fill="#ddd") # X=0

        for i, row in enumerate(self.rows):
            cmd = row.get('Command')
            if cmd not in ['A0', 'A1', 'D1', 'G0', 'G1', 'A3']: continue
            
            raw_x = float(row.get('X', 0))
            raw_y = float(row.get('Y', 0))
            x, y = to_canvas(raw_x, raw_y)
            
            # ハイライト判定
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
                        width += 2 # 太くする
                    
                    # 色決定
                    if is_selected:
                        color = "#00FFFF" # 水色 (Cyan)
                    else:
                        norm_f = min(1.0, max(0.0, (f - 300) / 3700))
                        r = int(255 * norm_f)
                        b = int(255 * (1 - norm_f))
                        color = f"#{r:02x}00{b:02x}"
                    
                    line_id = self.canvas.create_line(prev_x, prev_y, x, y, 
                                        width=width, fill=color, capstyle=tk.ROUND)
                    
                    self.line_ids[i] = line_id # IDを保存
                
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