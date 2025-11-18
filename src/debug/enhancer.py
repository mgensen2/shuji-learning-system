import tkinter as tk
from tkinter import filedialog, messagebox
import re
import math
import random # 描画のためにランダム値を使用

# --- 1. 永字八法コマンド変換レシピ（補間ターゲット値） --- (省略。前回のコードと同じ)
# ... [EIGHT_STROKES 辞書は前回のコードを参照] ...
EIGHT_STROKES = {
    "側 (点)": {
        "Type": "A2", "V_adj": 2, "Special": {},
        "V": (15, 15), "F": (2000, 2000), "W": (200, 200)
    },
    "勒 (横画)": {
        "Type": "B2", "V_adj": 0, "Special": {"END_A3": True},
        "V": (-5, -5), "F": (500, 500), "W": (300, 300)
    },
    "努 (縦画)": {
        "Type": "B2", "V_adj": 2, "Special": {"END_A3": True},
        "V": (5, 5), "F": (300, 300), "W": (500, 500)
    },
    "趯 (跳ね)": {
        "Type": "B2", "V_adj": 2, "Special": {"END_A3": True},
        "V": (10, 20), "F": (2000, 4000), "W": (1000, 1500)
    },
    "策 (短横画)": {
        "Type": "A2", "V_adj": 0, "Special": {},
        "V": (-10, -10), "F": (1500, 1500), "W": (500, 500)
    },
    "掠 (左はらい)": {
        "Type": "B2", "V_adj": 1, "Special": {"END_A3": True},
        "V": (10, -15), "F": (700, 1500), "W": (1200, 1200)
    },
    "啄 (短いはらい)": {
        "Type": "A2", "V_adj": 2, "Special": {},
        "V": (15, 15), "F": (2500, 2500), "W": (400, 400)
    },
    "磔 (右はらい)": {
        "Type": "B2", "V_adj": 0, "Special": {"END_A3": True, "D1_INSERT_POS": 0.6},
        "PHASE1": {"V": (-10, 5), "F": (600, 600), "W": (300, 300)},
        "PHASE2": {"V": (5, 20), "F": (1500, 1500), "W": (1000, 1000)}
    }
}

# --- 2. メインアプリケーションクラス ---
class EijiHappouEnhancer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("永字八法エンハンサー (GUI版)")
        
        # --- ★修正: 定数/インスタンス変数の定義を冒頭に移動★ ---
        self.CANVAS_WIDTH = 500
        self.CANVAS_HEIGHT = 400
        self.INITIAL_X = 50
        self.INITIAL_Y = 50
        self.MAX_VOLUME = 100 
        self.MIN_VOLUME = 10
        self.MAX_LINE_WIDTH = 8
        self.MIN_LINE_WIDTH = 2
        self.MAX_FREQ = 4000
        self.MIN_FREQ = 300
        # ---------------------------------------------------
        
        self.original_commands = []
        self.enhanced_commands = []
        
        self.create_widgets()
    def create_widgets(self):
        # ... [前回のコードと同じファイル操作・設定エリアのコード] ...
        # (前回の create_widgets メソッドの内容を貼り付け、以下の変更を加える)
        
        # --- 既存コード貼り付け位置 ---

        # メインフレーム
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack()

        # ファイル操作エリア
        file_frame = tk.LabelFrame(main_frame, text="1. ファイル操作", padx=5, pady=5)
        file_frame.pack(fill="x", pady=5)
        tk.Button(file_frame, text="元のノイズファイルを開く", command=self.load_file).pack(side="left", padx=5)
        self.file_status = tk.Label(file_frame, text="ファイル未読み込み")
        self.file_status.pack(side="left", padx=10)
        
        # Cコマンド設定エリア
        c_frame = tk.LabelFrame(main_frame, text="2. Cコマンドデフォルト設定", padx=5, pady=5)
        c_frame.pack(fill="x", pady=5)
        tk.Label(c_frame, text="F (Hz):").pack(side="left"); self.c_f = tk.Entry(c_frame, width=8); self.c_f.insert(0, "1000"); self.c_f.pack(side="left", padx=5)
        tk.Label(c_frame, text="W (Hz):").pack(side="left"); self.c_w = tk.Entry(c_frame, width=8); self.c_w.insert(0, "500"); self.c_w.pack(side="left", padx=5)
        tk.Label(c_frame, text="V_adj (0/1/2):").pack(side="left"); self.c_v_adj = tk.Entry(c_frame, width=5); self.c_v_adj.insert(0, "0"); self.c_v_adj.pack(side="left", padx=5)

        # --- 新しいキャンバスエリア ---
        canvas_frame = tk.LabelFrame(main_frame, text="運筆軌跡表示 (太さ:ボリューム, 色:周波数)", padx=5, pady=5)
        canvas_frame.pack(fill="x", pady=5)
        self.canvas = tk.Canvas(canvas_frame, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg="white", borderwidth=2, relief="sunken")
        self.canvas.pack()
        # -----------------------------
        
        # 運筆変換エリア
        convert_frame = tk.LabelFrame(main_frame, text="3. 運筆変換設定", padx=5, pady=5)
        convert_frame.pack(fill="x", pady=5)

        tk.Label(convert_frame, text="始点 (行No.):").pack(side="left"); self.start_line = tk.Entry(convert_frame, width=5); self.start_line.pack(side="left", padx=5)
        tk.Label(convert_frame, text="終点 (行No.):").pack(side="left"); self.end_line = tk.Entry(convert_frame, width=5); self.end_line.pack(side="left", padx=5)

        tk.Label(convert_frame, text="技法:").pack(side="left")
        self.stroke_var = tk.StringVar(self)
        self.stroke_var.set("勒 (横画)")
        stroke_menu = tk.OptionMenu(convert_frame, self.stroke_var, *EIGHT_STROKES.keys())
        stroke_menu.pack(side="left", padx=5)

        tk.Button(convert_frame, text="⚡ 変換実行", command=self.apply_enhancement, bg="#4CAF50", fg="white").pack(side="left", padx=10)

        # ログ/出力エリア
        output_frame = tk.LabelFrame(main_frame, text="4. 変換コマンドログ", padx=5, pady=5)
        output_frame.pack(fill="both", expand=True, pady=5)
        
        self.log_text = tk.Text(output_frame, height=10, width=60) # 高さを調整
        self.log_text.pack(side="left", fill="both", expand=True)
        
        tk.Button(output_frame, text="✅ ファイル保存", command=self.save_file).pack(side="bottom", fill="x", pady=5)


    # --- 3. ファイル操作メソッド --- (変更なし)
    def load_file(self):
        # ... [既存の load_file メソッドの内容] ...
        filepath = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filepath:
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            self.original_commands = [line.strip() for line in f if line.strip()]
        
        self.enhanced_commands = list(self.original_commands)
        self.file_status.config(text=f"ファイル読み込み完了 ({len(self.original_commands)}行)")
        self.display_commands(self.enhanced_commands)
        
        # ★ 軌跡の初期描画を追加 ★
        self.draw_trajectory(self.enhanced_commands)
        messagebox.showinfo("情報", "ファイルを読み込みました。\n行番号を確認し、変換範囲を指定してください。")
        
    def save_file(self):
        # ... [既存の save_file メソッドの内容] ...
        if not self.enhanced_commands:
            messagebox.showerror("エラー", "変換対象のコマンドがありません。")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if not filepath:
            return
        
        c_command = f"C {self.c_f.get()} {self.c_w.get()} {self.c_v_adj.get()}"
        output_data = [c_command] + self.enhanced_commands

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_data))
        
        messagebox.showinfo("成功", f"コマンドをファイルに保存しました:\n{filepath}")

    def display_commands(self, commands):
        # ... [既存の display_commands メソッドの内容] ...
        self.log_text.delete(1.0, tk.END)
        for i, cmd in enumerate(commands):
            self.log_text.insert(tk.END, f"{i+1:03d}: {cmd}\n")
            
    # --- 4. コマンド変換ロジック（コア） --- (変更なし、適用後に描画を追加)
    def apply_enhancement(self):
        # ... [既存の apply_enhancement メソッドの内容] ...
        try:
            # ... [前回のロジック：入力チェック、ターゲットコマンド抽出、補間ロジック実行、コマンド置き換え] ...
            
            # (前回の apply_enhancement の処理をコピーし、以下を追加)
            
            # --- ここに前回の apply_enhancement の内容が入る ---
            
            # 入力チェック
            if not self.enhanced_commands:
                raise ValueError("ファイルを読み込んでください。")
            
            start_idx = int(self.start_line.get()) - 1
            end_idx = int(self.end_line.get()) - 1
            stroke_name = self.stroke_var.get()

            if not (0 <= start_idx <= end_idx < len(self.enhanced_commands)):
                raise ValueError("無効な行番号の範囲です。")

            target_cmds = self.enhanced_commands[start_idx : end_idx + 1]
            stroke_data = EIGHT_STROKES[stroke_name]
            
            total_duration = 0
            
            parsed_params = []
            for cmd in target_cmds:
                match = re.match(r'(A1|B1)\s+(\d+)\s+(\d+)\s+(\d+)', cmd)
                # A2/B2の形式でフィルタ設定が省略されていないコマンドを処理
                match_a2b2 = re.match(r'(A2|B2)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', cmd)
                
                if match:
                    cmd_type, speaker, duration, volume = match.groups()
                    duration, volume = int(duration), int(volume)
                    total_duration += duration
                    parsed_params.append({'S': speaker, 'T': duration, 'V': volume})
                elif match_a2b2:
                    # 既に変換済みのコマンドを再変換しようとした場合、エラーとするかスキップするか選択
                    raise ValueError(f"範囲内に既にA2/B2形式のコマンドが含まれています。A1/B1のみを選択してください。: {cmd}")
                else:
                    raise ValueError(f"範囲内にA1/B1形式ではないコマンドが含まれています: {cmd}")

            if total_duration == 0:
                raise ValueError("選択範囲の合計再生時間が0msです。")

            # 新しいコマンドシーケンスの生成
            new_commands = self._generate_enhanced_commands(stroke_name, stroke_data, parsed_params, total_duration)

            # コマンドの置き換え
            self.enhanced_commands[start_idx : end_idx + 1] = new_commands

            messagebox.showinfo("成功", f"'{stroke_name}' の運筆を適用しました。")
            self.display_commands(self.enhanced_commands)
            
            # ★ 軌跡の再描画を追加 ★
            self.draw_trajectory(self.enhanced_commands)

        except Exception as e:
            messagebox.showerror("エラー", str(e))
            
    # ... [既存の _interpolate, _generate_enhanced_commands メソッドの内容] ...
    def _interpolate(self, start_val, end_val, current_pos):
        # ... [既存の _interpolate の内容] ...
        return round(start_val + (end_val - start_val) * current_pos)

    def _generate_enhanced_commands(self, stroke_name, stroke_data, params, total_duration):
        # ... [既存の _generate_enhanced_commands の内容] ...
        # (前回のコードからコピーして使用してください)
        
        new_cmds = []
        elapsed_time = 0
        is_split_stroke = (stroke_name == "磔 (右はらい)")
        
        if is_split_stroke:
            D1_pos = stroke_data['Special']['D1_INSERT_POS']
            phase1_duration = total_duration * D1_pos
            D1_inserted = False
            
            targets = stroke_data['PHASE1']
            V_start, V_end = targets['V']
            F_start, F_end = targets['F']
            W_start, W_end = targets['W']
        else:
            V_start, V_end = stroke_data['V']
            F_start, F_end = stroke_data['F']
            W_start, W_end = stroke_data['W']

        V_adj = stroke_data['V_adj']
        cmd_type = stroke_data['Type']

        for i, p in enumerate(params):
            duration = p['T']
            elapsed_time += duration
            
            if is_split_stroke and elapsed_time > phase1_duration and not D1_inserted:
                
                new_cmds.append("D1 300")
                D1_inserted = True

                targets = stroke_data['PHASE2']
                V_start, V_end = targets['V']
                F_start, F_end = targets['F']
                W_start, W_end = targets['W']

                phase2_elapsed_time = elapsed_time - (total_duration * D1_pos)
                phase2_total_duration = total_duration * (1 - D1_pos)
                
                current_time_pos = phase2_elapsed_time / phase2_total_duration
            
            elif is_split_stroke and D1_inserted:
                current_time_pos = (elapsed_time - (total_duration * D1_pos)) / (total_duration * (1 - D1_pos))
            else:
                current_time_pos = elapsed_time / total_duration

            v_interp = self._interpolate(p['V'] + V_start, p['V'] + V_end, current_time_pos)
            f_interp = self._interpolate(F_start, F_end, current_time_pos)
            w_interp = self._interpolate(W_start, W_end, current_time_pos)
            
            new_cmd = (
                f"{cmd_type} {p['S']} {duration} {v_interp} "
                f"{f_interp} {w_interp} {V_adj}"
            )
            new_cmds.append(new_cmd)

        if stroke_data['Special'].get("END_A3"):
            new_cmds.append(f"A3 {params[0]['S']}")
            
        return new_cmds

    # --- 5. 軌跡描画メソッド (新規追加) ---
    
    def get_color_from_freq(self, freq):
        """周波数に基づいて色のRBG値を計算 (低F=青, 高F=赤)"""
        # 正規化: [MIN_FREQ, MAX_FREQ] -> [0, 1]
        norm_freq = (freq - self.MIN_FREQ) / (self.MAX_FREQ - self.MIN_FREQ)
        norm_freq = max(0, min(1, norm_freq)) # 0から1の範囲にクリップ

        # 青(0)から赤(1)へのグラデーション (H/S/VのVを固定した色相変化を簡略化)
        # 低い周波数ほど青 (RGB: 00FFC0 -> 0000FF), 高い周波数ほど赤 (RGB: FF0000)
        
        # 簡易的な青-緑-赤のグラデーション:
        r = int(255 * norm_freq)       # Fが高くなるほどRが増加
        g = int(255 * (1 - abs(norm_freq - 0.5) * 2)) # 中間(緑)でピーク
        b = int(255 * (1 - norm_freq)) # Fが低くなるほどBが増加
        
        return f'#{r:02x}{g:02x}{b:02x}'


    def draw_trajectory(self, commands):
        self.canvas.delete("all")
        
        # 描画の基準座標
        x, y = self.INITIAL_X, self.INITIAL_Y
        prev_x, prev_y = x, y
        
        # 描画の単位
        # 100ms 再生を 10ピクセル移動と仮定 (描画スピードを時間で制御)
        PIXELS_PER_100MS = 10 
        
        # 軌跡の方向を管理する（今回は簡易的に右下方向へ進める）
        angle_rad = math.radians(45) # 45度の右下方向

        # コマンドを解析し描画
        for cmd in commands:
            # B2/A2コマンド（フィルタ設定付き）を解析
            match_b2a2 = re.match(r'(A2|B2)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', cmd)
            # D1コマンドを解析
            match_d1 = re.match(r'D1\s+(\d+)', cmd)
            
            if match_b2a2:
                # B2/A2 [S] [T] [V] [F] [W] [V_adj]
                T, V, F = map(int, match_b2a2.groups()[2:5])
                
                # 描画太さ (ボリューム V に比例)
                V_clamped = max(self.MIN_VOLUME, min(self.MAX_VOLUME, V))
                width = self.MIN_LINE_WIDTH + (self.MAX_LINE_WIDTH - self.MIN_LINE_WIDTH) * \
                        ((V_clamped - self.MIN_VOLUME) / (self.MAX_VOLUME - self.MIN_VOLUME))
                
                # 描画色 (中心周波数 F に比例)
                F_clamped = max(self.MIN_FREQ, min(self.MAX_FREQ, F))
                color = self.get_color_from_freq(F_clamped)

                # 座標の移動 (時間 T に比例)
                distance = T / 100 * PIXELS_PER_100MS
                
                # 簡易的な座標移動。本来は技法に応じて移動方向が変わる
                # 例として、単純に右下に進めるか、ランダムな方向に進める
                if '勒' in cmd or '策' in cmd: # 横画
                     delta_x = distance
                     delta_y = 0
                elif '努' in cmd: # 縦画
                     delta_x = 0
                     delta_y = distance
                elif '掠' in cmd: # 左下はらい
                     delta_x = -distance * math.cos(math.radians(30))
                     delta_y = distance * math.sin(math.radians(30))
                else: # その他 (側、趯、磔など) や初期コマンド
                     delta_x = distance * math.cos(angle_rad)
                     delta_y = distance * math.sin(angle_rad)
                
                prev_x, prev_y = x, y
                x += delta_x
                y += delta_y

                # 軌跡の描画 (線)
                self.canvas.create_line(prev_x, prev_y, x, y, 
                                        width=width, fill=color, capstyle=tk.ROUND)
                
                # 点の強調 (勒、努、策以外の収筆)
                if '側' in cmd or '啄' in cmd or '趯' in cmd or '磔' in cmd:
                     self.canvas.create_oval(x-width/2, y-width/2, x+width/2, y+width/2, 
                                             fill=color, outline=color)
                     
            elif match_d1:
                # D1 (待機) の場合、座標は動かさないが、点で強調
                prev_x, prev_y = x, y
                self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black", outline="black")
                
            elif 'A3' in cmd:
                # A3 (停止) の場合、収筆として点で強調
                prev_x, prev_y = x, y
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", outline="red")
            
            else:
                 # 未変換のA1/B1やその他のコマンドは細い線で表示
                 distance = 10 # 仮想的な移動距離
                 prev_x, prev_y = x, y
                 x += distance * math.cos(angle_rad)
                 y += distance * math.sin(angle_rad)
                 self.canvas.create_line(prev_x, prev_y, x, y, width=1, fill="gray")
                 
        # 描画の最終位置に小さな円を描画
        self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black")

if __name__ == "__main__":
    # --- 前回のコードの _interpolate と _generate_enhanced_commands をコピーして使用してください ---
    # (ここでは簡略化のため、既存のメソッド定義は省略しています)
    
    app = EijiHappouEnhancer()
    app.mainloop()