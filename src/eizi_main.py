import serial
from serial.tools import list_ports
import time
import os
import sys

# --- 設定・定数 ---
BAUDRATE = 115200
TIMEOUT = 0.1

# 永字八法の定義データ
# key: 識別子
# name: 表示名
# cmd_file: プロッタ/スピーカへの命令ファイル(.txt)
# voice_file: 事前に流す音声ガイドファイル(.mp3/.wav)
EIJI_EIGHT_PRINCIPLES = {
    "1": {"key": "soku",  "name": "1. 側 (ソク - 点)",         "cmd_file": "1_soku.txt",  "voice_file": "voice_soku.mp3"},
    "2": {"key": "roku",  "name": "2. 勒 (ロク - 横画)",       "cmd_file": "2_roku.txt",  "voice_file": "voice_roku.mp3"},
    "3": {"key": "do",    "name": "3. 努 (ド - 縦画)",         "cmd_file": "3_do.txt",    "voice_file": "voice_do.mp3"},
    "4": {"key": "teki",  "name": "4. 趯 (テキ - はね)",       "cmd_file": "4_teki.txt",  "voice_file": "voice_teki.mp3"},
    "5": {"key": "saku",  "name": "5. 策 (サク - 右上がり)",   "cmd_file": "5_saku.txt",  "voice_file": "voice_saku.mp3"},
    "6": {"key": "ryaku", "name": "6. 掠 (リャク - 左はらい)", "cmd_file": "6_ryaku.txt", "voice_file": "voice_ryaku.mp3"},
    "7": {"key": "taku1", "name": "7. 啄 (タク - 短い左はらい)","cmd_file": "7_taku1.txt", "voice_file": "voice_taku1.mp3"},
    "8": {"key": "taku2", "name": "8. 磔 (タク - 右はらい)",   "cmd_file": "8_taku2.txt", "voice_file": "voice_taku2.mp3"},
}

# --- デバイス制御クラス群 (前回の改良版) ---

class DeviceController:
    """デバイス制御の基底クラス"""
    def __init__(self, ser: serial.Serial):
        if not ser or not ser.is_open:
            raise ValueError("シリアルポートが開いていません。")
        self.ser = ser

    def send_command(self, command: str):
        line = command + '\n'
        self.ser.write(line.encode('utf-8'))
        # デバッグ用表示（運用時はコメントアウトしても可）
        # print(f"[{self.ser.port}] 送信: {command}")

class Speaker(DeviceController):
    """スピーカアレイ制御"""
    def play_command(self, line: str):
        self.send_command(line)
        time.sleep(0.05) 

class Plotter(DeviceController):
    """プロッター(GRBL)制御"""
    GRID_COUNT = 8
    AREA_SIZE = 200.0
    X_OFFSET = -210.0
    
    def __init__(self, ser: serial.Serial):
        super().__init__(ser)
        self.pen_is_up = True

    def _calculate_coords(self, speaker_num: int, speed_param: int):
        cell_size = self.AREA_SIZE / self.GRID_COUNT
        zero_based = int(speaker_num) - 1
        row = zero_based // self.GRID_COUNT
        col = zero_based % self.GRID_COUNT
        
        x = (col * cell_size + cell_size / 2) + self.X_OFFSET
        y = - (row * cell_size + cell_size / 2)
        
        # 速度計算 (25mm移動基準)
        feed_rate = (25.0 / (int(speed_param) / 1000.0)) * 60
        return x, y, feed_rate

    def move_pen(self, speaker_num, speed_param, pen_down=True):
        x, y, f = self._calculate_coords(speaker_num, speed_param)
        
        if pen_down:
            if self.pen_is_up: self.change_z(down=True)
            cmd = f'G1 X{x:.3f} Y{y:.3f} F{f:.1f}'
        else:
            if not self.pen_is_up: self.change_z(down=False)
            cmd = f'G0 X{x:.3f} Y{y:.3f}'
            
        self.send_command(cmd)

    def change_z(self, down: bool):
        """ペンの上げ下げ"""
        z_cmd = "G0 Z8" if down else "G0 Z0"
        self.send_command(z_cmd)
        self.pen_is_up = not down
        self.wait_for_idle()

    def reset_position(self):
        self.change_z(down=False)
        self.send_command("G0 X0 Y0")
        self.wait_for_idle()

    def wait_for_idle(self):
        """動作完了待ち"""
        while True:
            self.send_command('?')
            try:
                res = self.ser.readline().decode('utf-8').strip()
                if 'Idle' in res:
                    break
            except Exception:
                pass
            time.sleep(0.1)

# --- 永字八法システム本体 ---

class EijihappoSystem:
    def __init__(self, plotter: Plotter, speaker: Speaker):
        self.plotter = plotter
        self.speaker = speaker

    def play_voice_guide(self, voice_file: str):
        """
        音声ガイドを再生する関数
        注意: 環境に合わせて実装を変更してください。
        """
        print(f"\n♪ 音声ガイド再生中: {voice_file}")
        
        if os.path.exists(voice_file):
            # --- 実装例: pygameを使用する場合 (推奨) ---
            # import pygame
            # pygame.mixer.init()
            # pygame.mixer.music.load(voice_file)
            # pygame.mixer.music.play()
            # while pygame.mixer.music.get_busy():
            #     time.sleep(0.1)
            
            # --- 簡易実装: 音声ファイルの長さ分待機するシミュレーション ---
            time.sleep(3) # 仮に3秒待機
            print("♪ 再生完了")
        else:
            print(f"警告: 音声ファイル '{voice_file}' が見つかりません。スキップします。")
            time.sleep(1)

    def execute_principle(self, principle_id: str):
        """指定されたIDの法を実行する"""
        data = EIJI_EIGHT_PRINCIPLES.get(principle_id)
        if not data:
            print("エラー: 未定義のIDです")
            return

        print(f"\n=== {data['name']} を開始します ===")
        
        # 1. 音声ガイド再生 (完了まで待機)
        self.play_voice_guide(data['voice_file'])
        
        # 2. 動作実行
        cmd_file = data['cmd_file']
        if not os.path.exists(cmd_file):
            print(f"エラー: 命令ファイル '{cmd_file}' が見つかりません。")
            return

        print(f"動作開始: {cmd_file}")
        self.run_command_file(cmd_file)
        
        # 3. 終了処理
        self.plotter.reset_position()
        print(f"=== {data['name']} 終了 ===\n")

    def run_command_file(self, file_path):
        """命令ファイルの解析と実行"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                
                parts = line.split(' ')
                cmd = parts[0]
                params = parts[1:]

                # コマンド分岐
                if cmd in ["A1", "A2"]: # 描画移動
                    self.plotter.move_pen(params[0], params[1], pen_down=True)
                    self.speaker.play_command(line) # スピーカへもコマンド送信
                    self.plotter.wait_for_idle()
                
                elif cmd == "A0": # 空移動
                    self.plotter.move_pen(params[0], params[1], pen_down=False)
                    self.plotter.wait_for_idle()
                
                elif cmd == "D1": # ディレイ
                    time.sleep(int(params[0]) / 1000.0 + 0.1)

    def run_individual_mode(self):
        """個別選択モード"""
        while True:
            print("\n--- 個別モード: 実践する法を選んでください ---")
            for pid, data in EIJI_EIGHT_PRINCIPLES.items():
                print(f"  {pid}: {data['name']}")
            print("  0: 戻る")

            choice = input("番号を入力 >> ")
            if choice == "0": break
            
            if choice in EIJI_EIGHT_PRINCIPLES:
                self.execute_principle(choice)
            else:
                print("無効な番号です")

    def run_continuous_mode(self):
        """連続実行モード"""
        print("\n--- 連続モード: 全ての法を順番に実行します ---")
        input("Enterキーを押すと開始します...")
        
        for pid in sorted(EIJI_EIGHT_PRINCIPLES.keys()):
            self.execute_principle(pid)
            
            print("次の法へ進みますか？ (Enter: 次へ / n: 中断)")
            if input() == 'n':
                break
        print("連続モード終了")

# --- ユーティリティ ---
def select_port(prompt_text):
    print(f"\n{prompt_text}")
    ports = list_ports.comports()
    for i, p in enumerate(ports):
        print(f"  {i}: {p.device} ({p.description})")
    
    if not ports: return None
    try:
        idx = int(input("番号 >> "))
        return serial.Serial(ports[idx].device, BAUDRATE, timeout=TIMEOUT)
    except:
        return None

def main():
    print("=== 永字八法 学習システム ===")
    
    # 1. デバイス接続
    sp_ser = select_port("スピーカアレイのポートを選択:")
    pl_ser = select_port("プロッタのポートを選択:")
    
    if not sp_ser or not pl_ser:
        print("ポート接続エラー。終了します。")
        return

    # インスタンス生成
    speaker = Speaker(sp_ser)
    plotter = Plotter(pl_ser)
    system = EijihappoSystem(plotter, speaker)

    # GRBL初期化待ち
    time.sleep(2)
    plotter.reset_position()

    # 2. メインメニュー
    try:
        while True:
            print("\n--- メインメニュー ---")
            print("1: 個別モード (一つずつ選択)")
            print("2: 連続モード (一通り通しで行う)")
            print("q: 終了")
            
            mode = input("選択 >> ")
            
            if mode == '1':
                system.run_individual_mode()
            elif mode == '2':
                system.run_continuous_mode()
            elif mode == 'q':
                break
            
    except KeyboardInterrupt:
        print("\n中断されました")
    finally:
        plotter.reset_position()
        sp_ser.close()
        pl_ser.close()
        print("終了")

if __name__ == "__main__":
    main()