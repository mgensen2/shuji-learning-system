import serial
from serial.tools import list_ports
import time
import os
import sys
import shutil
import subprocess
import re

# --- 設定: 永字八法の定義 ---
# cmd_file: プロッタ/スピーカへの命令ファイル
# voice_file: 事前に流す解説音声ファイル
EIJI_PRINCIPLES = {
    1: {"name": "側 (ソク - 点)",         "cmd_file": "1_soku.txt",  "voice_file": "voice_soku.mp3"},
    2: {"name": "勒 (ロク - 横画)",       "cmd_file": "2_roku.txt",  "voice_file": "voice_roku.mp3"},
    3: {"name": "努 (ド - 縦画)",         "cmd_file": "3_do.txt",    "voice_file": "voice_do.mp3"},
    4: {"name": "趯 (テキ - はね)",       "cmd_file": "4_teki.txt",  "voice_file": "voice_teki.mp3"},
    5: {"name": "策 (サク - 右上がり)",   "cmd_file": "5_saku.txt",  "voice_file": "voice_saku.mp3"},
    6: {"name": "掠 (リャク - 左はらい)", "cmd_file": "6_ryaku.txt", "voice_file": "voice_ryaku.mp3"},
    7: {"name": "啄 (タク - 短い左はらい)","cmd_file": "7_taku1.txt", "voice_file": "voice_taku1.mp3"},
    8: {"name": "磔 (タク - 右はらい)",   "cmd_file": "8_taku2.txt", "voice_file": "voice_taku2.mp3"},
    # テスト用に追加
    9: {"name": "テスト (cho.txt)",       "cmd_file": "cho.txt",     "voice_file": "voice_test.mp3"},
}

# --- ユーティリティ ---
def _safe_serial_write(ser, text):
    """安全なシリアル送信"""
    if ser is None or text is None:
        return
    try:
        s = str(text).strip()
        if not s: return
        # Gコードなどは末尾に改行が必要
        s += '\n'
        ser.write(s.encode('utf-8'))
        try:
            ser.flush()
        except Exception:
            pass
    except Exception as e:
        print(f"_safe_serial_write error: {e}")

def _play_sound_file(path, wait=False):
    """音声再生 (省略: 前回のコードと同様)"""
    if not os.path.exists(path):
        if wait: time.sleep(1)
        return
    try:
        # Mac/Linux/Windows対応 (簡易版)
        if sys.platform == 'darwin':
            subprocess.run(['afplay', path]) if wait else subprocess.Popen(['afplay', path])
        elif sys.platform.startswith('win'):
            cmd = f"(New-Object Media.SoundPlayer '{path}').PlaySync()" if wait else f"(New-Object Media.SoundPlayer '{path}').Play()"
            subprocess.run(['powershell', '-c', cmd])
        else:
            cmd = ['aplay', path]
            subprocess.run(cmd) if wait else subprocess.Popen(cmd)
    except Exception:
        pass

def select_port(prompt="ポート選択"):
    """ポート選択ヘルパー"""
    print(f"\n--- {prompt} ---")
    ports = list_ports.comports()
    if not ports: return None
    for i, p in enumerate(ports):
        print(f"  {i}: {p.device} - {p.description}")
    while True:
        try:
            val = input("番号を入力 (qで中断) >> ")
            if val.lower() == 'q': return None
            num = int(val)
            if 0 <= num < len(ports):
                return serial.Serial(ports[num].device, 115200, timeout=0.1)
        except: pass

# --- デバイスクラス ---
class Speaker:
    def __init__(self, ser):
        self.ser = ser

    def write(self, line):
        """スピーカコマンドを送信"""
        text = str(line).strip()
        if not text: return
        _safe_serial_write(self.ser, text)
        print(f"[Speaker] 送信: {text}")

class Plotter:
    def __init__(self, ser):
        self.ser = ser

    def write_raw(self, gcode):
        """生のGコードを送信"""
        text = str(gcode).strip()
        if not text: return
        _safe_serial_write(self.ser, text)
        print(f"[Plotter] 送信: {text}")

    def reset(self):
        self.write_raw("G0 X0 Y0 Z0")

    def sync(self, timeout=10.0):
        """GRBLのIdle状態を待機"""
        if not self.ser: return
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                self.ser.write(b'?\n')
                time.sleep(0.1)
                while self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if ('Idle' in line) or line.startswith('ok'):
                        return
        except Exception:
            pass

# --- 実行ロジック (ここを修正) ---

def parse_and_execute_line(line, pl, sp):
    """
    cho.txtのような複合形式の行を解析して実行する
    形式例:
      1. G0 X-156.07 ...           -> プロッタのみ
      2. S A2 18 ...               -> スピーカのみ (先頭S除去)
      3. G1 ... F1592  A2 ...      -> プロッタ + スピーカ (タブ/スペース区切り)
    """
    line = line.strip()
    # コメント行や空行はスキップ
    if not line or line.startswith('#') or line.startswith('['):
        return

    gcode_part = None
    speaker_part = None

    # パターン判定
    if line.startswith('S '):
        # スピーカ単独コマンド (例: "S A2 ...")
        # 先頭の "S " を取り除く
        speaker_part = line[2:].strip()
    
    elif line.startswith('G'):
        # Gコードを含む行。スピーカコマンドが後ろについているか確認
        # "A2" や "A1" が出現する位置で分割を試みる
        # タブ区切りの場合が多いが、スペース区切りの可能性も考慮して正規表現で分割
        
        # 正規表現: Gコード部分と、Aから始まるスピーカ部分を分離
        # 例: "G1 ... F1000   A2 ..." -> group1="G1...F1000", group2="A2 ..."
        match = re.search(r'(G[0-9].*?)\s+(A[0-9].*)', line)
        
        if match:
            gcode_part = match.group(1).strip()
            speaker_part = match.group(2).strip()
        else:
            # スピーカコマンドがない純粋なGコード
            gcode_part = line

    # --- 実行 ---
    
    # 1. プロッタ動作開始 (Gコードがある場合)
    if gcode_part:
        pl.write_raw(gcode_part)
    
    # 2. スピーカ命令送信 (ある場合)
    # プロッタが動き出した直後、あるいは同時に音を出す
    if speaker_part:
        sp.write(speaker_part)
    
    # 3. 同期 (プロッタが動いた場合のみ待機)
    if gcode_part:
        pl.sync()
    else:
        # スピーカ単独の場合は少し待つ (コマンドによるが、ここでは短時間)
        time.sleep(0.05)


def execute_file(filename, pl, sp):
    """指定されたファイルを読み込んで実行する"""
    if not os.path.exists(filename):
        print(f"エラー: ファイル '{filename}' が存在しません。")
        return

    print(f"動作開始: {filename}")
    
    # 初期化
    pl.reset()
    pl.sync()

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for raw in f:
                parse_and_execute_line(raw, pl, sp)
                
    except KeyboardInterrupt:
        print(" 中断されました。")
    except Exception as e:
        print(f" エラー: {e}")
    
    # 終了処理
    pl.write_raw("G0 Z0") # ペン上げ
    pl.reset()
    pl.sync()
    print("完了\n")


def run_principle(p_id, pl, sp):
    """ひとつの法を実行する"""
    info = EIJI_PRINCIPLES.get(p_id)
    if not info: return

    print(f"\n=== {info['name']} ===")
    
    # 解説音声 (あれば)
    if info.get('voice_file'):
        print("解説音声を再生します...")
        _play_sound_file(info['voice_file'], wait=True)
    
    # 動作実行
    print("動作を開始します...")
    execute_file(info['cmd_file'], pl, sp)

# --- メイン ---
def main():
    print("=== 永字八法 学習システム (Raw G-code対応版) ===")
    
    sp_ser = select_port("スピーカアレイ設定")
    if not sp_ser: return
    sp = Speaker(sp_ser)

    pl_ser = select_port("プロッタ設定")
    if not pl_ser:
        sp_ser.close()
        return
    pl = Plotter(pl_ser)

    time.sleep(2)
    pl.reset()

    try:
        while True:
            print("\nメニュー:")
            # 定義されている法を表示
            for k, v in EIJI_PRINCIPLES.items():
                print(f"  {k}: {v['name']}")
            print("  q: 終了")
            
            choice = input("選択 >> ").strip().lower()
            if choice == 'q': break
            
            try:
                pid = int(choice)
                if pid in EIJI_PRINCIPLES:
                    run_principle(pid, pl, sp)
            except ValueError:
                pass

    finally:
        pl.write_raw("G0 Z0")
        pl.reset()
        sp_ser.close()
        pl_ser.close()

if __name__ == "__main__":
    main()