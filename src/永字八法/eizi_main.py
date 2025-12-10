import serial
from serial.tools import list_ports
import time
import os
import sys
import shutil
import subprocess
import re

# --- 設定: 永字八法の定義 ---
EIJI_PRINCIPLES = {
    1: {"name": "側 (ソク - 点)",         "cmd_file": "1_soku.txt",  "voice_file": "soku.wav"},
    2: {"name": "勒 (ロク - 横画)",       "cmd_file": "2_roku.txt",  "voice_file": "roku.wav"},
    3: {"name": "努 (ド - 縦画)",         "cmd_file": "3_do.txt",    "voice_file": "do.wav"},
    4: {"name": "趯 (テキ - はね)",       "cmd_file": "4_teki.txt",  "voice_file": "eki.wav"},
    5: {"name": "策 (サク - 右上がり)",   "cmd_file": "5_saku.txt",  "voice_file": "saku.wav"},
    6: {"name": "掠 (リャク - 左はらい)", "cmd_file": "6_ryaku.txt", "voice_file": "ryaku.wav"},
    7: {"name": "啄 (タク - 短い左はらい)","cmd_file": "7_taku1.txt", "voice_file": "aku1.wav"},
    8: {"name": "磔 (タク - 右はらい)",   "cmd_file": "8_taku2.txt", "voice_file": "aku2.wav"},
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
    """音声再生"""
    if not os.path.exists(path):
        if wait: time.sleep(1)
        return
    try:
        # Mac/Linux/Windows対応 (簡易版)
        if sys.platform == 'darwin':
            subprocess.run(['afplay', path]) if wait else subprocess.Popen(['afplay', path])
        elif sys.platform.startswith('win'):
            cmd = f"(New-Object Media.SoundPlayer '{path}').PlaySync()" if wait else f"(New-Object Media.SoundPlayer '{path}').Play()"
            subprocess.run(['powershell', '-c', cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        """生のGコードを送信 (応答を待たない / 初期化用)"""
        text = str(gcode).strip()
        if not text: return
        _safe_serial_write(self.ser, text)
        print(f"[Plotter] 送信(Raw): {text}")

    # ★追加: ストリーミング配信用メソッド
    def send_stream(self, command):
        """コマンドを送信し、'ok' が返ってくるまで待機する（動きの完了は待たない）"""
        if not self.ser:
            return
        
        line = str(command).strip()
        _safe_serial_write(self.ser, line)
        # print(f"[Plotter] Stream送信: {line}") # ログが多すぎる場合はコメントアウト

        # GRBLからの 'ok' を待つ
        while True:
            try:
                resp = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if resp == 'ok':
                    break # okが来たら即座に次へ（動きは止まらない）
                if resp.lower().startswith('error'):
                    print(f"[Plotter] Error受信: {resp}")
                    break
            except Exception:
                pass

    def reset(self):
        self.write_raw("G0 X0 Y0 Z0")

    def sync(self, timeout=10.0):
        """
        GRBLの動きが完全に止まる(Idle状態)まで待機
        ★重要: 'ok' は無視して 'Idle' だけを見るように変更
        """
        if not self.ser: return
        deadline = time.time() + timeout
        # print("[Plotter] sync start (Waiting for Idle)")
        try:
            while time.time() < deadline:
                try:
                    self.ser.write(b'?\n')
                except:
                    break
                
                start_read = time.time()
                while time.time() - start_read < 0.2:
                    try:
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        if not line: continue
                        
                        # ★修正: Idle のみを完了とみなす
                        if 'Idle' in line:
                            return
                    except Exception:
                        pass
                time.sleep(0.05)
        except Exception as e:
            print(f"Sync error: {e}")

# --- 実行ロジック ---

def parse_and_execute_line(line, pl, sp):
    """
    1行を解析して実行する
    ★変更: sync() を削除し、send_stream() を使用してカクつきを防止
    """
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('['):
        return

    gcode_part = None
    speaker_part = None

    # パターン判定
    if line.startswith('S '):
        # スピーカ単独コマンド
        speaker_part = line[2:].strip()
    
    elif line.startswith('G'):
        # Gコードを含む行
        match = re.search(r'(G[0-9].*?)\s+(A[0-9].*)', line)
        if match:
            gcode_part = match.group(1).strip()
            speaker_part = match.group(2).strip()
        else:
            gcode_part = line

    # --- 実行 ---
    
    # 1. プロッタ動作 (Gコードがある場合)
    if gcode_part:
        # ★修正: write_raw + sync ではなく、send_stream を使う
        pl.send_stream(gcode_part)
    
    # 2. スピーカ命令送信 (ある場合)
    if speaker_part:
        sp.write(speaker_part)
    
    # ★修正: ここにあった pl.sync() を削除しました
    # これによりプロッタが移動中でも次の行の命令を送り込みます

    if not gcode_part and speaker_part:
        # スピーカ単独の場合は、送ってすぐに次に行くと早すぎる場合があるので少しだけ待つ
        time.sleep(0.02)


def execute_file(filename, pl, sp):
    """指定されたファイルを読み込んで実行する"""
    if not os.path.exists(filename):
        print(f"エラー: ファイル '{filename}' が存在しません。")
        return

    print(f"動作開始: {filename}")
    
    # 初期化 (ここは安全のため sync する)
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
    
    # 終了処理 (最後にペンを上げて、動きが止まるまで待つ)
    pl.send_stream("G0 Z0") # ペン上げも stream で送る
    pl.sync()               # 全ての動作が終わるまで待機
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
    print("=== 永字八法 学習システム (Smooth Motion Ver.) ===")
    
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
        pl.sync() # 終了時も待機
        sp_ser.close()
        pl_ser.close()

if __name__ == "__main__":
    main()