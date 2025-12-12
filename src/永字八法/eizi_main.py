import serial
from serial.tools import list_ports
import time
import os
import sys
import subprocess
import re

# --- 設定: 永字八法の定義 ---
EIJI_PRINCIPLES = {
    1: {"name": "側 (ソク - 点)",         "cmd_file": "1_soku.txt",  "voice_file": "soku.wav"},
    2: {"name": "勒 (ロク - 横画)",       "cmd_file": "2_roku.txt",  "voice_file": "roku.wav"},
    3: {"name": "努 (ド - 縦画)",         "cmd_file": "3_do.txt",    "voice_file": "do.wav"},
    4: {"name": "趯 (テキ - はね)",       "cmd_file": "4_teki.txt",  "voice_file": "teki.wav"},
    5: {"name": "策 (サク - 右上がり)",   "cmd_file": "5_saku.txt",  "voice_file": "saku.wav"},
    6: {"name": "掠 (リャク - 左はらい)", "cmd_file": "6_ryaku.txt", "voice_file": "ryaku.wav"},
    7: {"name": "啄 (タク - 短い左はらい)","cmd_file": "7_taku1.txt", "voice_file": "taku1.wav"},
    8: {"name": "磔 (タク - 右はらい)",   "cmd_file": "8_taku2.txt", "voice_file": "taku2.wav"},
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

    def send_stream(self, command):
        """コマンドを送信し、'ok' が返ってくるまで待機する（動きの完了は待たない）"""
        if not self.ser:
            return
        
        line = str(command).strip()
        _safe_serial_write(self.ser, line)

        while True:
            try:
                resp = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if resp == 'ok':
                    break 
                if resp.lower().startswith('error'):
                    print(f"[Plotter] Error受信: {resp}")
                    break
            except Exception:
                pass

    def reset(self):
        """原点復帰 (Z0にしてからX0Y0へ)"""
        self.send_stream("G0 Z0") # ペンを上げる
        self.send_stream("G0 X0 Y0")

    def sync(self, timeout=10.0):
        """GRBLの動きが完全に止まる(Idle状態)まで待機"""
        if not self.ser: return
        deadline = time.time() + timeout
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
                        if 'Idle' in line:
                            return
                    except Exception:
                        pass
                time.sleep(0.05)
        except Exception as e:
            print(f"Sync error: {e}")

# --- 実行ロジック ---

def parse_and_execute_line(line, pl, sp):
    """1行を解析して実行する"""
    line = line.strip()
    if not line or line.startswith('#') or line.startswith('['):
        return

    gcode_part = None
    speaker_part = None

    if line.startswith('S '):
        speaker_part = line[2:].strip()
    elif line.startswith('G'):
        match = re.search(r'(G[0-9].*?)\s+(A[0-9].*)', line)
        if match:
            gcode_part = match.group(1).strip()
            speaker_part = match.group(2).strip()
        else:
            gcode_part = line

    if gcode_part:
        pl.send_stream(gcode_part)
    
    if speaker_part:
        sp.write(speaker_part)
    
    if not gcode_part and speaker_part:
        time.sleep(0.02)


def execute_file(filename, pl, sp):
    """指定されたファイルを読み込んで実行する"""
    if not os.path.exists(filename):
        print(f"エラー: ファイル '{filename}' が存在しません。")
        return

    print(f"動作開始: {filename}")
    
    # 動作前にIdle確認
    pl.sync()

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for raw in f:
                parse_and_execute_line(raw, pl, sp)
                
    except KeyboardInterrupt:
        print(" 中断されました。")
    except Exception as e:
        print(f" エラー: {e}")
    
    # 終了処理 (ペンを上げて待機)
    pl.send_stream("G0 Z0") 
    pl.sync() 
    print("完了\n")


def run_principle(p_id, pl, sp, return_home=True):
    """
    ひとつの法を実行する
    return_home: Trueなら実行後に原点(X0Y0)に戻る
    """
    info = EIJI_PRINCIPLES.get(p_id)
    if not info: return

    print(f"\n=== {info['name']} ===")
    
    # 解説音声
    if info.get('voice_file'):
        print("解説音声を再生します...")
        _play_sound_file(info['voice_file'], wait=True)
    
    # 動作実行
    print("動作を開始します...")
    execute_file(info['cmd_file'], pl, sp)

    # ★変更点: フラグがTrueなら原点に戻る
    if return_home:
        print("原点へ復帰します...")
        pl.reset()
        pl.sync()

def run_all_principles(pl, sp):
    """★追加: 全工程を連続実行する"""
    print("\n====== 永字八法 連続再生モード ======")
    
    # 1番から8番までを対象とする（9番のテストは除外）
    target_ids = sorted([k for k in EIJI_PRINCIPLES.keys() if k != 9])
    
    for pid in target_ids:
        # 連続実行中は、筆の流れを維持するため各画ごとの原点復帰は行わない
        run_principle(pid, pl, sp, return_home=False)
        time.sleep(0.5) # 次の画への少しの間
        
    print("\n全工程完了。原点へ復帰します。")
    pl.reset()
    pl.sync()

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
    pl.sync()

    try:
        while True:
            print("\nメニュー:")
            # 辞書順に表示
            for k in sorted(EIJI_PRINCIPLES.keys()):
                print(f"  {k}: {EIJI_PRINCIPLES[k]['name']}")
            print("  a: 全てを連続再生 (1〜8)")
            print("  q: 終了")
            
            choice = input("選択 >> ").strip().lower()
            if choice == 'q':
                break
            elif choice == 'a':
                # ★追加: 連続再生
                run_all_principles(pl, sp)
            else:
                try:
                    pid = int(choice)
                    if pid in EIJI_PRINCIPLES:
                        # ★変更: 個別実行時は必ず原点に戻る
                        run_principle(pid, pl, sp, return_home=True)
                except ValueError:
                    pass

    finally:
        print("システム終了処理中...")
        pl.reset()
        pl.sync()
        sp_ser.close()
        pl_ser.close()

if __name__ == "__main__":
    main()