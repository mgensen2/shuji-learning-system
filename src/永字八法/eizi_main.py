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

# --- ★追加: A0動作時の各ディレイ時間設定 (連続モード時のみ有効) ---
DELAY_A0_PRE_MOVE   = 1.0  # ① 移動開始前の待機
DELAY_A0_PRE_SOUND  = 1.0  # ② 移動完了後、音声再生前の待機
DELAY_A0_POST_SOUND = 1.0  # ③ 音声再生後の待機

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

def parse_and_execute_line(line, pl, sp, is_continuous=False, voice_file=None, played_state=None):
    """
    1行を解析して実行する
    is_continuous=True の場合、A0コマンド実行時に「移動→音声」の順序制御を行う
    """
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

    # --- A0判定 ---
    is_a0 = False
    if speaker_part and 'A0' in speaker_part:
        is_a0 = True

    # --- 実行ロジック ---
    if is_continuous and is_a0:
        # === 連続モード かつ A0 の場合の特別ロジック ===
        # 順序: Wait -> Move -> Sync -> Wait -> Sound(Explanation) -> Wait -> Write...
        
        # 1. 移動前のディレイ
        time.sleep(DELAY_A0_PRE_MOVE)
        
        # 2. プロッタ移動 (筆上げ移動: G0...)
        if gcode_part:
            pl.send_stream(gcode_part)
            # ★重要: 移動が完了するまで待つ
            pl.sync()
        
        # 3. 音声再生 (解説ボイス)
        # まだこのファイルの解説を再生していない場合のみ再生
        if voice_file and played_state and not played_state[0]:
            time.sleep(DELAY_A0_PRE_SOUND)
            print(f"解説音声再生: {voice_file}")
            _play_sound_file(voice_file, wait=True)
            played_state[0] = True # 再生済みフラグを立てる
            time.sleep(DELAY_A0_POST_SOUND)

        # 4. スピーカアレイへのコマンド送信
        if speaker_part:
            sp.write(speaker_part)
        
    else:
        # === 通常モード または A0以外 ===
        if gcode_part:
            pl.send_stream(gcode_part)
        
        if speaker_part:
            sp.write(speaker_part)
        
        if not gcode_part and speaker_part:
            time.sleep(0.02)


def execute_file(filename, pl, sp, is_continuous=False, voice_file=None):
    """
    指定されたファイルを読み込んで実行する
    is_continuous=Trueのときは voice_file を受け取り、最初のA0で再生する
    """
    if not os.path.exists(filename):
        print(f"エラー: ファイル '{filename}' が存在しません。")
        return

    print(f"動作開始: {filename}")
    
    # 動作前にIdle確認
    pl.sync()

    # 解説音声を再生したかどうかのフラグ (参照渡しするためにリスト化)
    played_state = [False]

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for raw in f:
                parse_and_execute_line(
                    raw, pl, sp, 
                    is_continuous=is_continuous, 
                    voice_file=voice_file, 
                    played_state=played_state
                )
                
    except KeyboardInterrupt:
        print(" 中断されました。")
    except Exception as e:
        print(f" エラー: {e}")
    
    # 終了処理 (ペンを上げて待機)
    pl.send_stream("G0 Z0") 
    pl.sync() 
    print("完了\n")


def run_principle(p_id, pl, sp, return_home=True, is_continuous=False):
    """
    ひとつの法を実行する
    """
    info = EIJI_PRINCIPLES.get(p_id)
    if not info: return

    print(f"\n=== {info['name']} ===")
    
    # 音声ファイルのパス
    v_file = info.get('voice_file')

    # ★モードによる分岐
    if is_continuous:
        # 連続モード:
        # ここでは再生せず、ファイル実行中の「最初の移動(A0)の後」に再生させる
        # voice_file を execute_file に渡す
        print("動作を開始します (音声は移動後に再生)...")
        execute_file(info['cmd_file'], pl, sp, is_continuous=True, voice_file=v_file)
    else:
        # 個別モード:
        # 先に解説音声を再生してから動く (従来通り)
        if v_file:
            print("解説音声を再生します...")
            _play_sound_file(v_file, wait=True)
        
        print("動作を開始します...")
        execute_file(info['cmd_file'], pl, sp, is_continuous=False, voice_file=None)

    # 終了後の原点復帰
    if return_home:
        print("原点へ復帰します...")
        pl.reset()
        pl.sync()

def run_all_principles(pl, sp):
    """全工程を連続実行する (このモードのみDelayと移動後音声が有効)"""
    print("\n====== 永字八法 連続再生モード ======")
    
    # 1番から8番までを対象とする
    target_ids = sorted([k for k in EIJI_PRINCIPLES.keys() if k != 9])
    
    for pid in target_ids:
        # 連続実行モードで呼び出し
        run_principle(pid, pl, sp, return_home=False, is_continuous=True)
        # 次の画への少しの間
        time.sleep(0.5)
        
    print("\n全工程完了。原点へ復帰します。")
    pl.reset()
    pl.sync()

# --- メイン ---
def main():
    print("=== 永字八法 学習システム (Smart Motion Ver.) ===")
    
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
            for k in sorted(EIJI_PRINCIPLES.keys()):
                print(f"  {k}: {EIJI_PRINCIPLES[k]['name']}")
            print("  a: 全てを連続再生 (1〜8)")
            print("  q: 終了")
            
            choice = input("選択 >> ").strip().lower()
            if choice == 'q':
                break
            elif choice == 'a':
                # 連続再生 (移動 → 音声 モード)
                run_all_principles(pl, sp)
            else:
                try:
                    pid = int(choice)
                    if pid in EIJI_PRINCIPLES:
                        # 個別実行 (音声 → 移動 モード)
                        run_principle(pid, pl, sp, return_home=True, is_continuous=False)
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