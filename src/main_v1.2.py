import serial
from serial.tools import list_ports
import time
import os
import threading
import sys
import shutil
import subprocess

# --- 定数 / グローバル ---
RECORD_FILE = None
RECORD_EXECUTE = True
_RECORD_LOCK = threading.Lock()

STROKE_COUNT = 0          # A0 を受けるたびにインクリメント

# --- ユーティリティ ----------------------------------------------------------------
def _safe_serial_write(ser, text):
    """ser が有効なら必ず改行を付けて書き込み、可能なら flush。例外は捕捉してログ出力のみ。"""
    if ser is None or text is None:
        return
    try:
        s = str(text)
        if not s.endswith('\n'):
            s += '\n'
        ser.write(s.encode('utf-8'))
        try:
            ser.flush()
        except Exception:
            pass
    except Exception as e:
        print(f"_safe_serial_write error: {e}")

def _play_sound_file(path):
    """mac/windows/linux で非同期に再生を試みる。外部コマンドに依存。"""
    if not os.path.exists(path):
        print(f"音声ファイルが見つかりません: {path}")
        return
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(['afplay', path])
        elif sys.platform.startswith('win'):
            subprocess.Popen(['powershell', '-c', f"(New-Object Media.SoundPlayer '{path}').Play()"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            if shutil.which('paplay'):
                subprocess.Popen(['paplay', path])
            elif shutil.which('aplay'):
                subprocess.Popen(['aplay', path])
            elif shutil.which('ffplay'):
                subprocess.Popen(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', path])
            else:
                print("音声再生コマンドが見つかりません (afplay/aplay/paplay).")
    except Exception as e:
        print(f"音声再生エラー: {e}")

# --- シリアル / UI ヘルパー ----------------------------------------------------------
def select_run_mode():
    """実行モードを選択する"""
    print("実行モードを選択してください:")
    print("  1: 両方 (Speaker & Plotter)")
    print("  2: スピーカのみ (Speaker Only)")
    print("  3: プロッタのみ (Plotter Only)")
    
    while True:
        choice = input("番号を入力してください >> ").strip()
        if choice == '1':
            return 'both'
        elif choice == '2':
            return 'speaker_only'
        elif choice == '3':
            return 'plotter_only'
        else:
            print("1, 2, 3 のいずれかを入力してください。")

def select_port(device_name="デバイス"):
    """利用可能なシリアルポートを列挙して選択・オープンして返す。失敗時は None。"""
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.timeout = 0.1

    print(f"--- {device_name} のポート選択 ---")
    ports = list_ports.comports()
    if not ports:
        print("エラー: 利用可能なシリアルポートが見つかりません。")
        return None

    if len(ports) == 1:
        port = ports[0]
        print(f"ポートが1つ見つかりました: {port.device} ({port.description})")
        ser.port = port.device
    else:
        print("利用可能なポート:")
        for i, p in enumerate(ports):
            print(f"  {i}: {p.device} - {p.description}")
        try:
            num = int(input("接続するポートの番号を入力してください >> "))
            if num < 0 or num >= len(ports):
                print("エラー: 無効な番号です。")
                return None
            ser.port = ports[num].device
        except ValueError:
            print("エラー: 数字で入力してください。")
            return None

    try:
        ser.open()
        print(f"{device_name}: ポート {ser.port} を開きました。\n")
        return ser
    except serial.SerialException as e:
        print(f"エラー: ポートを開けませんでした - {e}")
        return None

def select_file():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    if not files:
        print("ファイルが見つかりません。")
        return None
    print("送信するファイルを選択してください:")
    for i, f in enumerate(files):
        print(f"  {i + 1}: {f}")
    while True:
        try:
            choice = int(input("番号を入力してください: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print("無効な番号です。")
        except ValueError:
            print("数値を入力してください。")

def yes_no_input(prompt="Please respond with 'yes' or 'no' [y/N]: "):
    while True:
        choice = input(prompt).lower().strip()
        if choice == '':
            return False
        if choice in ['y', 'ye', 'yes']:
            return True
        if choice in ['n', 'no']:
            return False
        print("y または n を入力してください。")

# --- 記録関連 -----------------------------------------------------------------------
def select_recording():
    global RECORD_FILE, RECORD_EXECUTE
    while True:
        choice = input("変換したプロッタ命令とスピーカ命令を記録しますか？ (y/N): ").strip().lower()
        if choice in ('y','yes'):
            fname = input("保存するファイル名を入力（空欄で 'recorded_commands.txt'）: ").strip()
            if not fname:
                fname = "recorded_commands.txt"
            try:
                with open(fname, 'a', encoding='utf-8'):
                    pass
                RECORD_FILE = fname
                ex = input("保存時にデバイスも動かしますか？ (y/N): ").strip().lower()
                RECORD_EXECUTE = True if ex in ('y','yes') else False
                print(f"記録先: {RECORD_FILE} 実行フラグ: {RECORD_EXECUTE}")
                return
            except Exception as e:
                print(f"ファイル作成エラー: {e}")
                RECORD_FILE = None
                RECORD_EXECUTE = True
                return
        elif choice in ('n','no',''):
            RECORD_FILE = None
            RECORD_EXECUTE = True
            return
        else:
            print("y または n を入力してください。")

def _save_record(original_lines, plotter_line):
    """記録フォーマット:
       - G... [元のスピーカ命令]
       - D1 <ms> [元のスピーカ命令]
       - S <speaker_cmd>  (スピーカ単独)
    """
    global RECORD_FILE
    if not RECORD_FILE:
        return
    try:
        speaker_str = ' '.join(original_lines).strip()
        if plotter_line:
            line = plotter_line.strip()
            if speaker_str:
                line = f"{line} {speaker_str}"
        else:
            line = f"S {speaker_str}"
        line += '\n'
        with _RECORD_LOCK:
            with open(RECORD_FILE, 'a', encoding='utf-8') as fw:
                fw.write(line)
    except Exception as e:
        print(f"記録保存エラー: {e}")

# --- デバイスクラス -----------------------------------------------------------------
class Speaker:
    def __init__(self, ser):
        self.ser = ser

    def write(self, line):
        text = str(line).strip()
        try:
            _safe_serial_write(self.ser, text)
            if self.ser:
                print(f"[Speaker] 送信: {text}")
                time.sleep(0.02)
        except Exception as e:
            print(f"Speaker send error: {e}")

    def play_a0_sound(self, idx):
        """STROKE_COUNT に対応するファイルを非同期再生する。./sounds/001.wav 形式を期待。"""
        # シリアルが無効（プロッタのみモードなど）の場合は音声も再生しない
        if self.ser is None:
            return

        try:
            fname = f"{int(idx):03d}"
            exts = ('.wav', '.mp3', '.m4a', '.aiff', '.aif')
            search_dirs = ['./sounds', '.']
            found = None
            for d in search_dirs:
                for ext in exts:
                    p = os.path.join(d, fname + ext)
                    if os.path.exists(p):
                        found = p
                        break
                if found:
                    break
            if not found:
                print(f"音声ファイルが見つかりません: {fname} (検索先: {search_dirs})")
                return
            _play_sound_file(found)
            print(f"[Speaker] 音声再生: {found}")
        except Exception as e:
            print(f"play_a0_sound エラー: {e}")

class Plotter:
    def __init__(self, ser):
        self.ser = ser

    def mapping(self, cmd_tokens):
        """スピーカ番号 -> プロッタ中心座標と feedrate を返す。"""
        num = int(cmd_tokens[1])
        grid_count = 8
        area_size = 200
        cell_size = area_size / grid_count
        zero_based = num - 1
        row = zero_based // grid_count
        col = zero_based % grid_count
        x_min = col * cell_size
        y_min = row * cell_size
        x_max = x_min + cell_size
        y_max = y_min + cell_size
        center_x = ((x_min + x_max) / 2) - 210
        center_y = -((y_min + y_max) / 2)
        delay = int(cmd_tokens[2])
        # feedrate 計算（既存ロジック維持）
        feed = 25 / (delay / 1000.0)
        feed = feed * 60
        return (center_x, center_y, feed)

    def write(self, center_x, center_y, feed, branch):
        if branch:
            line = f"G0 X{center_x:.3f} Y{center_y:.3f}"
        else:
            line = f"G1 X{center_x:.3f} Y{center_y:.3f} F{int(feed)}"
        try:
            _safe_serial_write(self.ser, line)
            if self.ser:
                print(f"[Plotter] 送信: {line}")
        except Exception as e:
            print(f"Plotter write error: {e}")

    def reset(self):
        _safe_serial_write(self.ser, "G0 X0 Y0 Z0")
        if self.ser:
            print("[Plotter] reset 送信")

    def down(self):
        _safe_serial_write(self.ser, "G0 Z8")
        if self.ser:
            print("[Plotter] down 送信")

    def up(self):
        _safe_serial_write(self.ser, "G0 Z0")
        if self.ser:
            print("[Plotter] up 送信")

    def sync(self, timeout=5.0):
        """タイムアウト付きでデバイス応答を待つ。応答形式に寛容に対応する。"""
        if not self.ser:
            # プロッタなしモードの場合、待機せずに即リターン（またはシミュレーション遅延を入れることも検討）
            # 現状はログも出さずにスキップするだけに留める（ログがうるさくなるため）
            return
        
        deadline = time.time() + timeout
        print("[Plotter] sync start")
        try:
            while time.time() < deadline:
                try:
                    self.ser.write(b'?\n')
                except Exception as e:
                    print(f"[Plotter] sync write error: {e}")
                    break
                start = time.time()
                while time.time() - start < 0.5:
                    try:
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    except Exception:
                        line = ''
                    if not line:
                        time.sleep(0.02)
                        continue
                    print(f"[Plotter] ステータス応答: {line}")
                    if ('Idle' in line) or line.lower().startswith('ok'):
                        print("[Plotter] 移動が完了しました")
                        print("[Plotter] sync end")
                        return
            print(f"[Plotter] sync タイムアウト ({timeout}s)。続行します。")
        except KeyboardInterrupt:
            print("処理を中断しました（sync）。")
        except Exception as e:
            print(f"[Plotter] sync 予期せぬエラー: {e}")
        print("[Plotter] sync end")

# --- 記録再生 -----------------------------------------------------------------------
def play_recorded_file(filename, pl, sp):
    """記録ファイルを読み込み、プロッタとスピーカを順に実行する"""
    if not os.path.exists(filename):
        print("ファイルが存在しません:", filename)
        return
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            print(f"'{filename}' を再生します．．．")
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw or raw.startswith('#'):
                    continue
                tokens = raw.split()
                head = tokens[0].upper()
                if head == 'S':
                    speaker_cmd = ' '.join(tokens[1:]) if len(tokens) > 1 else ''
                    if speaker_cmd:
                        sp.write(speaker_cmd)
                        print(f"[{lineno}] スピーカ単独送信: {speaker_cmd}")
                    continue
                if head == 'D1' or head.startswith('D'):
                    try:
                        ms = int(tokens[1]) if len(tokens) > 1 else 0
                    except Exception:
                        ms = 0
                    delay = (ms / 1000.0) + 0.05
                    print(f"[{lineno}] ホスト側スリープ: {delay}s")
                    time.sleep(delay)
                    if len(tokens) > 2:
                        speaker_cmd = ' '.join(tokens[2:])
                        sp.write(speaker_cmd)
                        print(f"[{lineno}] D1 行内 スピーカ送信: {speaker_cmd}")
                    continue
                if head.startswith('G'):
                    split_index = None
                    for i, t in enumerate(tokens):
                        if t and t[0].upper() in ('A','B','C','D','S'):
                            split_index = i
                            break
                    if split_index is not None:
                        plotter_line = ' '.join(tokens[:split_index])
                        speaker_cmd = ' '.join(tokens[split_index:])
                    else:
                        plotter_line = ' '.join(tokens)
                        speaker_cmd = None
                    
                    # Plotterへの送信 (serがNoneなら内部でスキップされる)
                    if pl:
                         try:
                            # ログは write 内部で出力条件分岐済みだが、sync呼び出し前に確認
                            if pl.ser:
                                print(f"[{lineno}] プロッタへ送信: {plotter_line}")
                            pl.write(None, None, None, None) # ※注意: この関数play_recorded_fileは元の設計で pl.write を直接呼んでいない。
                            # 元コードでは _safe_serial_write(pl.ser, plotter_line) を呼んでいたため、
                            # クラスメソッドではなく直接送信していました。修正します。
                            _safe_serial_write(pl.ser, plotter_line) 
                            pl.sync()
                         except Exception as e:
                            print(f"[{lineno}] プロッタ送信エラー: {e}")
                    
                    if speaker_cmd:
                        parts = speaker_cmd.split()
                        if parts and parts[0].upper() == 'A0':
                            global STROKE_COUNT
                            STROKE_COUNT += 1
                            try:
                                sp.play_a0_sound(STROKE_COUNT)
                            except Exception as e:
                                print(f"[{lineno}] A0 再生エラー: {e}")
                        sp.write(speaker_cmd)
                        if sp.ser:
                            print(f"[{lineno}] スピーカへ送信: {speaker_cmd}")
                    continue
                print(f"[{lineno}] 未知の行形式: {raw}")
    except Exception as e:
        print(f"再生エラー: {e}")

# --- メインロジック -----------------------------------------------------------------
def main():
    global STROKE_COUNT
    flag = True
    first_setup = True
    branch = 0
    ser1 = None # Speaker Serial
    ser2 = None # Plotter Serial

    select_recording()

    while flag:
        print("\n習字学習システム")
        
        if first_setup:
            mode = select_run_mode()

            # --- スピーカ接続 ---
            if mode in ['both', 'speaker_only']:
                ser1 = select_port("スピーカ")
                if ser1 is None:
                    print("スピーカポートが開けません。終了します。")
                    return
            else:
                print("スピーカ: 未接続モード")
                ser1 = None
            sp = Speaker(ser1)

            # --- プロッタ接続 ---
            if mode in ['both', 'plotter_only']:
                ser2 = select_port("プロッタ")
                if ser2 is None:
                    print("プロッタポートが開けません。終了します。")
                    # 開いていたスピーカも閉じる
                    if ser1: ser1.close()
                    return
            else:
                print("プロッタ: 未接続モード")
                ser2 = None
            
            pl = Plotter(ser2)
            if ser2:
                pl.reset()
            
            first_setup = False

        file = select_file()
        if not file:
            print("ファイルが選択されませんでした。終了します。")
            break

        use_record_playback = input("このファイルを記録済ファイルとして再生しますか？ (y/N): ").strip().lower() in ('y','yes')

        try:
            if use_record_playback:
                play_recorded_file(file, pl, sp)
            else:
                with open(file, 'r', encoding='utf-8') as f:
                    print(f"'{file}' の内容で動作を開始します．．．")
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        tmp = line.split()
                        cmd = tmp[0]
                        if cmd in ("A1","A2"):
                            data = pl.mapping(tmp)
                            print(f"変換後: {data}")
                            center_x, center_y, feed = data
                            plot_line = f"G1 X{center_x:.3f} Y{center_y:.3f} F{int(feed)}"
                            _save_record([line], plot_line)
                            if not (RECORD_FILE and not RECORD_EXECUTE):
                                if branch:
                                    pl.down()
                                    branch = 0
                                pl.write(center_x, center_y, feed, branch)
                                sp.write(line)
                            else:
                                print("保存のみモード: 実行をスキップ")
                        elif cmd == "A0":
                            data = pl.mapping(tmp)
                            print(f"変換後: {data}")
                            plot_line = f"G0 X{data[0]:.3f} Y{data[1]:.3f}"
                            _save_record([line], plot_line)
                            if not (RECORD_FILE and not RECORD_EXECUTE):
                                pl.up()
                                STROKE_COUNT += 1
                                print(f"画数: {STROKE_COUNT}")
                                try:
                                    sp.play_a0_sound(STROKE_COUNT)
                                except Exception as e:
                                    print(f"A0 音声再生エラー: {e}")
                                pl.write(data[0], data[1], data[2], 1)
                                time.sleep(0.5)
                            else:
                                print("保存のみモード: A0 実行をスキップ")
                            branch = 1
                        elif cmd in ("B1","B2","C"):
                            _save_record([line], None)
                            sp.write(line)
                        elif cmd == "D1":
                            _save_record([line], ' '.join(tmp))
                            if not (RECORD_FILE and not RECORD_EXECUTE):
                                try:
                                    ms = int(tmp[1])
                                except Exception:
                                    ms = 0
                                delay = (ms / 1000.0) + 0.05
                                print(f"delay処理:{ms}ms -> sleep {delay}s")
                                time.sleep(delay)
                            else:
                                print("保存のみモード: D1 実行をスキップ")
                        else:
                            print("形式が不正\n")
                        pl.sync()
        except KeyboardInterrupt:
            print("処理を中断しました。")
        except Exception as e:
            print(f"エラーが発生しました: {e}")

        time.sleep(0.5)
        pl.reset()

        print("つづけますか？ (y/N)")
        if yes_no_input():
            STROKE_COUNT = 0
            print("画数をリセットしました。")
        else:
            flag = False
            if ser1: ser1.close()
            if ser2: ser2.close()

if __name__ == "__main__":
    main()