import serial
from serial.tools import list_ports
import time
import os
import threading
import sys
import shutil
import subprocess

# --- 定数 / グローバル ----------------------------------------------------------------
RECORD_FILE = None
RECORD_EXECUTE = True
_RECORD_LOCK = threading.Lock()

STROKE_COUNT = 0          # A0 を受けるたびにインクリメント

# ★ A0動作時の各ディレイ時間設定 (秒単位)
DELAY_A0_PRE_SOUND  = 1.0  # ① 前の画が終わってから、音声が流れる前の待機
DELAY_A0_POST_SOUND = 0.5  # ② 音声が喋り終わった後の待機
DELAY_A0_PRE_MOVE   = 1.0  # ③ その後、プロッタが移動し始める前の待機
DELAY_A0_POST_MOVE  = 1.0  # ④ 移動完了後、筆を下ろす前の待機

# ★ 終了時の設定
ENDING_SOUND_FILE = "end.wav"  # 終了時に流す音声ファイル名
DELAY_PRE_ENDING  = 1.0           # 終了音声が「流れる前」の待機時間
DELAY_ENDING      = 2.0           # 音声が「喋り終わった後」の待機時間

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

def _play_sound_file(path, wait=False):
    """
    音声ファイルを再生する。
    wait=True の場合、再生が終わるまで処理をブロックする。
    """
    if not os.path.exists(path):
        if wait:
            time.sleep(1.0)
        return

    try:
        # --- Mac (afplay) ---
        if sys.platform == 'darwin':
            if wait:
                subprocess.run(['afplay', path])
            else:
                subprocess.Popen(['afplay', path])

        # --- Windows (PowerShell SoundPlayer) ---
        elif sys.platform.startswith('win'):
            method = "PlaySync()" if wait else "Play()"
            cmd = f"(New-Object Media.SoundPlayer '{path}').{method}"
            subprocess.run(['powershell', '-c', cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # --- Linux (paplay/aplay/ffplay) ---
        else:
            player = None
            if shutil.which('paplay'):
                player = 'paplay'
            elif shutil.which('aplay'):
                player = 'aplay'
            elif shutil.which('ffplay'):
                player = 'ffplay'
            
            if player:
                args = [player, path]
                if player == 'ffplay':
                    args = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', path]
                
                if wait:
                    subprocess.run(args)
                else:
                    subprocess.Popen(args)
            else:
                print("音声再生コマンドが見つかりません (afplay/aplay/paplay).")

    except Exception as e:
        print(f"音声再生エラー: {e}")

# --- シリアル / UI ヘルパー ----------------------------------------------------------
def select_run_mode():
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

    def play_a0_sound(self, idx, wait=False):
        """
        wait=True なら再生完了を待つ。
        ★修正: プロッタモード(self.ser is None)でもPC音声を再生するため、
        シリアル接続チェックを外して再生処理を行うように変更。
        """
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
            
            _play_sound_file(found, wait=wait)
            print(f"[PC Audio] 音声再生完了: {found}")
            
        except Exception as e:
            print(f"play_a0_sound エラー: {e}")

class Plotter:
    def __init__(self, ser):
        self.ser = ser

    def mapping(self, cmd_tokens):
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
                print(f"[Plotter] 送信(write): {line}")
        except Exception as e:
            print(f"Plotter write error: {e}")

    def send_stream(self, command):
        if not self.ser:
            return
        line = str(command).strip()
        _safe_serial_write(self.ser, line)
        print(f"[Plotter] Stream送信: {line}")
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
        if not self.ser:
            return
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                try:
                    self.ser.write(b'?\n')
                except Exception as e:
                    print(f"[Plotter] sync write error: {e}")
                    break
                
                start = time.time()
                while time.time() - start < 0.2:
                    try:
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    except Exception:
                        line = ''
                    if not line:
                        continue
                    if 'Idle' in line:
                        return
        except KeyboardInterrupt:
            print("処理を中断しました（sync）。")
        except Exception as e:
            print(f"[Plotter] sync 予期せぬエラー: {e}")

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
                
                # --- スピーカ単独コマンド (S) ---
                if head == 'S':
                    speaker_cmd = ' '.join(tokens[1:]) if len(tokens) > 1 else ''
                    if speaker_cmd:
                        sp.write(speaker_cmd)
                        print(f"[{lineno}] スピーカ単独送信: {speaker_cmd}")
                    continue

                # --- 遅延コマンド (D1) ---
                if head == 'D1' or head.startswith('D'):
                    try:
                        ms = int(tokens[1]) if len(tokens) > 1 else 0
                    except Exception:
                        ms = 0
                    delay = (ms / 1000.0) + 0.05
                    print(f"[{lineno}] ホスト側スリープ: {delay}s")
                    if pl: pl.sync() 
                    time.sleep(delay)
                    if len(tokens) > 2:
                        speaker_cmd = ' '.join(tokens[2:])
                        sp.write(speaker_cmd)
                        print(f"[{lineno}] D1 行内 スピーカ送信: {speaker_cmd}")
                    continue
                
                # --- プロッタコマンド (G) ---
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
                    
                    parts = speaker_cmd.split() if speaker_cmd else []
                    is_a0 = (parts and parts[0].upper() == 'A0')
                    
                    # ★ A0の場合: 音声 -> 移動 -> 移動後Delay
                    if is_a0:
                        if pl: pl.sync()

                        # 1. 音声関連
                        time.sleep(DELAY_A0_PRE_SOUND)
                        global STROKE_COUNT
                        STROKE_COUNT += 1
                        try:
                            sp.play_a0_sound(STROKE_COUNT, wait=True)
                        except Exception as e:
                            print(f"[{lineno}] A0 再生エラー: {e}")
                        
                        if speaker_cmd:
                             sp.write(speaker_cmd)
                             if sp.ser: print(f"[{lineno}] スピーカへ送信: {speaker_cmd}")

                        time.sleep(DELAY_A0_POST_SOUND)

                        # 2. 移動関連
                        time.sleep(DELAY_A0_PRE_MOVE)
                        if pl:
                            try:
                                pl.send_stream(plotter_line)
                                pl.sync()
                            except Exception as e:
                                print(f"[{lineno}] プロッタ送信エラー: {e}")
                        
                        time.sleep(DELAY_A0_POST_MOVE)

                    else:
                        # A0以外 (通常の描画など)
                        # ★修正: スピーカのみモードの時、実行速度が速すぎて音が被るのを防ぐため、擬似的に待機する
                        if (not pl.ser) and speaker_cmd and (parts and parts[0] in ['A1', 'A2']):
                             try:
                                 # A1 1 500 ... の 500(ms) を取得
                                 ms = int(parts[2])
                                 sim_delay = ms / 1000.0
                                 # print(f"  [Simulate] Drawing wait: {sim_delay}s")
                                 time.sleep(sim_delay)
                             except Exception:
                                 pass

                        if pl:
                            try:
                                pl.send_stream(plotter_line)
                            except Exception as e:
                                print(f"[{lineno}] プロッタ送信エラー: {e}")
                        if speaker_cmd:
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
                            center_x, center_y, feed = data
                            plot_line = f"G1 X{center_x:.3f} Y{center_y:.3f} F{int(feed)}"
                            
                            _save_record([line], plot_line)
                            
                            if not (RECORD_FILE and not RECORD_EXECUTE):
                                if branch:
                                    pl.down()
                                    branch = 0
                                
                                # ★修正: スピーカのみモードでの高速重複防止
                                if (not pl.ser):
                                    try:
                                        ms = int(tmp[2])
                                        sim_delay = ms / 1000.0
                                        time.sleep(sim_delay)
                                    except:
                                        pass
                                
                                pl.send_stream(plot_line)
                                sp.write(line)
                            else:
                                print("保存のみモード: 実行をスキップ")

                        elif cmd == "A0":
                            data = pl.mapping(tmp)
                            print(f"変換後: {data}")
                            plot_line = f"G0 X{data[0]:.3f} Y{data[1]:.3f}"
                            _save_record([line], plot_line)
                            
                            if not (RECORD_FILE and not RECORD_EXECUTE):
                                # ★ A0専用ロジック (順番: 音 -> 移動 -> 移動後Delay)
                                pl.sync() # 直前の動きを止める

                                # 1. 音声処理
                                time.sleep(DELAY_A0_PRE_SOUND)
                                STROKE_COUNT += 1
                                print(f"画数: {STROKE_COUNT}")
                                try:
                                    sp.play_a0_sound(STROKE_COUNT, wait=True)
                                except Exception as e:
                                    print(f"A0 音声再生エラー: {e}")
                                time.sleep(DELAY_A0_POST_SOUND)

                                # 2. 移動処理
                                time.sleep(DELAY_A0_PRE_MOVE)
                                pl.up()
                                pl.write(data[0], data[1], data[2], 1)
                                pl.sync() 
                                
                                # 3. 移動後ディレイ
                                time.sleep(DELAY_A0_POST_MOVE)

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
                                pl.sync()
                                time.sleep(delay)
                            else:
                                print("保存のみモード: D1 実行をスキップ")
                        else:
                            print("形式が不正\n")
                        
        except KeyboardInterrupt:
            print("処理を中断しました。")
        except Exception as e:
            print(f"エラーが発生しました: {e}")

        # --- 終了時の音声再生とディレイ ---
        print("\n--- 処理完了 ---")
        
        print(f"終了前待機: {DELAY_PRE_ENDING}秒...")
        time.sleep(DELAY_PRE_ENDING)

        if os.path.exists(ENDING_SOUND_FILE):
             print(f"終了音声 ('{ENDING_SOUND_FILE}') を再生します...")
             _play_sound_file(ENDING_SOUND_FILE, wait=True)
        else:
             print(f"終了音声 ('{ENDING_SOUND_FILE}') が見つからないためスキップします。")

        print(f"終了後待機: {DELAY_ENDING}秒...")
        time.sleep(DELAY_ENDING)
        
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