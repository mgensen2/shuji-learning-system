import serial
from serial.tools import list_ports
import time
import os
import threading
import subprocess
import queue

# --- 追加: 変換座標保存設定とロック ---
OUTPUT_COORDS_FILE = None        # 保存先ファイルパスまたは None
OUTPUT_COORDS_EXECUTE = True     # True: 保存しつつデバイスも動かす / False: 保存のみ（デバイスは動かさない）
_OUTPUT_COORDS_LOCK = threading.Lock()

def select_coords_logging():
    """変換した座標をファイルに保存するか選択し、グローバルを設定する。"""
    global OUTPUT_COORDS_FILE, OUTPUT_COORDS_EXECUTE
    while True:
        choice = input("変換座標をテキストに保存しますか？ (y/N): ").strip().lower()
        if choice in ('y', 'yes'):
            fname = input("保存するファイル名を入力（空欄で 'converted_coords.txt'）: ").strip()
            if not fname:
                fname = "converted_coords.txt"
            try:
                # ファイルを作成して書き込み権限を確認（追記モード）
                with open(fname, 'a', encoding='utf-8'):
                    pass
                OUTPUT_COORDS_FILE = fname
                ex = input("保存時にプロッタとスピーカを実際に動かしますか？ (y/N) : ").strip().lower()
                OUTPUT_COORDS_EXECUTE = True if ex in ('y', 'yes') else False
                print(f"座標保存: {OUTPUT_COORDS_FILE} 実行フラグ: {OUTPUT_COORDS_EXECUTE}")
                return
            except Exception as e:
                print(f"ファイル作成エラー: {e}")
                return
        elif choice in ('n', 'no', ''):
            OUTPUT_COORDS_FILE = None
            OUTPUT_COORDS_EXECUTE = True
            return
        else:
            print("y または n を入力してください。")

def _save_converted_coords(original_cmd_parts, data):
    """センター座標と delay をテキストファイルに追記する。thread-safe。"""
    global OUTPUT_COORDS_FILE
    if not OUTPUT_COORDS_FILE or not data:
        return
    try:
        center_x, center_y, delay = data
        cmd_str = ' '.join(original_cmd_parts)
        line = f"{center_x:.3f} {center_y:.3f} {int(delay)} {cmd_str}\n"
        with _OUTPUT_COORDS_LOCK:
            with open(OUTPUT_COORDS_FILE, 'a', encoding='utf-8') as fw:
                fw.write(line)
    except Exception as e:
        print(f"座標保存エラー: {e}")

def main():
    print("習字学習システム\n")
    mode = select_mode()
    ser1 = select_port("スピーカアレイのシリアルポート選択")
    if ser1 is None:
        print("スピーカポートが開けませんでした。終了します。")
        return
    sp = Speaker(ser1)
    ser2 = select_port("プロッタのシリアルポート選択")
    if ser2 is None:
        print("プロッタポートが開けませんでした。終了します。")
        try:
            sp.stop()
        except Exception:
            pass
        return
    pl = Plotter(ser2)

    # 追加: 起動時に座標保存設定を確認
    select_coords_logging()

    if mode == "reverse":
        # 座標ファイルを読み込んで逆変換モードで実行
        reverse_mode_file(pl, sp)
    else:
        # 通常モード（既存の処理）
        while True:
            file = select_file()
            if not file:
                print("ファイルが選択されませんでした。終了します。")
                break
            try:
                with open(file, 'r') as f:
                    print(f"'{file}'の内容で動作を開始します．．．")

                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        tmp = line.split()
                        print(f"命令: {tmp}")
                        handle_command(tmp, pl, sp)
            except Exception as e:
                print(f"エラーが発生しました: {e}")
            except KeyboardInterrupt:
                print("処理を中断しました。")
                break
            time.sleep(0.5)
            pl.reset()
            # 続けますかで 'y' を押したときに A0 カウントをリセット
            cont = yes_no_input("つづけますか？")
            if cont:
                sp.a0_count = 0
            else:
                break
    # ワーカー停止してからシリアルを閉じる
    try:
        sp.stop()
    except Exception:
        pass
    if ser1:
        try:
            ser1.close()
        except Exception:
            pass
    if ser2:
        try:
            ser2.close()
        except Exception:
            pass

def select_mode():
    print("モードを選択してください:")
    print("  1: 通常（命令ファイル→座標変換）")
    print("  2: 逆変換（座標ファイル→スピーカ命令生成 + プロッタ移動）")
    while True:
        choice = input("番号を入力してください: ").strip()
        if choice == "1":
            return "normal"
        elif choice == "2":
            return "reverse"
        else:
            print("無効な番号です。1か2を入力してください。")

def reverse_mode_file(pl, sp):
    """
    ファイルに書かれた座標行を読み込み、対応するスピーカ命令（例: A1 num delay）を生成して送る。
    ファイルの各行フォーマット:
      X Y F [CMD]
    例:
      10.0 -5.0 200 A1
      -5.0 20.0 150
    CMD を省略した場合は A1 を使う。
    """
    file = select_file()
    if not file:
        print("ファイルが選択されませんでした。終了します。")
        return
    try:
        with open(file, 'r') as f:
            print(f"'{file}' の座標リストを逆変換で処理します．．．")
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    print(f"[{lineno}] フォーマットエラー（X Y F が必要）: {line}")
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    fval = int(float(parts[2]))
                    cmd_type = parts[3] if len(parts) >= 4 else "A1"
                except Exception as e:
                    print(f"[{lineno}] 数値変換エラー: {e} -- {line}")
                    continue
                num = coordinates_to_num(pl, x, y)
                if num is None:
                    print(f"[{lineno}] 座標が範囲外です: X={x}, Y={y}")
                    continue
                tmp = [cmd_type, str(num), str(fval)]
                print(f"[{lineno}] 生成命令: {tmp}")
                # 既存の handle_command を使って送信・描画・同期を行う
                try:
                    handle_command(tmp, pl, sp)
                except KeyboardInterrupt:
                    print("処理が中断されました。")
                    break
                except Exception as e:
                    print(f"[{lineno}] 実行エラー: {e}")
    except Exception as e:
        print(f"ファイル読み込みエラー: {e}")

def coordinates_to_num(pl, center_x, center_y):
    """
    Plotter.mapping() の逆変換を行う（pl の設定に合わせて）。
    return: num (1..grid_count*grid_count) or None
    """
    try:
        grid_count = 8
        area_size = 200
        cell_size = area_size / grid_count
        # mapping() で center_x = ((x_min + x_max)/2) - 210
        # and center_y = -((y_min + y_max)/2)
        x_adj = center_x + 210
        y_adj = -center_y
        if x_adj < 0 or y_adj < 0:
            return None
        col_index = int(x_adj // cell_size)
        row_index = int(y_adj // cell_size)
        if 0 <= col_index < grid_count and 0 <= row_index < grid_count:
            num = row_index * grid_count + col_index + 1
            return num
        return None
    except Exception:
        return None

def handle_command(tmp, pl, sp):
    cmd = tmp[0]
    # 変更点: OUTPUT_COORDS_FILE が有効で OUTPUT_COORDS_EXECUTE が False のときは
    # 「保存のみ」モードとしてデバイス実行をスキップする。
    if cmd == "C":
        print("config命令（キューに積む）\n")
        sp.enqueue(' '.join(tmp))
    elif cmd in ("A1", "A2"):
        print("ホワイトノイズorバンドパスok")
        data = pl.mapping(tmp)
        print(f"変換後:{data}")
        # 変換結果をファイルに保存（有効な場合）
        _save_converted_coords(tmp, data)
        # 保存のみモードならデバイスは動かさない
        if OUTPUT_COORDS_FILE and not OUTPUT_COORDS_EXECUTE:
            print("保存のみモード: プロッタ・スピーカの実行をスキップします。")
            return
        # 通常処理
        pl.down()  # ペンを下げる
        t_plotter = threading.Thread(target=pl.write, args=(*data,), kwargs={'branch': 0})
        t_plotter.start()
        sp.enqueue(' '.join(tmp))
        t_plotter.join()
    elif cmd == "A0":
        data = pl.mapping(tmp)
        print(f"変換後:{data}")
        # 変換結果をファイルに保存（有効な場合）
        _save_converted_coords(tmp, data)
        # 保存のみモードならデバイスは動かさない
        if OUTPUT_COORDS_FILE and not OUTPUT_COORDS_EXECUTE:
            print("保存のみモード: A0 の実行をスキップします。")
            return
        # 通常処理
        pl.up()    # ペンを上げる
        try:
            sp.play_a0_sound()
        except Exception as e:
            print(f"音声再生エラー: {e}")
        pl.write(*data, branch=1)
        time.sleep(0.5)
    elif cmd in ("B1", "B2"):
        print("複数命令スピーカ処理（キューへ）\n")
        sp.enqueue(' '.join(tmp))
    elif cmd == "D1":
        delay = (int(tmp[1]) / 1000) + 0.1
        print(f"delay処理:{delay}ms\n")
        time.sleep(delay)
    else:
        print("形式が不正\n")
    # sync はデバイス実行した場合のみ呼ぶ（保存のみなら skip）
    if not (OUTPUT_COORDS_FILE and not OUTPUT_COORDS_EXECUTE):
        pl.sync()

def select_port(title):
    print(f"{title}\n")
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.timeout = 0.1
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
        for i, port in enumerate(ports):
            print(f"  {i}: {port.device} - {port.description}")
        while True:
            try:
                num = int(input("接続するポートの番号を入力してください >> "))
                if 0 <= num < len(ports):
                    ser.port = ports[num].device
                    break
                else:
                    print("エラー: 無効な番号です。")
            except ValueError:
                print("エラー: 数字で入力してください。")
    try:
        ser.open()
        print(f"ポート {ser.port} を開きました。")
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

def yes_no_input(msg="Please respond with 'yes' or 'no' [y/N]: "):
    while True:
        choice = input(f"{msg} ").lower().strip()
        if choice == "":
            return False
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("y または n を入力してください。")

class Speaker:
    def __init__(self, ser, sounds_dirs=None):
        self.ser = ser
        # 再生候補を検索するディレクトリ一覧
        self.sounds_dirs = sounds_dirs or ['.', './sounds']
        # A0 の呼び出し回数カウンタ（初回A0で1になる）
        self.a0_count = 0

        # キューとワーカー（Cコマンド等の間引き・バッチ送信用）
        self._q = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def write(self, line):
        line = str(line).strip() + '\n'
        try:
            if self.ser:
                self.ser.write(line.encode('utf-8'))
            print(f"送信: {line.strip()}")
            # 軽いウェイトで連続送信の負荷を低減
            time.sleep(0.01)
        except Exception as e:
            print(f"Speaker write error: {e}")

    def enqueue(self, line):
        # 非同期で送る（バーストはワーカーで間引かれる）
        try:
            self._q.put(line)
        except Exception as e:
            print(f"enqueue error: {e}")

    def _worker_loop(self):
        # バースト時は最後のコマンドだけを送る（短時間の連続Cを間引く）
        BATCH_INTERVAL = 0.05  # 50ms
        while not self._stop_event.is_set():
            try:
                item = self._q.get(timeout=BATCH_INTERVAL)
                latest = item
                # すぐに溜まった分を取り出して最後のものだけにする
                while True:
                    try:
                        nxt = self._q.get_nowait()
                        latest = nxt
                    except queue.Empty:
                        break
                self._send_raw(latest)
            except queue.Empty:
                continue
        # 終了時に残りをフラッシュ
        while True:
            try:
                item = self._q.get_nowait()
                self._send_raw(item)
            except queue.Empty:
                break

    def _send_raw(self, line):
        try:
            # write は改行を追加するのでそのまま渡す
            self.write(line)
        except Exception as e:
            print(f"Speaker send error: {e}")

    def stop(self):
        self._stop_event.set()
        self._worker.join(timeout=1)

    def _find_file_for_index(self, index):
        base = f"{index:03d}"
        exts = ('.mp3', '.wav', '.m4a', '.aiff', '.aif', '.aac')
        for d in self.sounds_dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                name_lower = f.lower()
                # 接頭辞が 001 などに一致するファイルを優先して探す
                if name_lower.startswith(base) and name_lower.endswith(exts):
                    return os.path.join(d, f)
            # 直接拡張子付きファイル名を検索（例: ./001.mp3）
            for ext in exts:
                p = os.path.join(d, base + ext)
                if os.path.exists(p):
                    return p
        return None

    def play_a0_sound(self):
        # カウントを増やして対応するファイル名を探す
        self.a0_count += 1
        path = self._find_file_for_index(self.a0_count)
        if not path:
            print(f"A0回数 {self.a0_count} に対応する音声ファイルが見つかりません（検索先: {self.sounds_dirs}）。")
            return
        # macOS の afplay を使って非同期再生
        try:
            subprocess.Popen(['afplay', path])
            print(f"音声を再生しました: {path}")
        except FileNotFoundError:
            print("afplay が見つかりません。別の方法で再生してください。")
        except Exception as e:
            print(f"音声再生エラー: {e}")

class Plotter:
    def __init__(self, ser):
        self.ser = ser
    def mapping(self, line):
        num = int(line[1])
        grid_count = 8
        area_size = 200
        cell_size = area_size / grid_count
        zero_based_number = num - 1
        row_index = zero_based_number // grid_count
        col_index = zero_based_number % grid_count
        x_min = col_index * cell_size
        y_min = row_index * cell_size
        x_max = x_min + cell_size
        y_max = y_min + cell_size
        center_x = ((x_min + x_max) / 2) - 210
        center_y = -((y_min + y_max) / 2)
        delay = int(line[2])
        delay = 25 / (delay / 1000)
        delay = delay * 60
        return (center_x, center_y, delay)
    def write(self, center_x, center_y, delay, branch=0):
        if not self.ser:
            print("Plotter: シリアルが開いていません (write をスキップ)。")
            return
        try:
            if branch:
                line = f'G0 X{center_x} Y{center_y}'
            else:
                line = f'G1 X{center_x} Y{center_y} F{delay}'
            line = str(line) + '\n'
            self.ser.write(line.encode('utf-8'))
            print(f"送信: {line.strip()}")
        except Exception as e:
            print(f"Plotter write error: {e}")
    def reset(self):
        if not self.ser:
            print("Plotter: シリアルが開いていません (reset をスキップ)。")
            return
        try:
            line = "G0 X0 Y0 Z0\n"
            self.ser.write(line.encode('utf-8'))
        except Exception as e:
            print(f"Plotter reset error: {e}")
    def down(self):
        if not self.ser:
            print("Plotter: シリアルが開いていません (down をスキップ)。")
            return
        try:
            line = "G0 Z8\n"
            self.ser.write(line.encode('utf-8'))
        except Exception as e:
            print(f"Plotter down error: {e}")
    def up(self):
        if not self.ser:
            print("Plotter: シリアルが開いていません (up をスキップ)。")
            return
        try:
            line = "G0 Z0\n"
            self.ser.write(line.encode('utf-8'))
        except Exception as e:
            print(f"Plotter up error: {e}")
    def sync(self):
        if not self.ser:
            print("Plotter: シリアルが開いていません (sync をスキップ)。")
            return
        print("sync start\n")
        while True:
            try:
                self.ser.write(b'?\n')
                response = self.ser.readline().decode('utf-8').strip()
                if response.startswith('<') and response.endswith('>'):
                    print(f"ステータス: {response}")
                    if 'Idle' in response:
                        print("\n移動が完了しました")
                        break
                time.sleep(0.1)
            except serial.SerialException as e:
                print(f"シリアル通信エラー: {e}")
                break
            except KeyboardInterrupt:
                print("処理を中断しました。")
                break
        print("sync end\n")

# ファイルをスクリプトとして直接実行できるようにする
if __name__ == "__main__":
    main()
