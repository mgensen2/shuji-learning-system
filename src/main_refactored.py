import serial
from serial.tools import list_ports
import time
import os
import threading
import subprocess

def main():
    print("習字学習システム\n")
    ser1 = select_port("スピーカアレイのシリアルポート選択")
    sp = Speaker(ser1)
    ser2 = select_port("プロッタのシリアルポート選択")
    pl = Plotter(ser2)

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
        if not yes_no_input("つづけますか？"):
            break
        sp.a0_count=0  # A0カウントをリセット
    ser1.close()
    ser2.close()

def handle_command(tmp, pl, sp):
    cmd = tmp[0]
    if cmd == "C":
        print("config命令\n")
        # ここに設定処理を追加可能
    elif cmd in ("A1", "A2"):
        print("ホワイトノイズorバンドパスok")
        pl.down()  # ペンを下げる
        data = pl.mapping(tmp)
        print(f"変換後:{data}")
        # 並列実行
        t_plotter = threading.Thread(target=pl.write, args=(*data,), kwargs={'branch': 0})
        t_speaker = threading.Thread(target=sp.write, args=(' '.join(tmp),))
        t_plotter.start()
        t_speaker.start()
        t_plotter.join()
        t_speaker.join()
    elif cmd == "A0":
        pl.up()    # ペンを上げる
        data = pl.mapping(tmp)
        print(f"変換後:{data}")
        # A0カウントに応じた音声を再生（非同期）
        try:
            sp.play_a0_sound()
        except Exception as e:
            print(f"音声再生エラー: {e}")
        pl.write(*data, branch=1)
        time.sleep(0.5)
    elif cmd in ("B1", "B2"):
        print("複数命令スピーカ処理\n")
        # 実装例: sp.write(' '.join(tmp))
    elif cmd == "D1":
        delay = (int(tmp[1]) / 1000) + 0.1
        print(f"delay処理:{delay}ms\n")
        time.sleep(delay)
    else:
        print("形式が不正\n")
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
        choice = input(f"{msg} ").lower()
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False

class Speaker:
    def __init__(self, ser, sounds_dirs=None):
        self.ser = ser
        # 再生候補を検索するディレクトリ一覧
        self.sounds_dirs = sounds_dirs or ['.', './sounds']
        # A0 の呼び出し回数カウンタ（初回A0で1になる）
        self.a0_count = 0

    def write(self, line):
        line = str(line) + '\n'
        self.ser.write(line.encode('utf-8'))
        print(f"送信: {line.strip()}")
        time.sleep(0.1)
        print("送信が完了しました。")

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
        if branch:
            line = f'G0 X{center_x} Y{center_y}'
        else:
            line = f'G1 X{center_x} Y{center_y} F{delay}'
        line = str(line) + '\n'
        self.ser.write(line.encode('utf-8'))
        print(f"送信: {line.strip()}")
    def reset(self):
        line = "G0 X0 Y0 Z0\n"
        self.ser.write(line.encode('utf-8'))
    def down(self):
        line = "G0 Z10\n"
        self.ser.write(line.encode('utf-8'))
    def up(self):
        line = "G0 Z0\n"
        self.ser.write(line.encode('utf-8'))
    def sync(self):
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

if __name__ == "__main__":
    main()
