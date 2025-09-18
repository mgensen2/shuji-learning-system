import serial
from serial.tools import list_ports
import time
import os
import math
def select_port(): #ポート選択関数
    ser = serial.Serial()
    ser.baudrate = 11520    # esp32,プロッターのレート
    ser.timeout = 0.1       # タイムアウトの時間

    ports = list_ports.comports()    # ポートデータを取得
    
    devices = [info.device for info in ports]

    if len(devices) == 0:
        # シリアル通信できるデバイスが見つからなかった場合
        print("error: device not found")
        return None
    elif len(devices) == 1:
        print("only found %s" % devices[0])
        ser.port = devices[0]
    else:
        # ポートが複数見つかった場合それらを表示し選択させる
        for i in range(len(devices)):
            print("input %3d: open %s" % (i,devices[i]))
        print("input number of target port >> ",end="")
        num = int(input())
        ser.port = devices[num]
    
    # 開いてみる
    try:
        ser.open()
        return ser
    except:
        print("error when opening serial")
        return None

def select_file():
    """
    カレントディレクトリのファイルを表示し、ユーザーに選択させる関数
    """
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

class Speaker :
    def __init__(self,ser,filename):
        self.ser=ser
        self.filename=filename
    def write(self):
        try :
            with open(self.filename, 'r') as f:
                print(f"'{self.filename}'の内容を送信します...")
                for line in f:
                    tmpline = line
                    self.ser.write(line.encode('utf-8'))
                    print(f"送信: {line.strip()}")
                    time.sleep(0.1) # 必要に応じてディレイを調整
            print("送信が完了しました。")
        except FileNotFoundError:
            print(f"ファイルが見つかりません: {self.filename}")
        except Exception as e:
            print(f"エラーが発生しました: {e}")

class Plotter :
    def __init__(self,line):
        self.line=line
    def mapping(self):
        line = line.split(' ')
        num=line[2]
        total_areas = 64
        grid_count = 8
        area_size = 200
        # 入力された番号が1から64の範囲内かチェック
        # 1つのセルのサイズ
        cell_size = area_size / grid_count 
        # 200 / 8 = 25
        # エリア番号から行と列のインデックス（0から始まる）を逆算
        # 1-64 -> 0-63 に変換して計算しやすくする
        zero_based_number = num - 1
        # 行インデックス = 商
        row_index = zero_based_number // grid_count
        # 列インデックス = 余り
        col_index = zero_based_number % grid_count
        # インデックスから座標範囲を計算
        x_min = col_index * cell_size
        y_min = row_index * cell_size
        x_max = x_min + cell_size
        y_max = y_min + cell_size

