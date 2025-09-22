import serial
from serial.tools import list_ports
import time
import os
#メイン関数
def main():
    flag=1
    point=1
    branch=0
    while(flag):
        print("習字学習システム\n")
        if(point):#初回のみシリアル設定フラグ処理
            print("スピーカアレイのシリアルポート選択\n")
            ser1 = select_port()
            sp = Speaker(ser1)
            print("プロッタのシリアルポート選択\n")
            ser2 = select_port()
            pl = Plotter(ser2)
            point = 0
        print("読み込みファイルを選択\n")
        file = select_file()
        try:
            with open(file, 'r') as f:
                print(f"'{file}'の内容で動作を開始します．．．")
                for line in f:#行が終わるまで繰り返し
                    line=line.split(' ')#半角スペースで区切り
                    tmp = line
                    if tmp[0] == "A0" :#delay命令の場合，次の移動は高速移動
                        branch = 1
                    else :
                        branch = 0
                    data = pl.mapping(line)
                    sp.write(line)
                    pl.write(data[1],data[2],data[3],branch)
                        
                    
        except Exception as e:
            print(f"エラーが発生しました: {e}")
        print("つづけますか？ y or n\n")
        yn=yes_no_input()
        if(yn==0):
            flag=0#nの場合，繰り返し処理終了
            ser1.close()
            ser2.close()


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

def yes_no_input():
    while True:
        choice = input("Please respond with 'yes' or 'no' [y/N]: ").lower()
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False

class Speaker :
    def __init__(self,ser):
        self.ser=ser
    def write(self,line):
        line = line + '\n'
        self.ser.write(line.encode('utf-8'))
        print(f"送信: {self.line.strip()}")
        time.sleep(0.1) # 必要に応じてディレイを調整
        print("送信が完了しました。")

class Plotter :
    def __init__(self,ser):
        self.ser=ser
    def mapping(self,line):
        tmpline = line.split(' ')
        num=tmpline[1]#1番目のパラメータがスピーカ番号
        grid_count = 8
        area_size = 200
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
        #中心座標を計算(右上が原点でマイナス符号)
        center_x = ((x_min + x_max) / 2)-210
        center_y = -((y_min+ y_max) / 2)
        delay= tmpline[2]#2番目のパラメータがdelay
        #grblの送り速度計算
        delay = 60000 / delay
        return(center_x,center_y,delay)
    def write(self,center_x,center_y,delay,branch):
        if branch :
            line = f'G0 {center_x} {center_y}' + '\n'
        else :
            line = f'G1 {center_x} {center_y} F{delay}' + '\n'
        self.ser.write(line.encode('utf-8'))
        print(f"送信: {self.line.strip()}")
    def next(self):
        
        pass
#おまじない
if __name__ == "main":
    main()