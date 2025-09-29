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
            print("\nプロッタのシリアルポート選択\n")
            ser2 = select_port()
            pl = Plotter(ser2)
            point = 0
        print("読み込みファイルを選択\n")
        file = select_file()
        try:
            with open(file, 'r') as f:
                print(f"'{file}'の内容で動作を開始します．．．")
                for line in f:#行が終わるまで繰り返し
                    line = line.strip("\n")
                    tmp=line.split(' ')#半角スペースで区切り
                    print(tmp)
                    #命令分岐
                    if tmp[0] == "C" :
                        print("config命令\n")
                    elif tmp[0] == "A1" or  tmp[0] == "A2": 
                        print("ホワイトノイズorバンドパスok")
                        data = pl.mapping(tmp)
                        print(f"変換後:{data}")
                        print("mapping ok")
                        pl.write(data[0],data[1],data[2],branch)
                        sp.write(line)
                        branch =0
                    elif tmp[0] == "A0" :
                        pl.up()
                        branch = 1
                    elif tmp[0] == "B1" or tmp[0] == "B2":
                        print("複数命令スピーカ処理\n")
                    else :
                        print("形式が不正\n")
        except Exception as e:#エラー表示
            print(f"エラーが発生しました: {e}")
        pl.up()
        pl.reset()
        print("つづけますか？\n")
        yn=yes_no_input()
        if(yn==0):
            flag=0#nの場合，繰り返し処理終了
            #ポートクローズ処理
            ser1.close()
            ser2.close()

def select_port():
    """
    利用可能なシリアルポートをリスト表示し、ユーザーに選択させて、
    そのポートを開いたserial.Serialオブジェクトを返す関数。
    デバイスの説明も表示する。
    """
    ser = serial.Serial()
    ser.baudrate = 115200  # ボーレート
    ser.timeout = 0.1      # タイムアウトの時間

    # 利用可能なシリアルポートの情報を取得
    ports = list_ports.comports()

    # ポートが見つからない場合
    if not ports:
        print("エラー: 利用可能なシリアルポートが見つかりません。")
        return None

    # ポートが1つだけ見つかった場合
    if len(ports) == 1:
        port = ports[0]
        print(f"ポートが1つ見つかりました: {port.device} ({port.description})")
        ser.port = port.device
    # ポートが複数見つかった場合
    else:
        print("利用可能なポート:")
        # enumerateを使い、インデックスとポート情報を同時に取得して表示
        for i, port in enumerate(ports):
            print(f"  {i}: {port.device} - {port.description}")
        
        try:
            # ユーザーに入力を促す
            num = int(input("接続するポートの番号を入力してください >> "))
            if num < 0 or num >= len(ports):
                print("エラー: 無効な番号です。")
                return None
            ser.port = ports[num].device
        except ValueError:
            print("エラー: 数字で入力してください。")
            return None

    # 選択されたポートを開く
    try:
        ser.open()
        print(f"ポート {ser.port} を開きました。")
        return ser
    except serial.SerialException as e:
        print(f"エラー: ポートを開けませんでした - {e}")
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
#yes or no判定関数
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
        line = str(line) + '\n'
        self.ser.write(line.encode('utf-8'))
        print(f"送信: {line.strip()}")
        time.sleep(0.1) # 必要に応じてディレイを調整
        print("送信が完了しました。")

class Plotter :
    def __init__(self,ser):
        self.ser=ser
        self.res="hoge"
    def mapping(self,line):
        num=line[1]#1番目のパラメータがスピーカ番号
        num = int(num)
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
        delay= line[2]#3番目のパラメータがdelay
        delay = int(delay)
        #grblの送り速度計算
        delay = 25 / (delay/1000)
        delay = delay * 60
        return(center_x,center_y,delay)
    def write(self,center_x,center_y,delay,branch):
        if branch :
            line = f'G0 X{center_x} Y{center_y}'
            line = str(line) + '\n'
            self.ser.write(line.encode('utf-8'))
            Plotter.sync(self)
            Plotter.down(self)
        else :
            line = f'G1 X{center_x} Y{center_y} F{delay}'
            line = str(line) + '\n'
            self.ser.write(line.encode('utf-8'))
            Plotter.sync(self)
        print(f"送信: {line.strip()}")
    def reset(self):#プロッターを初期位置に戻す
        line = "G0 X0 Y0"+ "\n"
        self.ser.write(line.encode('utf-8'))
    def down(self) :#ホルダー下げ
        line = "G0 Z8" + "\n"
        self.ser.write(line.encode('utf-8'))
    def up(self) :#ホルダー上げ
        line= "G0 Z0" + "\n"
        self.ser.write(line.encode('utf-8'))
    def sync(self) : #タイミング同調
        while self.res != "Idle" :
            tmp='?' + '\n'
            self.ser.write(tmp.encode('utf-8'))
            self.res = self.ser.readline()
            self.res =self.res.decode('utf-8')
            self.res= self.res[1:5]
            print(self.res)
            time.sleep(0.01)


#おまじない
if __name__ == "__main__":
    main()