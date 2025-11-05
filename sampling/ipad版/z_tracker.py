import cv2
import mediapipe as mp
import time
import csv
import os
import numpy as np

# --- 設定 ---
CAMERA_INDEX = 0      # カメラの番号 (適宜調整)
OUTPUT_FILENAME = "z_data.csv"
FINGER_TIP = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP # 人差し指の先端

# --- 状態管理 ---
STATE_IDLE = 0            # 初期状態
STATE_CALIBRATE_MIN = 1   # 筆圧1 (上端) のキャリブレーション
STATE_CALIBRATE_MAX = 2   # 筆圧8 (下端) のキャリブレーション
STATE_TRACKING = 3        # トラッキング中
state = STATE_IDLE

# --- キャリブレーション用変数 ---
# y_min_pressure: 筆圧1 (ペンが上がった状態) のY座標
# y_max_pressure: 筆圧8 (ペンが下がった状態) のY座標
y_min_pressure = float('inf')
y_max_pressure = float('-inf')

# --- MediaPipe Hand ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,              # 検出する手を1つに
    min_detection_confidence=0.7, # 検出信頼度
    min_tracking_confidence=0.5   # 追跡信頼度
)

# --- データ記録 ---
z_axis_data = [] # ここにデータを溜め込む
start_time = time.time() # スクリプト開始時刻（PCの時計）

# --- ヘルパー関数 ---
def put_text(image, text, position):
    """画像に縁取り文字を描画する"""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def get_pressure_level(y_pos, y_min, y_max):
    """Y座標を1-8の筆圧レベルに変換する"""
    if y_max <= y_min:
        return 1 # キャリブレーションが不正な場合は 1 を返す
    
    # 0.0 (筆圧1) 〜 1.0 (筆圧8) の範囲に正規化
    normalized = (y_pos - y_min) / (y_max - y_min)
    
    # 0.0〜1.0の範囲外をクリップ
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # 0.0-1.0 を 0-7 の整数に変換し、+1 して 1-8 のレベルにする
    level = int(np.floor(normalized * 7.999)) + 1 # 7.999...で丸め誤差吸収
    return level

# --- メイン処理 ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"エラー: カメラ(インデックス {CAMERA_INDEX})を開けません。")
    exit()

print("カメラを起動しました。")
print("---------------------------------------------------------")
print("使い方:")
print(" 1. iPadの左上の黒い四角と、ペンの先端が映るようにカメラをセットします。")
print(" 2. [c] キーを押してキャリブレーションを開始します。")
print(" 3. (CALIBRATE MIN) ペンを「上げた」状態(筆圧1)で [c] を押します。")
print(" 4. (CALIBRATE MAX) ペンを「下げた」状態(筆圧8)で [c] を押します。")
print(" 5. (TRACKING) 自動でトラッキングが開始されます。iPadで描画してください。")
print(" 6. 終了したら [q] キーを押します。")
print("---------------------------------------------------------")

current_y = -1.0 # 現在のY座標
pressure_level = 0 # 現在の筆圧レベル

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを取得できません。")
            break

        # PCの現在時刻をエポック秒で取得
        current_time_pc = time.time()
        # 経過時間 (ミリ秒)
        elapsed_ms = (current_time_pc - start_time) * 1000

        # 左右反転 (自撮りモード) し、BGR -> RGB に変換
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # パフォーマンス向上のため
        
        # MediaPipeで処理
        results = hands.process(image)
        
        # BGRに戻す
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, _ = image.shape

        # --- 手の検出処理 ---
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 人差し指の先端のY座標を取得 (0.0〜1.0の正規化座標)
            tip_y_normalized = hand_landmarks.landmark[FINGER_TIP].y
            # ピクセル座標に変換
            current_y = tip_y_normalized * h

            # 手の骨格を描画
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 人差し指の先端に円を描画
            tip_x = hand_landmarks.landmark[FINGER_TIP].x * w
            cv2.circle(image, (int(tip_x), int(current_y)), 10, (0, 0, 255), -1)
        else:
            current_y = -1.0 # 手が検出されなかった

        # --- キー入力処理 ---
        key = cv2.waitKey(5) & 0xFF

        if key == ord('q'):
            print("終了します。")
            break
        
        if key == ord('c'):
            if state == STATE_IDLE:
                state = STATE_CALIBRATE_MIN
                print("-> STATE: CALIBRATE MIN (筆圧1: 上げた状態)")
            
            elif state == STATE_CALIBRATE_MIN:
                if current_y != -1.0:
                    y_min_pressure = current_y
                    state = STATE_CALIBRATE_MAX
                    print(f"  MIN Y (筆圧1) を {y_min_pressure:.2f} にセット")
                    print("-> STATE: CALIBRATE MAX (筆圧8: 下げた状態)")
                else:
                    print("  エラー: 手が検出されていません。")
            
            elif state == STATE_CALIBRATE_MAX:
                if current_y != -1.0:
                    y_max_pressure = current_y
                    print(f"  MAX Y (筆圧8) を {y_max_pressure:.2f} にセット")

                    # minとmaxが逆だったら入れ替える
                    if y_min_pressure > y_max_pressure:
                        y_min_pressure, y_max_pressure = y_max_pressure, y_min_pressure
                        print("  (Min/Maxが逆だったため入れ替えました)")
                    
                    print(f"キャリブレーション完了: Range [{y_min_pressure:.2f} - {y_max_pressure:.2f}]")
                    state = STATE_TRACKING
                    print("-> STATE: TRACKING (記録中...)")
                else:
                    print("  エラー: 手が検出されていません。")

        # --- 状態ごとの処理と描画 ---
        if state == STATE_IDLE:
            put_text(image, "Press [c] to start calibration", (20, 40))
        
        elif state == STATE_CALIBRATE_MIN:
            put_text(image, "STATE: CALIBRATE MIN (Pen UP)", (20, 40))
            put_text(image, "Hold pen UP (Pressure 1) and press [c]", (20, 70))
            if current_y != -1.0:
                cv2.line(image, (0, int(current_y)), (w, int(current_y)), (255, 255, 0), 2)
        
        elif state == STATE_CALIBRATE_MAX:
            put_text(image, "STATE: CALIBRATE MAX (Pen DOWN)", (20, 40))
            put_text(image, "Hold pen DOWN (Pressure 8) and press [c]", (20, 70))
            cv2.line(image, (0, int(y_min_pressure)), (w, int(y_min_pressure)), (0, 255, 0), 2) # Minライン
            if current_y != -1.0:
                cv2.line(image, (0, int(current_y)), (w, int(current_y)), (255, 255, 0), 2)

        elif state == STATE_TRACKING:
            put_text(image, "STATE: TRACKING... (Press [q] to stop)", (20, 40))
            
            # キャリブレーション範囲を描画
            cv2.line(image, (0, int(y_min_pressure)), (w, int(y_min_pressure)), (0, 255, 0), 2) # Min
            cv2.line(image, (0, int(y_max_pressure)), (w, int(y_max_pressure)), (0, 0, 255), 2) # Max
            
            if current_y != -1.0:
                pressure_level = get_pressure_level(current_y, y_min_pressure, y_max_pressure)
                put_text(image, f"Pressure Level: {pressure_level} / 8", (20, 70))
                
                # データを記録
                z_axis_data.append({
                    "timestamp_pc": current_time_pc, # 同期用の生データ (エポック秒)
                    "elapsed_ms": elapsed_ms,        # 確認用 (ミリ秒)
                    "y_position": current_y,         # 生のY座標
                    "pressure_level": pressure_level # 1-8
                })
            else:
                put_text(image, "Hand not detected", (20, 70))
                # 手が検出されなくても、時刻だけ記録（データ欠損として）
                z_axis_data.append({
                    "timestamp_pc": current_time_pc,
                    "elapsed_ms": elapsed_ms,
                    "y_position": -1.0,
                    "pressure_level": 0 # 0 = 欠損
                })

        # 画面表示
        cv2.imshow('MediaPipe Hand Z-Tracker', image)

finally:
    # --- 終了処理 ---
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # --- CSVに保存 ---
    if z_axis_data:
        print(f"データを {OUTPUT_FILENAME} に保存中...")
        try:
            with open(OUTPUT_FILENAME, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=z_axis_data[0].keys())
                writer.writeheader()
                writer.writerows(z_axis_data)
            print(f"保存完了: {os.path.abspath(OUTPUT_FILENAME)}")
        except Exception as e:
            print(f"CSVの保存中にエラーが発生しました: {e}")
    else:
        print("記録されたデータがありません。")