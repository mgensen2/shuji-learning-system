import cv2
import os
import time

# --- パラメータ設定 ---
CAP_DEVICE_ID = 0  # カメラデバイスID (通常は0)
SAVE_DIR = "calibration_images"  # 画像の保存先ディレクトリ
IMG_WIDTH = 1280  # カメラの解像度（横）
IMG_HEIGHT = 720  # カメラの解像度（縦）

# 保存先ディレクトリを作成
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"ディレクトリ '{SAVE_DIR}' を作成しました。")

cap = cv2.VideoCapture(CAP_DEVICE_ID)
if not cap.isOpened():
    print(f"エラー: カメラ (ID: {CAP_DEVICE_ID}) を開けません。")
    exit()

# カメラの解像度を設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

print("--- 画像収集を開始します ---")
print("カメラウィンドウをアクティブにして操作してください。")
print(" [s] キー: 現在 S のフレームを保存")
print(" [q] キー: 収集を終了")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("エラー: フレームを取得できません。")
        break

    # 収集状況を画面に表示
    display_frame = frame.copy()
    cv2.putText(display_frame,
                f"Saved Images: {count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame,
                "Press [s] to save, [q] to quit",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image Collection - Press [s] to save, [q] to quit", display_frame)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # 画像を保存
        filename = os.path.join(SAVE_DIR, f"calib_{count:03d}.png")
        cv2.imwrite(filename, frame)
        print(f"保存しました: {filename}")
        count += 1
    
    elif key == ord('q'):
        # 終了
        print("収集を終了します。")
        break

cap.release()
cv2.destroyAllWindows()