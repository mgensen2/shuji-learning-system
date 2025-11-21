import cv2
import time
import os
import sys
from datetime import datetime

# --- 設定 ---
OUTPUT_DIR = "recordings"
# カメラ解像度 (高いほど良いが、FPSが落ちない範囲で)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30.0 

def select_camera_index(prompt_text):
    print(f"--- {prompt_text} のカメラを選択 ---")
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    if not available: return None
    
    for index in available:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened(): continue
        print(f"Checking Camera {index}...")
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.putText(frame, f"Index: {index} ({prompt_text}) - Press 'y' to use, 'n' to skip", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Camera Select", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                cv2.destroyAllWindows()
                cap.release()
                return index
            if key == ord('n'):
                break
        cap.release()
    cv2.destroyAllWindows()
    return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. カメラ選択
    top_idx = select_camera_index("Top-Down (XY) Camera")
    if top_idx is None: sys.exit("Topカメラが必要です")
    
    side_idx = select_camera_index("Side-View (Z/Pressure) Camera")
    if side_idx is None: sys.exit("Sideカメラが必要です")

    cap_top = cv2.VideoCapture(top_idx)
    cap_side = cv2.VideoCapture(side_idx)

    # 解像度設定
    for cap in [cap_top, cap_side]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

    print("\n--- 録画待機モード ---")
    print(" [r] キー: 録画開始 / 停止")
    print(" [c] キー: ★完成画像の撮影★ (書き終わって筆をどけてから押す)")
    print(" [q] キー: 終了")

    is_recording = False
    writer_top = None
    writer_side = None
    session_name = ""
    
    while True:
        ret_top, frame_top = cap_top.read()
        ret_side, frame_side = cap_side.read()
        if not ret_top or not ret_side: break

        current_time = time.time()

        # 録画中なら書き込み
        if is_recording:
            if writer_top and writer_side:
                writer_top.write(frame_top)
                writer_side.write(frame_side)
            
            # 録画中表示
            cv2.circle(frame_top, (50, 50), 20, (0, 0, 255), -1)
            cv2.putText(frame_top, "REC", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 画面表示 (縮小して表示)
        show_top = cv2.resize(frame_top, (640, 360))
        show_side = cv2.resize(frame_side, (640, 360))
        cv2.imshow("Top", show_top)
        cv2.imshow("Side", show_side)

        key = cv2.waitKey(1) & 0xFF

        # [r] 録画の開始/停止
        if key == ord('r'):
            if not is_recording:
                # 開始
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_name = os.path.join(OUTPUT_DIR, f"session_{timestamp}")
                
                # VideoWriter初期化
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer_top = cv2.VideoWriter(f"{session_name}_top.mp4", fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                writer_side = cv2.VideoWriter(f"{session_name}_side.mp4", fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                
                is_recording = True
                print(f"--- 録画開始: {session_name} ---")
            else:
                # 停止
                is_recording = False
                if writer_top: writer_top.release()
                if writer_side: writer_side.release()
                writer_top = None
                writer_side = None
                print("--- 録画停止 ---")
                print("★ 筆を紙からどけて、[c]キーで完成画像を撮影してください！")

        # [c] 完成画像の撮影 (Capture)
        if key == ord('c'):
            if is_recording:
                print("エラー: 録画を停止してから撮影してください。")
            elif session_name == "":
                print("エラー: 先に録画を行ってください。")
            else:
                # Topカメラの今のフレームを「完成画像」として保存
                final_img_path = f"{session_name}_final.png"
                cv2.imwrite(final_img_path, frame_top)
                print(f"--- 完成画像を保存しました: {final_img_path} ---")
                print("これで解析の準備が整いました。[q]で終了できます。")

        if key == ord('q'):
            break

    cap_top.release()
    cap_side.release()
    if writer_top: writer_top.release()
    if writer_side: writer_side.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()