import cv2
import numpy as np

def thicken_red_lines(input_path, output_path, thickness=5):
    """
    画像の赤色部分を検出して太くする関数
    :param input_path: 入力画像のパス
    :param output_path: 出力画像のパス
    :param thickness: 太くする度合い（数値が大きいほど太くなります）
    """
    # 画像の読み込み
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"エラー: 画像が見つかりません ({input_path})")
        return

    # 色空間をBGRからHSVに変換（赤色の抽出をしやすくするため）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色の範囲を定義 (HSV空間)
    # 赤色は色相(H)が0付近と180付近の両端にあるため、2つの範囲を定義します
    
    # 範囲1 (0〜10)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    
    # 範囲2 (170〜180)
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # 赤色領域のマスクを作成
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # 線の太さを調整するカーネルを作成
    kernel = np.ones((thickness, thickness), np.uint8)

    # マスク画像を膨張させる（これで線が太くなります）
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # 結果画像の作成（元画像をコピー）
    result_img = img.copy()

    # 膨張させたマスクの部分を「赤色」で塗りつぶす
    # 色はBGR形式で指定 (青, 緑, 赤) -> (0, 0, 255)
    result_img[dilated_mask > 0] = [0, 0, 255]

    # 画像の保存
    cv2.imwrite(output_path, result_img)
    print(f"完了しました。保存先: {output_path}")

# --- 実行部分 ---
if __name__ == "__main__":
    # ここに入力ファイル名を指定してください
    input_file = "kou.png"   # 元の画像ファイル名
    output_file = "output.jpg" # 保存するファイル名
    
    # thicknessの値を変更すると太さが変わります（3, 5, 7...など）
    thicken_red_lines(input_file, output_file, thickness=10)