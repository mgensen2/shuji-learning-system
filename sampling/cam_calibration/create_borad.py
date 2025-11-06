import cv2
import numpy as np

# --- パラメータ設定 ---
SQUARES_X = 7  # チェッカーボードの（内側の）角の数（横）
SQUARES_Y = 5  # チェッカーボードの（内側の）角の数（縦）
SQUARE_LENGTH = 0.03  # チェッカーボードの正方形の1辺の長さ（メートル単位）
MARKER_LENGTH = 0.015 # ArUcoマーカーの1辺の長さ（メートル単位）
DICTIONARY_NAME = cv2.aruco.DICT_4X4_100 # 使用するArUcoマーカーの辞書

# --- ボードの定義 ---
dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_NAME)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X,SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)

# --- 画像の生成 ---
# 画像サイズ（ピクセル）
IMG_WIDTH_PX = 2000
IMG_HEIGHT_PX = 1400

# ボードを描画
img = board.generateImage((IMG_WIDTH_PX, IMG_HEIGHT_PX))

# 画像を保存
output_filename = "charuco_board.png"
cv2.imwrite(output_filename, img)

print(f"ChArUcoボードを '{output_filename}' として保存しました。")
print("これを印刷して平らな板に貼り付けて使用してください。")

# (オプション) ボードの表示
cv2.imshow("ChArUco Board", img)
cv2.waitKey(0)
cv2.destroyAllWindows()