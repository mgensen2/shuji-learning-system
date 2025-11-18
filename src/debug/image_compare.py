import cv2
import numpy as np
import sys
import os

# --- 設定項目 ---

# 1. 基準となる画像（お手本）
TARGET_IMAGE_FILE = 'calligraphy_sample.png' 
# ※ もし手元になければ、適当な画像を指定するか、下の「ダミー生成」機能が動きます

# 2. 比較したい画像（自分の書いた文字）
#    (前回のプログラムで保存した 'calligraphy_image.png' など)
INPUT_IMAGE_FILE = 'calligraphy_image.png'

# 3. 比較結果の保存ファイル名
OUTPUT_DIFF_IMAGE = 'comparison_result.png'

# 4. 画像処理サイズ (計算を合わせるためリサイズします)
PROCESS_SIZE = 800

# -----------------

def create_dummy_sample_if_missing(filepath):
    """お手本画像がない場合にダミーを作成する（テスト用）"""
    if not os.path.exists(filepath):
        print(f"--- {filepath} が見つかりません。ダミーのお手本を生成します。 ---")
        img = np.full((PROCESS_SIZE, PROCESS_SIZE, 3), 255, dtype=np.uint8)
        # 少しズレた位置に文字を書く
        cv2.putText(img, "Sumi", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 30, cv2.LINE_AA)
        cv2.line(img, (400, 400), (600, 700), (0,0,0), 20)
        cv2.imwrite(filepath, img)

def compare_calligraphy(target_path, input_path):
    print(f"--- 画像比較を開始します ---")
    print(f"お手本: {target_path}")
    print(f"入力  : {input_path}")

    # 1. 画像の読み込み
    img_target = cv2.imread(target_path)
    img_input = cv2.imread(input_path)

    if img_target is None:
        print(f"エラー: お手本画像 {target_path} が読み込めません。")
        return
    if img_input is None:
        print(f"エラー: 入力画像 {input_path} が読み込めません。")
        return

    # 2. サイズ統一 (お手本のサイズに合わせる、または固定サイズ)
    #    ここでは設定した PROCESS_SIZE に両方をリサイズします
    img_target = cv2.resize(img_target, (PROCESS_SIZE, PROCESS_SIZE))
    img_input = cv2.resize(img_input, (PROCESS_SIZE, PROCESS_SIZE))

    # 3. グレースケール化 & 2値化（白黒はっきりさせる）
    #    紙=白(255), 墨=黒(0) を前提とします
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
    gray_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    # Otsuの二値化で自動的にしきい値を決定
    # 反転させて「墨の部分を白(255)、紙を黒(0)」として扱うと計算しやすい
    _, bin_target = cv2.threshold(gray_target, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, bin_input = cv2.threshold(gray_input, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 4. 一致率 (IoU: Intersection over Union) の計算
    #    IoU = (重なっている面積) / (どちらか片方でも塗られている面積)
    
    # 重なっている部分 (AND演算: 両方255の場所)
    intersection = cv2.bitwise_and(bin_target, bin_input)
    
    # どちらかが塗られている部分 (OR演算: どちらかが255の場所)
    union = cv2.bitwise_or(bin_target, bin_input)

    # ピクセル数を数える
    count_intersection = cv2.countNonZero(intersection)
    count_union = cv2.countNonZero(union)

    if count_union == 0:
        score = 0.0
    else:
        score = count_intersection / count_union

    print(f"\n--- 判定結果 ---")
    print(f"一致率 (IoUスコア): {score * 100:.1f}%")
    
    if score > 0.8:
        print("評価: 大変素晴らしいです！ (Excellent)")
    elif score > 0.6:
        print("評価: よく書けています。 (Good)")
    else:
        print("評価: 形を意識して練習しましょう。 (Keep trying)")

    # 5. 違いの可視化画像の作成
    #    背景を白にする
    result_img = np.full((PROCESS_SIZE, PROCESS_SIZE, 3), 255, dtype=np.uint8)

    # マスクを作成
    # A: お手本にある (Target=255)
    # B: 入力にある (Input=255)
    
    # ケース1: 一致 (A and B) -> 黒 (0,0,0)
    mask_match = intersection # すでに計算済み

    # ケース2: 書き不足 (A and not B) -> 青 (255, 0, 0) ※OpenCVはBGR順
    # お手本にあるが、入力にはない
    mask_missing = cv2.bitwise_and(bin_target, cv2.bitwise_not(bin_input))

    # ケース3: はみ出し (not A and B) -> 赤 (0, 0, 255)
    # お手本にはないが、入力にはある
    mask_extra = cv2.bitwise_and(cv2.bitwise_not(bin_target), bin_input)

    # 色を塗る
    # 画像[マスク] = 色 (OpenCVの画像配列は [y, x] = [B, G, R])
    
    # 赤 (はみ出し)
    result_img[mask_extra == 255] = [0, 0, 255] 
    
    # 青 (書き不足)
    result_img[mask_missing == 255] = [255, 0, 0]
    
    # 黒 (一致) - 最後に塗ることで、境界線をきれいに見せる
    result_img[mask_match == 255] = [0, 0, 0]

    # お手本の輪郭線を薄い緑で重ねて、ガイドにする
    contours, _ = cv2.findContours