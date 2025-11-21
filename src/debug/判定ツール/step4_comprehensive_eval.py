# step1, step3 の関数をインポートして使う、あるいは結果を集計する想定
import step1_score_images
import step3_score_csvs

def main():
    print("=== 総合評価開始 ===")
    
    # Step 1: 画像評価 (IoU)
    print("\n--- Step 1: Shape Evaluation (IoU) ---")
    # step1の関数を呼び出し（戻り値を返すようにstep1を少し改造すると良いですが、ここでは実行のみ）
    step1_score_images.main()
    
    # Step 3: 動的評価 (DTW, Velocity)
    print("\n--- Step 3: Dynamic Evaluation (DTW) ---")
    step3_score_csvs.main()
    
    print("\n=== 評価終了 ===")

if __name__ == "__main__":
    main()