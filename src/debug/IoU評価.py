import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 設定 ---
INPUT_CSV = 'IoU_result.csv'
FONT_NAME = 'MS Gothic' # Windows用 (Macなら 'Hiragino Sans')

# グラフの表示順（聴覚→触覚→提案）とラベル名
CONDITION_ORDER = ['C', 'B', 'A']
CONDITION_LABELS = {
    'A': 'A:提案手法\n(両方)', 
    'B': 'B:触覚のみ', 
    'C': 'C:聴覚のみ'
}

def main():
    # 1. データ読み込み
    if not os.path.exists(INPUT_CSV):
        print(f"エラー: {INPUT_CSV} が見つかりません。")
        return
    try:
        df = pd.read_csv(INPUT_CSV)
    except:
        df = pd.read_csv(INPUT_CSV, encoding='shift_jis')
        
    # ラベルの置換（改行を入れて幅を抑える）
    df['Condition_Name'] = df['Condition'].map(CONDITION_LABELS)
    
    # フォント設定
    plt.rcParams['font.family'] = FONT_NAME

    # --- Level 1: 全体比較 (棒グラフ) ---
    print("\n【Level 1】全体比較")
    # 集計
    summary1 = df.groupby('Condition')['IoU'].agg(['mean', 'std', 'sem', 'count'])
    print(summary1)
    summary1.to_csv('L1_Overall_Summary.csv', encoding='utf-8-sig')
    
    # グラフ
    plt.figure(figsize=(8, 6), constrained_layout=True) # 重なり防止レイアウト
    sns.barplot(data=df, x='Condition', y='IoU', order=CONDITION_ORDER, 
                palette='viridis', capsize=.1, errorbar='se')
    
    # X軸ラベルを置換して読みやすく
    plt.xticks(ticks=[0, 1, 2], labels=[CONDITION_LABELS[c] for c in CONDITION_ORDER])
    plt.title('Level 1: 全体比較 (条件ごとの平均IoU)')
    plt.ylabel('IoU Score')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('L1_Overall_Graph.png')
    
    # --- Level 2: 文字別 (棒グラフ) ---
    print("\n【Level 2】文字別")
    summary2 = df.groupby(['Correct_Char', 'Condition'])['IoU'].agg(['mean', 'std', 'count'])
    print(summary2)
    summary2.to_csv('L2_Character_Summary.csv', encoding='utf-8-sig')
    
    plt.figure(figsize=(10, 6), constrained_layout=True)
    sns.barplot(data=df, x='Correct_Char', y='IoU', hue='Condition', 
                hue_order=CONDITION_ORDER, palette='viridis', capsize=.1)
    
    # 凡例のラベルをわかりやすく
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [CONDITION_LABELS[l] for l in CONDITION_ORDER]
    plt.legend(handles, new_labels, title='条件', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.title('Level 2: 文字種別のIoU比較')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('L2_Character_Graph.png')

    # --- Level 3: 個人別 (折れ線) ---
    print("\n【Level 3】個人別")
    summary3 = df.pivot_table(index='Subject_ID', columns='Condition', values='IoU')
    print(summary3)
    summary3.to_csv('L3_Individual_Summary.csv', encoding='utf-8-sig')
    
    plt.figure(figsize=(11, 6), constrained_layout=True)
    
    # 個人ごとの変化をプロット
    sns.pointplot(data=df, x='Condition', y='IoU', hue='Subject_ID', 
                  order=CONDITION_ORDER, dodge=True, markers='o', scale=0.8)
    
    plt.xticks(ticks=[0, 1, 2], labels=[CONDITION_LABELS[c] for c in CONDITION_ORDER])
    plt.title('Level 3: 個人ごとのIoU変化 (ID別)')
    plt.ylabel('Mean IoU Score')
    
    # 凡例を外に出す (グラフと重ならないように)
    plt.legend(title='Subject ID', bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('L3_Individual_Graph.png')
    
    print("\n全分析完了。画像とCSVを確認してください。")

if __name__ == "__main__":
    main()