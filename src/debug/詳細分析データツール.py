import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 設定 ---
INPUT_FILE = '書字評価システム実験追加アンケート(1-10).csv' # ファイル名を確認してください
FONT_NAME = 'MS Gothic' 

# 条件名の定義（CSV内の文字列の一部が含まれていればOK）
# KEY: 表示名, VALUE: 検索キーワード
CONDITIONS = {
    'A:提案手法': '提案',
    'B:触覚のみ': '触覚',
    'C:聴覚のみ': '聴覚'
}
ORDER_LIST = list(CONDITIONS.keys())

def main():
    # データ読み込み
    if not os.path.exists(INPUT_FILE):
        print(f"エラー: {INPUT_FILE} が見つかりません。")
        return
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        df = pd.read_csv(INPUT_FILE, encoding='shift_jis')

    # 順位データの列を探す
    rank_col = [c for c in df.columns if '順位' in c or '並べて' in c][0]
    
    # --- 集計用データフレーム作成 ---
    # 行：条件、列：順位(1位=Hard, 2位, 3位=Easy)
    rank_stats = {name: {1:0, 2:0, 3:0} for name in CONDITIONS}
    
    # ペアワイズ比較（勝敗表） [Easy > Hard]
    # 例: "A_vs_C": [Aが勝ち(Easy), Cが勝ち(Easy), 引き分け]
    pairwise = {
        'A_vs_C': {'win': 0, 'lose': 0, 'draw': 0}, # 提案 vs 聴覚
        'A_vs_B': {'win': 0, 'lose': 0, 'draw': 0}  # 提案 vs 触覚
    }

    valid_count = 0
    for val in df[rank_col]:
        if pd.isna(val): continue
        
        # Google Forms形式: "1位の条件; 2位の条件; 3位の条件"
        # 質問が「やりづらさ(負荷)」順なので、
        # index 0 = 最もやりづらい (Hardest)
        # index 1 = 普通
        # index 2 = 最も楽 (Easiest)
        items = str(val).split(';')
        
        # 空文字除去
        items = [x for x in items if x.strip()]
        
        if len(items) < 3: continue # データ不備スキップ
        valid_count += 1
        
        # 各条件の順位を特定
        ranks = {} # { 'A:提案手法': 0(Hard), ... }
        
        for i, item_name in enumerate(items):
            rank_val = i + 1 # 1, 2, 3
            
            for disp_name, keyword in CONDITIONS.items():
                if keyword in item_name:
                    rank_stats[disp_name][rank_val] += 1
                    ranks[disp_name] = i # 0, 1, 2 (大きい方がEasy)
                    break
        
        # ペアワイズ判定 (Indexが大きい方が「楽＝勝ち」)
        # A vs C
        if 'A:提案手法' in ranks and 'C:聴覚のみ' in ranks:
            if ranks['A:提案手法'] > ranks['C:聴覚のみ']:
                pairwise['A_vs_C']['win'] += 1
            else:
                pairwise['A_vs_C']['lose'] += 1
                
        # A vs B
        if 'A:提案手法' in ranks and 'B:触覚のみ' in ranks:
            if ranks['A:提案手法'] > ranks['B:触覚のみ']:
                pairwise['A_vs_B']['win'] += 1
            else:
                pairwise['A_vs_B']['lose'] += 1

    print(f"有効回答数: {valid_count}")

    # --- グラフ描画 ---
    plt.rcParams['font.family'] = FONT_NAME
    fig = plt.figure(figsize=(12, 10))
    
    # 1. 順位内訳 (100%積み上げ棒グラフ)
    ax1 = fig.add_subplot(2, 2, 1)
    
    # データ整形
    data_list = []
    for name in ORDER_LIST:
        row = rank_stats[name]
        total = sum(row.values())
        if total == 0: total = 1
        # 割合に変換
        data_list.append([
            name, 
            row[1]/total*100, # 1位(Hard)
            row[2]/total*100, # 2位
            row[3]/total*100  # 3位(Easy)
        ])
    
    df_plot = pd.DataFrame(data_list, columns=['Condition', 'Hard(1st)', 'Mid(2nd)', 'Easy(3rd)'])
    
    # 積み上げ描画
    ax1.bar(df_plot['Condition'], df_plot['Hard(1st)'], label='1位:最も大変', color='#ff9999')
    ax1.bar(df_plot['Condition'], df_plot['Mid(2nd)'], bottom=df_plot['Hard(1st)'], label='2位', color='#ffff99')
    ax1.bar(df_plot['Condition'], df_plot['Easy(3rd)'], bottom=df_plot['Hard(1st)']+df_plot['Mid(2nd)'], label='3位:最も楽', color='#99ff99')
    
    ax1.set_title('「やりづらさ」順位の内訳 (分布)', fontsize=14)
    ax1.set_ylabel('割合 (%)')
    ax1.legend(loc='upper right')
    
    # 2. 「最も書きやすかった(楽だった)」条件 (円グラフ)
    # つまり "Easy(3rd)" の数が多い順
    ax2 = fig.add_subplot(2, 2, 2)
    easy_counts = [rank_stats[name][3] for name in ORDER_LIST]
    ax2.pie(easy_counts, labels=ORDER_LIST, autopct='%1.1f%%', startangle=90, 
            colors=['#ff9999', '#66b3ff', '#99ff99'], counterclock=False)
    ax2.set_title('「最も書きやすい」と評価された割合\n(やりづらさ3位の条件)', fontsize=14)

    # 3. 直接対決 (提案 vs 他条件)
    ax3 = fig.add_subplot(2, 1, 2)
    
    comparisons = [
        f"提案 vs 聴覚\n(提案の方が楽: {pairwise['A_vs_C']['win']}人)",
        f"提案 vs 触覚\n(提案の方が楽: {pairwise['A_vs_B']['win']}人)"
    ]
    win_rates = [
        pairwise['A_vs_C']['win'] / valid_count * 100,
        pairwise['A_vs_B']['win'] / valid_count * 100
    ]
    
    bars = ax3.barh(comparisons, win_rates, color=['#66b3ff', '#ffcc99'])
    ax3.set_xlim(0, 100)
    ax3.set_xlabel('提案手法の方が「書きやすい(楽)」と答えた人の割合 (%)')
    ax3.set_title('条件間の直接比較 (勝率)', fontsize=14)
    ax3.axvline(50, color='gray', linestyle='--') # 過半数ライン
    
    # 数値表示
    for bar in bars:
        width = bar.get_width()
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Ranking_Analysis_Detailed.png')
    print("詳細分析グラフを保存しました: Ranking_Analysis_Detailed.png")
    plt.show()

if __name__ == "__main__":
    main()