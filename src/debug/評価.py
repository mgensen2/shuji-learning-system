import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# --- 設定 ---
FONT_NAME = 'MS Gothic' 
plt.rcParams['font.family'] = FONT_NAME

FILE_ADD = '書字評価システム実験追加アンケート(1-6).csv'
FILE_EVAL = '第三者評価フォーム(1-7).csv'

# 条件マッピング
CONDITION_MAP = {
    '1': 'A:聴覚のみ',      
    '2': 'B:触覚のみ',      
    '3': 'C:提案手法'       
}
ORDER_LIST = sorted(list(CONDITION_MAP.values()))

# 点数変換 (日本語 -> 1~5)
SCORE_MAP = {
    '非常にできている': 5,
    '概ねできている': 4,
    'できている': 4,
    'どちらともいえない': 3,
    'あまりできていない': 2,
    '全くできていない': 1,
    '高い': 5,          # NASA-TLX用に追加
    'やや高い': 4,
    '中程度': 3,
    'やや低い': 2,
    '低い': 1
}

def load_data(path):
    if not os.path.exists(path):
        # 拡張子違い対応
        base, _ = os.path.splitext(path)
        if os.path.exists(base + '.xlsx'): return pd.read_excel(base + '.xlsx')
        return None
    try:
        return pd.read_csv(path)
    except:
        return pd.read_csv(path, encoding='shift_jis')

def clean_score(value):
    """テキストを1-5の数値に変換"""
    if pd.isna(value): return np.nan
    s_val = str(value)
    
    # 辞書にある言葉が含まれていれば変換
    for key, score in SCORE_MAP.items():
        if key in s_val:
            return score
            
    # 数字が含まれていれば抽出 (例: "5 やや高い" -> 5)
    match = re.search(r'\d+', s_val)
    if match:
        # もし元のデータが0-100や0-10系なら、ここで1-5に正規化が必要
        # 今回はデータを見る限り "1 低い" ~ "5 高い" のようなのでそのまま
        val = int(match.group())
        # 万が一 6以上の値が来たら5に丸める等の処理も可能
        return val
        
    return np.nan

def main():
    df_add = load_data(FILE_ADD)
    df_eval = load_data(FILE_EVAL)
    if df_add is None or df_eval is None: return

    # --- 1. NASA-TLX 集計 ---
    tlx_items = ['精神', '身体', '切迫', '達成', '努力', '不満']
    tlx_data = []

    for idx, row in df_add.iterrows():
        # 条件1
        c1 = [c for c in df_add.columns if any(k in c for k in tlx_items) and '2' not in c and '3' not in c]
        s1 = [clean_score(row[c]) for c in c1]
        if s1: tlx_data.append({'Condition': CONDITION_MAP['1'], 'Score': np.nanmean(s1)})
        
        # 条件2
        c2 = [c for c in df_add.columns if any(k in c for k in tlx_items) and '2' in c]
        s2 = [clean_score(row[c]) for c in c2]
        if s2: tlx_data.append({'Condition': CONDITION_MAP['2'], 'Score': np.nanmean(s2)})
        
        # 条件3
        c3 = [c for c in df_add.columns if any(k in c for k in tlx_items) and '3' in c]
        s3 = [clean_score(row[c]) for c in c3]
        if s3: tlx_data.append({'Condition': CONDITION_MAP['3'], 'Score': np.nanmean(s3)})

    df_tlx = pd.DataFrame(tlx_data)

    # --- 2. 第三者評価 集計 ---
    eval_items = ['字形', '線', '終筆', '総合']
    qual_data = []

    for idx, row in df_eval.iterrows():
        c1 = [c for c in df_eval.columns if any(k in c for k in eval_items) and '2' not in c and '3' not in c]
        s1 = [clean_score(row[c]) for c in c1]
        if s1: qual_data.append({'Condition': CONDITION_MAP['1'], 'Score': np.nanmean(s1)})

        c2 = [c for c in df_eval.columns if any(k in c for k in eval_items) and '2' in c]
        s2 = [clean_score(row[c]) for c in c2]
        if s2: qual_data.append({'Condition': CONDITION_MAP['2'], 'Score': np.nanmean(s2)})

        c3 = [c for c in df_eval.columns if any(k in c for k in eval_items) and '3' in c]
        s3 = [clean_score(row[c]) for c in c3]
        if s3: qual_data.append({'Condition': CONDITION_MAP['3'], 'Score': np.nanmean(s3)})

    df_quality = pd.DataFrame(qual_data)

    # --- 3. 順位 集計 ---
    rank_col = [c for c in df_add.columns if '順位' in c or '並べて' in c]
    rank_counts = {v: 0 for v in CONDITION_MAP.values()}
    
    if rank_col:
        col_name = rank_col[0]
        for val in df_add[col_name]:
            if pd.isna(val): continue
            first = str(val).split(';')[0]
            # マッチング
            for label in CONDITION_MAP.values():
                # "聴覚" などのキーワードが含まれていればカウント
                keyword = label.split(':')[1]
                if keyword in first or label in first:
                    rank_counts[label] += 1
                # CSV独自の表記対応 ("提案システム"など)
                elif "提案" in first and "提案" in label: rank_counts[label] += 1
                elif "触覚" in first and "触覚" in label: rank_counts[label] += 1
                elif "聴覚" in first and "聴覚" in label: rank_counts[label] += 1

    # --- 4. 描画 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # (1) TLX
    sns.barplot(x='Condition', y='Score', data=df_tlx, ax=axes[0], 
                order=ORDER_LIST, palette='Blues', capsize=.1, errorbar='se')
    axes[0].set_title('主観的負荷 (NASA-TLX)\n低いほど良い (1-5)', fontsize=14)
    axes[0].set_ylabel('Score (1-5)', fontsize=12)
    axes[0].set_ylim(1, 5) # 1-5に変更
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # (2) Quality
    sns.barplot(x='Condition', y='Score', data=df_quality, ax=axes[1], 
                order=ORDER_LIST, palette='Greens', capsize=.1, errorbar='se')
    axes[1].set_title('第三者評価 (書字品質)\n高いほど良い (1-5)', fontsize=14)
    axes[1].set_ylabel('Score (1-5)', fontsize=12)
    axes[1].set_ylim(1, 5) # 1-5に変更
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    # (3) Rank
    labels = list(rank_counts.keys())
    values = list(rank_counts.values())
    colors = sns.color_palette('Pastel1', len(labels))
    
    axes[2].pie(values, labels=labels, autopct='%1.1f%%', startangle=90, 
                colors=colors, textprops={'fontsize': 12})
    axes[2].set_title('「最も書きやすかった」条件', fontsize=14)

    plt.savefig('Survey_Analysis_Graph_v2.png', dpi=300)
    print("グラフを保存しました: Survey_Analysis_Graph_v2.png")
    plt.show()

if __name__ == "__main__":
    main()