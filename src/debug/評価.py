import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 日本語フォント設定（環境に合わせて変更してください）
# !pip install japanize-matplotlib # Google Colab等の場合
plt.rcParams['font.family'] = 'MS Gothic'

# --- 設定：ファイル名（CSVファイル名をここに指定） ---
file_add = '書字評価システム実験追加アンケート(1-6).csv'
file_orig = '書字支援システム評価アンケート(1-7).csv'
file_eval = '第三者評価フォーム(1-7).csv'

# --- 設定：条件の定義（重要！） ---
# ここは実験に合わせて必ず書き換えてください
# 「suffixなし」がどの条件か、「2」がどの条件か、「3」がどの条件か
condition_map = {
    '1': 'A:聴覚のみ',      # 列名に数字がないもの
    '2': 'B:触覚のみ',      # 列名に「2」があるもの
    '3': 'C:提案手法'       # 列名に「3」があるもの
}

# --- 設定：第三者評価の点数変換ルール ---
# エクセル内の言葉を点数に変えます
score_map = {
    '非常にできている': 5,
    'できている': 4,
    'どちらともいえない': 3,
    'あまりできていない': 2,
    '全くできていない': 1,
    # 必要に応じて言葉を追加してください
}

# --- 1. データ読み込みと前処理関数 ---
df_add = pd.read_csv(file_add)
df_eval = pd.read_csv(file_eval)

def clean_score(value):
    """ '5 やや高い' などの数字付きテキストや、日本語評価を数値に変換 """
    if pd.isna(value): return np.nan
    
    # パターン1: 数字が含まれている場合 ('5 やや高い')
    match = re.search(r'\d+', str(value))
    if match:
        return int(match.group())
    
    # パターン2: 日本語のみの場合 ('できている') -> 辞書で変換
    for key, score in score_map.items():
        if key in str(value):
            return score
            
    return np.nan

# --- 2. データの整形（主観評価 NASA-TLX） ---
tlx_items = ['精神', '身体', '切迫', '達成', '努力', '不満'] # キーワード検索
tlx_data = []

for idx, row in df_add.iterrows():
    pid = row['評価ID']
    # 条件1 (suffixなし)
    cols1 = [c for c in df_add.columns if any(k in c for k in tlx_items) and '2' not in c and '3' not in c]
    scores1 = [clean_score(row[c]) for c in cols1]
    
    # 条件2 (suffix 2)
    cols2 = [c for c in df_add.columns if any(k in c for k in tlx_items) and '2' in c]
    scores2 = [clean_score(row[c]) for c in cols2]
    
    # 条件3 (suffix 3)
    cols3 = [c for c in df_add.columns if any(k in c for k in tlx_items) and '3' in c]
    scores3 = [clean_score(row[c]) for c in cols3]

    if scores1: tlx_data.append({'ID': pid, 'Condition': condition_map['1'], 'TLX_Score': np.nanmean(scores1)})
    if scores2: tlx_data.append({'ID': pid, 'Condition': condition_map['2'], 'TLX_Score': np.nanmean(scores2)})
    if scores3: tlx_data.append({'ID': pid, 'Condition': condition_map['3'], 'TLX_Score': np.nanmean(scores3)})

df_tlx = pd.DataFrame(tlx_data)

# --- 3. データの整形（第三者評価） ---
eval_cols = ['字形', '線', '終筆', '総合'] # キーワード
quality_data = []

for idx, row in df_eval.iterrows():
    pid = row['評価ID']
    # 条件1, 2, 3 の抽出（ロジックはTLXと同じ）
    c1 = [clean_score(row[c]) for c in df_eval.columns if any(k in c for k in eval_cols) and '2' not in c and '3' not in c]
    c2 = [clean_score(row[c]) for c in df_eval.columns if any(k in c for k in eval_cols) and '2' in c]
    c3 = [clean_score(row[c]) for c in df_eval.columns if any(k in c for k in eval_cols) and '3' in c]

    if c1: quality_data.append({'ID': pid, 'Condition': condition_map['1'], 'Quality_Score': np.nanmean(c1)})
    if c2: quality_data.append({'ID': pid, 'Condition': condition_map['2'], 'Quality_Score': np.nanmean(c2)})
    if c3: quality_data.append({'ID': pid, 'Condition': condition_map['3'], 'Quality_Score': np.nanmean(c3)})

df_quality = pd.DataFrame(quality_data)

# --- 4. 順位データの集計（ここが追加機能） ---
# 順位の列を探す（「順位」または「並べて」が含まれる列）
rank_col = [c for c in df_add.columns if '順位' in c or '並べて' in c][0]

rank_counts = {condition_map['1']: 0, condition_map['2']: 0, condition_map['3']: 0}

for val in df_add[rank_col]:
    if pd.isna(val): continue
    # "条件A;条件B;条件C" を分解
    items = str(val).split(';')
    if len(items) > 0:
        first_place = items[0] # 1位の条件名を取得（フォームの選択肢の文字列）
        
        # フォームの選択肢名と、condition_mapの紐付けが難しいので
        # ここは簡易的に名前の一致を見ます。
        # 実際には「提案システム」などの文字列がそのまま入っているはずです。
        # グラフ用の辞書にカウント
        if first_place in rank_counts:
            rank_counts[first_place] += 1
        else:
            # マッピングが完全でない場合、そのままキーとして追加
            rank_counts[first_place] = rank_counts.get(first_place, 0) + 1

# --- 5. グラフ描画 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) NASA-TLX
sns.barplot(x='Condition', y='TLX_Score', data=df_tlx, ax=axes[0], capsize=.1, errorbar='se', palette='Blues')
axes[0].set_title('主観的負荷 (低いほど良い)')
axes[0].set_ylabel('NASA-TLX Score')

# (2) 第三者評価
sns.barplot(x='Condition', y='Quality_Score', data=df_quality, ax=axes[1], capsize=.1, errorbar='se', palette='Greens')
axes[1].set_title('第三者評価 (高いほど良い)')
axes[1].set_ylabel('Quality Score (1-5)')

# (3) 順位（1位に選ばれた回数）
# rank_countsのキーと値をリスト化
labels = list(rank_counts.keys())
values = list(rank_counts.values())
axes[2].pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Pastel1'))
axes[2].set_title('「最も書きやすかった」と選ばれた割合')

plt.tight_layout()
plt.show()