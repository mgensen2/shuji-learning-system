import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# --- 1. フォント設定（エラー回避版） ---
# Windowsなら 'MS Gothic', Macなら 'Hiragino Sans' などを指定
# ここでは標準的な設定にします
plt.rcParams['font.family'] = 'MS Gothic' 

# --- 2. データ読み込み ---
file_add = '書字評価システム実験追加アンケート(1-10).csv'
file_eval = '第三者評価フォーム(1-11).csv'

# 文字コードの自動判定読み込み
try:
    df_add = pd.read_csv(file_add, encoding='utf-8')
    df_eval = pd.read_csv(file_eval, encoding='utf-8')
except UnicodeDecodeError:
    df_add = pd.read_csv(file_add, encoding='shift_jis')
    df_eval = pd.read_csv(file_eval, encoding='shift_jis')

# --- 3. 条件マッピング（確定版） ---
# 1(suffixなし)=聴覚, 2=触覚, 3=提案
condition_map = {
    '1': 'A:聴覚のみ',
    '2': 'B:触覚のみ',
    '3': 'C:提案手法'
}

# --- 4. 前処理関数 ---
def clean_score(value):
    if pd.isna(value): return np.nan
    match = re.search(r'\d+', str(value))
    if match: return int(match.group())
    # 日本語評価の数値化
    score_dict = {'できている': 5, '概ね': 4, 'どちらとも': 3, 'あまり': 2, '全く': 1}
    for k, v in score_dict.items():
        if k in str(value): return v
    return np.nan

# --- 5. 集計処理 (TLX & Quality) ---
tlx_items = ['精神', '身体', '切迫', '達成', '努力', '不満']
eval_items = ['字形', '線', '終筆', '総合']

data_summary = []

# IDリストの取得（両方のファイルにあるIDを対象にする）
all_ids = set(df_add['評価ID']).union(set(df_eval['評価ID']))

for pid in all_ids:
    row_add = df_add[df_add['評価ID'] == pid].iloc[0] if not df_add[df_add['評価ID'] == pid].empty else None
    row_eval = df_eval[df_eval['評価ID'] == pid].iloc[0] if not df_eval[df_eval['評価ID'] == pid].empty else None
    
    # 3つの条件についてループ
    for suffix, cond_name in condition_map.items():
        # TLXスコア計算
        tlx_score = np.nan
        if row_add is not None:
            if suffix == '1':
                cols = [c for c in df_add.columns if any(k in c for k in tlx_items) and '2' not in c and '3' not in c]
            else:
                cols = [c for c in df_add.columns if any(k in c for k in tlx_items) and suffix in c]
            scores = [clean_score(row_add[c]) for c in cols]
            tlx_score = np.nanmean(scores) if scores else np.nan
            
        # 品質スコア計算
        qual_score = np.nan
        if row_eval is not None:
            if suffix == '1':
                cols = [c for c in df_eval.columns if any(k in c for k in eval_items) and '2' not in c and '3' not in c]
            else:
                cols = [c for c in df_eval.columns if any(k in c for k in eval_items) and suffix in c]
            scores = [clean_score(row_eval[c]) for c in cols]
            qual_score = np.nanmean(scores) if scores else np.nan
            
        if not np.isnan(tlx_score) or not np.isnan(qual_score):
            data_summary.append({
                'ID': pid,
                'Condition': cond_name,
                'TLX': tlx_score,
                'Quality': qual_score
            })

df_final = pd.DataFrame(data_summary)

# --- 6. グラフ描画 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) 負荷 (TLX)
sns.barplot(data=df_final, x='Condition', y='TLX', ax=axes[0], palette='Blues', capsize=.1, errorbar='se')
axes[0].set_title('主観的負荷 (低い方が楽)')
axes[0].set_ylabel('NASA-TLX Score')
axes[0].set_ylim(0, 10) # 0-10評価に変換されている場合。0-100なら適宜変更

# (2) 品質 (Quality)
sns.barplot(data=df_final, x='Condition', y='Quality', ax=axes[1], palette='Greens', capsize=.1, errorbar='se')
axes[1].set_title('書字品質 (高い方が良い)')
axes[1].set_ylabel('Quality Score (1-5)')
axes[1].set_ylim(1, 5)

# (3) 散布図 (トレードオフ確認用)
mean_vals = df_final.groupby('Condition')[['TLX', 'Quality']].mean().reset_index()
sns.scatterplot(data=mean_vals, x='TLX', y='Quality', hue='Condition', s=500, ax=axes[2], palette='Set2')
for i, row in mean_vals.iterrows():
    axes[2].text(row['TLX'], row['Quality']+0.1, row['Condition'], ha='center', fontsize=12, fontweight='bold')
axes[2].set_title('負荷 vs 品質 (右上ほど「大変だが高品質」)')
axes[2].set_xlabel('主観的負荷 (TLX)')
axes[2].set_ylabel('品質 (Quality)')
axes[2].grid(True, linestyle='--')

plt.tight_layout()
plt.show()

# 数値の出力
print("--- 条件別 平均スコア ---")
print(mean_vals)