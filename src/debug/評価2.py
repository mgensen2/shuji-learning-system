import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os # ファイル存在確認用

# --- 1. フォント設定 ---
# システムに合わせて適切な日本語フォントを指定してください
plt.rcParams['font.family'] = 'MS Gothic' # Windows用例
# plt.rcParams['font.family'] = 'Hiragino Sans' # Mac用例
# plt.rcParams['font.family'] = 'Noto Sans CJK JP' # Linux/Google Colab用例

# --- 2. データ読み込み ---
file_add = '書字評価システム実験追加アンケート(1-10).csv'
file_eval = '第三者評価フォーム(1-11).csv'

# ファイルの存在確認（エラーハンドリング追加）
if not os.path.exists(file_add) or not os.path.exists(file_eval):
    print(f"エラー: ファイルが見つかりません。")
    print(f"  - {file_add}: {'存在します' if os.path.exists(file_add) else '見つかりません'}")
    print(f"  - {file_eval}: {'存在します' if os.path.exists(file_eval) else '見つかりません'}")
    print("ファイル名やパスを確認してください。")
    exit()

# 文字コードの自動判定読み込み
try:
    df_add = pd.read_csv(file_add, encoding='utf-8')
    df_eval = pd.read_csv(file_eval, encoding='utf-8')
except UnicodeDecodeError:
    print("utf-8での読み込みに失敗しました。shift_jisで試行します。")
    df_add = pd.read_csv(file_add, encoding='shift_jis')
    df_eval = pd.read_csv(file_eval, encoding='shift_jis')
except Exception as e:
    print(f"ファイルの読み込み中にエラーが発生しました: {e}")
    exit()

# --- 3. 条件マッピング ---
# 1(suffixなし)=聴覚, 2=触覚, 3=提案
condition_map = {
    '1': 'C:聴覚のみ',
    '2': 'B:触覚のみ',
    '3': 'A:提案手法'
}

# --- 4. 前処理関数 ---
def clean_score(value):
    if pd.isna(value): return np.nan
    # 数値が含まれていれば抽出
    match = re.search(r'\d+', str(value))
    if match: return int(match.group())
    # 日本語評価の数値化 (第三者評価フォーム用)
    score_dict = {'できている': 5, '概ね': 4, 'どちらとも': 3, 'あまり': 2, '全く': 1}
    val_str = str(value)
    for k, v in score_dict.items():
        if k in val_str: return v
    return np.nan

# --- 5. 集計処理 (TLX & Quality) ---
tlx_items = ['精神', '身体', '切迫', '達成', '努力', '不満']
eval_items = ['字形', '線', '終筆', '総合']

data_summary = []

# IDリストの取得（両方のファイルにあるIDを対象にする）
if '評価ID' not in df_add.columns or '評価ID' not in df_eval.columns:
    print("エラー: CSVファイルに '評価ID' 列が見つかりません。列名を確認してください。")
    exit()

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
            
            scores = [clean_score(row_add[c]) for c in cols if c in row_add]
            # Noneを除外してから平均を計算
            valid_scores = [s for s in scores if pd.notna(s)]
            tlx_score = np.mean(valid_scores) if valid_scores else np.nan
            
        # 品質スコア計算
        qual_score = np.nan
        if row_eval is not None:
            if suffix == '1':
                cols = [c for c in df_eval.columns if any(k in c for k in eval_items) and '2' not in c and '3' not in c]
            else:
                cols = [c for c in df_eval.columns if any(k in c for k in eval_items) and suffix in c]
            
            scores = [clean_score(row_eval[c]) for c in cols if c in row_eval]
            # Noneを除外してから平均を計算
            valid_scores = [s for s in scores if pd.notna(s)]
            qual_score = np.mean(valid_scores) if valid_scores else np.nan
            
        if not np.isnan(tlx_score) or not np.isnan(qual_score):
            data_summary.append({
                'ID': pid,
                'Condition': cond_name,
                'TLX': tlx_score,
                'Quality': qual_score
            })

if not data_summary:
    print("エラー: 集計できるデータがありませんでした。列名やデータの値を確認してください。")
    exit()

df_final = pd.DataFrame(data_summary)

# --- 6. グラフ描画 ---

# ========================== 色設定エリア ==========================
# ここで好きな色を指定してください。
# 色の名前（'red', 'blue'など）、16進数カラーコード（'#FF0000'など）のリスト、
# またはSeabornのパレット名（'deep', 'pastel', 'Set2'など）が使えます。
# リストの場合、条件の順番（A:提案, B:触覚, C:聴覚 の順でソートされます）に対応します。

# 例1: 具体的な色のリストで指定 (自由度が高い)
# tlx_colors = ['#ff9999', '#66b3ff', '#99ff99']
# quality_colors = ['salmon', 'skyblue', 'lightgreen']
# scatter_colors = ['red', 'blue', 'green']

# 例2: Seabornのカラーパレット名で指定 (簡単)
tlx_colors =    ['#6cbc6e','#2f837f','#455681'] # 青系のグラデーション (末尾の_dは濃いめ)
quality_colors = ['#6cbc6e','#2f837f','#455681'] # 緑系のグラデーション
scatter_colors = ['#6cbc6e','#2f837f','#455681']      # 標準的なカラーパレット

# ===================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 条件の表示順序を固定する（A->B->C の順）
order = sorted(df_final['Condition'].unique())

# (1) 負荷 (TLX)
sns.barplot(data=df_final, x='Condition', y='TLX', ax=axes[0], 
            palette=tlx_colors, order=order, capsize=.1, errorbar='se')
axes[0].set_title('主観的負荷 (低い方が楽)', fontsize=14)
axes[0].set_ylabel('NASA-TLX Score', fontsize=12)
axes[0].set_ylim(0, 10) # 0-10評価の場合。必要に応じて変更してください
axes[0].tick_params(axis='x', rotation=15) # ラベルが重なる場合は角度をつける

# (2) 品質 (Quality)
sns.barplot(data=df_final, x='Condition', y='Quality', ax=axes[1], 
            palette=quality_colors, order=order, capsize=.1, errorbar='se')
axes[1].set_title('書字品質 (高い方が良い)', fontsize=14)
axes[1].set_ylabel('Quality Score (1-5)', fontsize=12)
axes[1].set_ylim(1, 5.2) # 上限を少し空ける
axes[1].tick_params(axis='x', rotation=15)

# (3) 散布図 (トレードオフ確認用)
mean_vals = df_final.groupby('Condition')[['TLX', 'Quality']].mean().reset_index()
# 散布図のマーカーサイズを調整
sns.scatterplot(data=mean_vals, x='TLX', y='Quality', hue='Condition', 
                s=300, ax=axes[2], palette=scatter_colors, hue_order=order, style='Condition')

# テキストラベルの追加（重なりを防ぐため少し調整）
for i, row in mean_vals.iterrows():
    axes[2].text(row['TLX'], row['Quality'] + 0.08, row['Condition'].split(':')[0], 
                 ha='center', va='bottom', fontsize=12, fontweight='bold', 
                 color=sns.color_palette(scatter_colors, n_colors=len(order))[order.index(row['Condition'])])

axes[2].set_title('負荷 vs 品質 (右上ほど「大変だが高品質」)', fontsize=14)
axes[2].set_xlabel('主観的負荷 (TLX)', fontsize=12)
axes[2].set_ylabel('品質 (Quality)', fontsize=12)
axes[2].grid(True, linestyle='--', alpha=0.7)
# 凡例をグラフの外側に配置
axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()

# 数値の出力
print("--- 条件別 平均スコア ---")
print(mean_vals.set_index('Condition'))