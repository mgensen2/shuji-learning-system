import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import platform

# ---------------------------------------------------------
# 1. 設定：日本語フォントの指定
# ---------------------------------------------------------
# お使いの環境に合わせてフォントファミリーを指定してください。
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.family'] = 'MS Gothic' # Windowsの標準フォント
elif system_name == 'Darwin': # Mac
    plt.rcParams['font.family'] = 'Hiragino Sans' # Macの標準フォント
else:
    # LinuxやGoogle Colabなど。
    # 日本語フォントがインストールされていない場合は文字化けします。
    # その場合は 'Japanize-matplotlib' 等のライブラリ利用を検討してください。
    plt.rcParams['font.family'] = 'IPAGothic' 

# ---------------------------------------------------------
# 2. データの読み込み
# ---------------------------------------------------------
# CSVファイル名を指定してください
file_path = "書字評価システム実験追加アンケート(1-10).csv"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# 3. データの前処理 (NASA-TLX)
# ---------------------------------------------------------
# 数値を取り出す関数 (例: "7 非常に高い" -> 7)
def extract_score(val):
    if pd.isna(val):
        return np.nan
    match = re.match(r'(\d+)', str(val))
    if match:
        return int(match.group(1))
    return np.nan

# 列の定義 (CSVの列順序が固定である前提)
# 聴覚: 列8-13, 触覚: 列14-19, 提案: 列20-25
tlx_cols_map = [
    ('聴覚', df.columns[8:14]),
    ('触覚', df.columns[14:20]),
    ('提案手法', df.columns[20:26])
]

subscales_jp = ['精神的負担', '身体的負担', '時間的切迫感', '作業達成感', '努力', '不満']

tlx_data = []

for cond_name, cols in tlx_cols_map:
    for idx, col in enumerate(cols):
        scores = df[col].apply(extract_score)
        for s in scores:
            tlx_data.append({
                'Condition': cond_name,
                'Subscale': subscales_jp[idx],
                'Score': s
            })

tlx_df = pd.DataFrame(tlx_data)

# ---------------------------------------------------------
# 4. データの前処理 (作業のやりやすさ)
# ---------------------------------------------------------
# 質問：「作業のやりづらさ」順 (1位=一番やりづらい, 3位=一番楽)
# グラフ化：「やりやすさ」 (1点=やりづらい, 3点=やりやすい) に変換

ease_scores = {'聴覚': [], '触覚': [], '提案手法': []}
ranking_col = df.columns[7]

for val in df[ranking_col]:
    if pd.isna(val):
        continue
    # "聴覚のみ; 提案システム; ..." となっているので分解
    items = [x.strip() for x in val.split(';') if x.strip()]
    
    for rank, item_raw in enumerate(items):
        # 名前を統一
        if '聴覚' in item_raw: name = '聴覚'
        elif '触覚' in item_raw: name = '触覚'
        elif '提案' in item_raw: name = '提案手法'
        else: continue
            
        # rank 0 (1位:やりづらい) -> Score 1
        # rank 1 (2位) -> Score 2
        # rank 2 (3位:やりやすい) -> Score 3
        score = rank + 1
        ease_scores[name].append(score)

ease_data = []
for name, scores in ease_scores.items():
    for s in scores:
        ease_data.append({'Condition': name, 'Score': s})
ease_df = pd.DataFrame(ease_data)

# ---------------------------------------------------------
# 5. グラフの描画
# ---------------------------------------------------------
sns.set(style="whitegrid") 
# seabornのスタイル設定でフォントが上書きされることがあるため、再度設定
if system_name == 'Windows':
    plt.rcParams['font.family'] = 'MS Gothic'
elif system_name == 'Darwin':
    plt.rcParams['font.family'] = 'Hiragino Sans'

# 表示順序
order_cond = ['聴覚', '触覚', '提案手法']
order_subscale = subscales_jp

# --- グラフ1: NASA-TLX ---
plt.figure(figsize=(10, 6))
sns.barplot(
    data=tlx_df, 
    x='Subscale', 
    y='Score', 
    hue='Condition', 
    hue_order=order_cond,
    order=order_subscale,
    capsize=0.1,
    palette='viridis',
    errorbar='se' # 標準誤差を表示
)
plt.title('NASA-TLX 集計結果')
plt.ylabel('スコア (1:低い - 7:高い)')
plt.xlabel('評価項目')
plt.ylim(0, 7.5)
plt.legend(title='条件')
plt.tight_layout()
plt.savefig('NASA_TLX_Result.png', dpi=300)
plt.show()

# --- グラフ2: 作業のやりやすさ ---
plt.figure(figsize=(6, 6))
sns.barplot(
    data=ease_df, 
    x='Condition', 
    y='Score', 
    order=order_cond,
    capsize=0.1,
    palette='viridis',
    errorbar='se'
)
plt.title('作業のやりやすさ (高いほどやりやすい)')
plt.ylabel('スコア (1:やりづらい - 3:やりやすい)')
plt.xlabel('条件')
plt.ylim(0, 3.5)
plt.tight_layout()
plt.savefig('Ease_of_Work_Result.png', dpi=300)
plt.show()