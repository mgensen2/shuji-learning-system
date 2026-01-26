import pandas as pd
import matplotlib.pyplot as plt
import platform

# 1. CSVファイルの読み込み
file_path = '書字評価システム実験追加アンケート(1-10).csv'
df = pd.read_csv(file_path)

# 2. 日本語フォントの設定
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.family'] = 'MS Gothic'
elif system_name == 'Darwin': # Mac
    plt.rcParams['font.family'] = 'Hiragino Sans'
else:
    plt.rcParams['font.family'] = 'sans-serif'

# 3. 文字サイズを全体的に大きく設定
plt.rcParams['font.size'] = 16  # 基本の文字サイズ

# 4. データ抽出処理
target_col = '「作業のやりづらさ（負荷の高さ）」を感じた順に並べてください。 （最もやりづらかったものを1位、最も楽だったものを3位としてください）'

def get_most_writable(text):
    if pd.isna(text):
        return None
    items = [x for x in str(text).split(';') if x.strip() != '']
    if not items:
        return None
    return items[-1]

df['most_writable'] = df[target_col].apply(get_most_writable)
counts = df['most_writable'].value_counts()

# 5. 色の設定（前回のグラフと同じ色合いに固定）
# 聴覚のみ＝青、提案手法＝オレンジ、触覚のみ＝緑
color_map = {
    '聴覚のみ': '#455681',    # 青
    '提案手法': "#6cbc6e",  # オレンジ
    '触覚のみ': '#2f837f'     # 緑
}
# データ順に合わせて色リストを作成
colors = [color_map.get(label, 'gray') for label in counts.index]

# 6. 円グラフの描画
plt.figure(figsize=(10, 10))  # グラフサイズも少し大きく

plt.pie(counts, 
        labels=counts.index, 
        colors=colors,           # 色を指定
        autopct='%1.1f%%',       # パーセンテージ表示
        startangle=90, 
        counterclock=False,
        textprops={'fontsize': 18, 'weight': 'bold'}, # ラベルと％の文字を大きく・太く
        pctdistance=0.6)         # ％の表示位置調整

plt.title('最も書きやすかった条件の割合', fontsize=24, pad=20) # タイトルを大きく
plt.axis('equal')

# 保存と表示
plt.savefig('most_writable_pie_chart_large.png')
plt.show()