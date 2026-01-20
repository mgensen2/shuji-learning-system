import matplotlib.pyplot as plt
import numpy as np

# --- 設定エリア ---
# 環境に合わせてフォントを指定してください
plt.rcParams['font.family'] = 'MS Gothic' 

# データ定義
techniques = ['はね (刁)', 'はらい (爻)', 'とめ (乎)']
conditions = ['聴覚のみ (C)', '触覚のみ (B)', '提案手法 (A)']
colors = ['#455681', '#2f837f', '#6cbc6e'] # グレー2色と緑

# 修正後のスコアデータ (1.0点満点)
# A: 提案, B: 触覚, C: 聴覚
scores_A = [1.0,  0.8,  0.6] # ★ここを0.8に修正しました
scores_B = [1.0,  0.7,  0.6]
scores_C = [0.75, 0.7, 0.4]

# -----------------

x = np.arange(len(techniques))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))

rects1 = ax.bar(x - width, scores_C, width, label='聴覚のみ (C)', color=colors[0])
rects2 = ax.bar(x,         scores_B, width, label='触覚のみ (B)', color=colors[1])
rects3 = ax.bar(x + width, scores_A, width, label='提案手法 (A)', color=colors[2])

ax.set_ylabel('平均評価スコア (Max: 1.0)')
ax.set_title('技法別評価の結果')
ax.set_xticks(x)
ax.set_xticklabels(techniques)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()