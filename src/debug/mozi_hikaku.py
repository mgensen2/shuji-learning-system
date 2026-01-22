import matplotlib.pyplot as plt
import numpy as np

# --- 設定エリア ---
plt.rcParams['font.family'] = 'MS Gothic'

# データ定義
# 評価対象の技法と文字
labels = ['はね (刁)', 'はらい (爻)', 'とめ (乎)']

# 生のスコアと満点
raw_scores = [1.4, 2.9, 3.1]
max_scores = [2.0, 4.0, 5.0]

# 正規化 (100点満点換算)
normalized_scores = [(s / m) * 100 for s, m in zip(raw_scores, max_scores)]

# --- 描画 ---
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7, 5))

# 棒グラフ (色分け: グレー系で統一しつつ濃淡をつける)
rects = ax.bar(x, normalized_scores, width=0.5, 
               color=['#95a5a6', '#2c3e50', '#7f8c8d'])

# 数値表示
for rect in rects:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11)

# 装飾
ax.set_ylabel('平均評価スコア (100点満点換算)')
ax.set_title('技法ごとの習熟度比較 (全条件平均)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()