import matplotlib.pyplot as plt
import numpy as np

# --- 設定エリア ---
plt.rcParams['font.family'] = 'MS Gothic' # 環境に合わせて変更

# 条件
conditions = ['聴覚のみ (C)', '触覚のみ (B)', '提案手法 (A)']

# データ (100点満点換算)
scores_expert = [93.3, 81.7, 73.7]  # 経験者
scores_novice = [43.3, 57.5, 61.7]  # 初学者

# --- 描画 ---
x = np.arange(len(conditions))
fig, ax = plt.subplots(figsize=(7, 5))

# プロット
ax.plot(x, scores_expert, marker='o', markersize=8, label='経験者群', 
        color='#2c3e50', linewidth=2, linestyle='--')
ax.plot(x, scores_novice, marker='s', markersize=8, label='初学者群', 
        color='#e74c3c', linewidth=2, linestyle='-')

# 数値表示
for i, txt in enumerate(scores_expert):
    ax.annotate(f'{txt}', (x[i], scores_expert[i]), 
                textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
for i, txt in enumerate(scores_novice):
    ax.annotate(f'{txt}', (x[i], scores_novice[i]), 
                textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)

# ★ここを変更しました
ax.set_ylabel('平均評価スコア (100点満点換算)')
ax.set_title('書道経験と提示条件の交互作用')

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylim(0, 105)
ax.legend(loc='best')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()