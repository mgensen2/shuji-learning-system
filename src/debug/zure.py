import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# ==========================================
# 1. 日本語フォント設定
# ==========================================
def configure_japanese_font():
    # japanize_matplotlibがあれば利用
    try:
        import japanize_matplotlib
        return
    except ImportError:
        pass

    # 環境に応じた標準的な日本語フォントの自動選択
    system = platform.system()
    target_fonts = []
    if system == 'Windows':
        target_fonts = ['Meiryo', 'MS Gothic', 'Yu Gothic']
    elif system == 'Darwin': # Mac
        target_fonts = ['Hiragino Sans', 'AppleGothic']
    else: # Linux / Google Colab
        target_fonts = ['Noto Sans CJK JP', 'TakaoGothic', 'IPAGothic']
    
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font in target_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            return

configure_japanese_font()

# ==========================================
# 2. ユーザー指定の設定エリア
# ==========================================
# 条件ごとの色設定
color_map = {
    'A': '#6cbc6e',  # 提案手法 (緑系)
    'B': '#2f837f',  # 触覚のみ (青紫系)
    'C': '#455681'   # 聴覚のみ (青紫系)
}

# 条件ごとの表示名設定
label_map = {
    'A': '提案手法',
    'B': '触覚のみ',
    'C': '聴覚のみ'
}

# ==========================================
# 3. データの読み込みと集計
# ==========================================
df = pd.read_csv('IoU_result.csv')

# 条件ごとの集計
df_condition = df.groupby('Condition')['Shift_Centroid_Dist'].mean().reindex(['A', 'B', 'C'])

# 文字ごとの集計
df_char = df.groupby('Correct_Char')['Shift_Centroid_Dist'].mean()

# グラフ描画用の設定適用
plot_labels = [label_map[cond] for cond in df_condition.index]
plot_colors = [color_map[cond] for cond in df_condition.index]

# ==========================================
# 4. グラフの描画
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (1) 条件ごとの棒グラフ
axes[0].bar(plot_labels, df_condition.values, color=plot_colors, zorder=3)
axes[0].set_title('条件ごとの重心のズレ')
axes[0].set_xlabel('条件')
axes[0].set_ylabel('Shift_Centroid_Dist 平均値')
axes[0].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# (2) 文字ごとの棒グラフ
axes[1].bar(df_char.index, df_char.values, color='orange', zorder=3)
axes[1].set_title('文字ごとの重心のズレ')
axes[1].set_xlabel('文字')
axes[1].set_ylabel('Shift_Centroid_Dist 平均値')
axes[1].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

plt.tight_layout()
plt.savefig('Shift_Centroid_Dist_Customized.png')
plt.show()