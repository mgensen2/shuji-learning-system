import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os

SAMPLE_CSV = "sample.csv"
USER_CSV = "normalized_data.csv" # Step2で正規化したものを使う想定

def load_points_and_velocity(filepath):
    if not os.path.exists(filepath): return None, None
    df = pd.read_csv(filepath, header=0)
    
    # 描画中のみ抽出
    df_draw = df[df['pen_state'] == 1].copy()
    
    # 座標点列
    points = df_draw[['x', 'y']].values
    
    # 速度計算
    df_draw['time_diff'] = df_draw['timestamp'].diff()
    df_draw['dist'] = np.sqrt(df_draw['x'].diff()**2 + df_draw['y'].diff()**2)
    df_draw['velocity'] = df_draw['dist'] / df_draw['time_diff']
    df_draw['velocity'] = df_draw['velocity'].fillna(0).replace([np.inf, -np.inf], 0)
    
    velocity = df_draw['velocity'].values
    return points, velocity

def calculate_dtw(seq1, seq2):
    if len(seq1) == 0 or len(seq2) == 0: return float('inf')
    d = cdist(seq1, seq2, metric='euclidean')
    n, m = d.shape
    cost = np.zeros((n, m))
    cost[0, 0] = d[0, 0]
    for i in range(1, n): cost[i, 0] = cost[i-1, 0] + d[i, 0]
    for j in range(1, m): cost[0, j] = cost[0, j-1] + d[0, j]
    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = d[i, j] + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    return cost[-1, -1] / (n + m)

def main():
    pts_s, vel_s = load_points_and_velocity(SAMPLE_CSV)
    pts_u, vel_u = load_points_and_velocity(USER_CSV)
    
    if pts_s is None or pts_u is None:
        print("Error loading data.")
        return

    # 1. DTW (軌跡の形状類似度)
    dtw_score = calculate_dtw(pts_s, pts_u)
    print(f"Trajectory DTW Score: {dtw_score:.4f} (Lower is better)")

    # 2. 速度相関 (リズムの類似度) - 簡易的にサイズを合わせて相関を見る例
    # 実際は速度波形に対してもDTWをかけるのがベストですが、ここでは基本統計で比較
    avg_v_s = np.mean(vel_s)
    avg_v_u = np.mean(vel_u)
    print(f"Avg Velocity: Sample={avg_v_s:.2f}, User={avg_v_u:.2f}")

if __name__ == "__main__":
    main()