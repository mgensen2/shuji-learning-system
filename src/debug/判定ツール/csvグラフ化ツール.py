import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 設定 ---
INPUT_CSV = "10_full.csv"  # グラフ化したいファイル名
# -----------

def generate_graphs(csv_file):
    if not os.path.exists(csv_file):
        print(f"エラー: ファイルが見つかりません -> {csv_file}")
        return

    # 1. データの読み込み (ヘッダーあり)
    try:
        df = pd.read_csv(csv_file, header=0)
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        return

    # 必須カラムの確認
    required_cols = ['timestamp', 'x', 'y', 'stroke_id', 'pen_state']
    if not all(col in df.columns for col in required_cols):
        print(f"エラー: CSVの形式が正しくありません。カラムを確認してください: {df.columns}")
        return

    # 2. 描画イベントのみ抽出 (pen_state == 1)
    df_draw = df[df['pen_state'] == 1].copy()
    
    if df_draw.empty:
        print("描画データ(pen_state=1)がありません。")
        return

    # 時間の正規化（開始を0秒にする）
    df_draw['relative_time'] = df_draw['timestamp'] - df_draw['timestamp'].min()

    # 3. 速度の計算 (ストロークごとに計算)
    def calc_velocity_per_stroke(group):
        group = group.sort_values('relative_time')
        dx = group['x'].diff()
        dy = group['y'].diff()
        dt = group['relative_time'].diff()
        dist = np.sqrt(dx**2 + dy**2)
        velocity = dist / dt
        return velocity

    df_draw['velocity'] = df_draw.groupby('stroke_id', group_keys=False).apply(calc_velocity_per_stroke)
    
    # 無効値の処理
    df_draw['velocity'] = df_draw['velocity'].fillna(0).replace([np.inf, -np.inf], 0)

    # --- グラフ1: 軌跡 (X-Y) ---
    plt.figure(figsize=(6, 6))
    for stroke_id, group in df_draw.groupby('stroke_id'):
        plt.plot(group['x'], group['y'], label=f'Stroke {stroke_id}')
        if not group.empty:
            # 書き始めを点で表示
            plt.scatter(group.iloc[0]['x'], group.iloc[0]['y'], s=30, marker='o')
            
    plt.title(f'Trajectory: {csv_file}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis() # 画面座標系に合わせてY軸を反転
    plt.grid(True)
    # plt.legend() # ストロークが多い場合はコメントアウト
    out_traj = "graph_trajectory.png"
    plt.savefig(out_traj)
    print(f"軌跡画像を保存しました: {out_traj}")
    plt.close()

    # --- グラフ2: 速度プロファイル ---
    plt.figure(figsize=(10, 4))
    for stroke_id, group in df_draw.groupby('stroke_id'):
        plt.plot(group['relative_time'], group['velocity'], label=f'Stroke {stroke_id}')
        
    plt.title(f'Velocity Profile: {csv_file}')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.grid(True)
    out_vel = "graph_velocity.png"
    plt.savefig(out_vel)
    print(f"速度グラフを保存しました: {out_vel}")
    plt.close()

if __name__ == "__main__":
    generate_graphs(INPUT_CSV)