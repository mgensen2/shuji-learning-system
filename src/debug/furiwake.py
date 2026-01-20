import pandas as pd
import os
import shutil
import random

# ==========================================
# 設定
# ==========================================
source_root = './実験データ' 
output_folder = './selected_images'
img_extension = '.png'
RANDOM_SEED = 41

# ---------------------------------------------------------
# 【追加設定】除外したいIDのリスト
# ブランクが長い、経験年数が短いなどの理由で除外するIDを指定
# ---------------------------------------------------------
EXCLUDE_IDS = [9] 

# ==========================================
# 1. データの読み込みと分類
# ==========================================
try:
    df_survey = pd.read_csv('書道経験アンケート(1-10).csv')
    df_progress = pd.read_csv('実験進捗シート.csv')
except FileNotFoundError as e:
    print(f"エラー: CSVファイルが見つかりません。\n{e}")
    exit()

# 除外IDの行をアンケートデータから削除しておく
df_survey = df_survey[~df_survey['ID'].isin(EXCLUDE_IDS)]

# 経験者と初学者のIDリストを作成
exp_ids = df_survey[df_survey['これまでに、学校の授業（図画工作・国語・書写）以外で、毛筆による書道を習ったことはありますか？'] == 'はい']['ID'].tolist()
beg_ids = df_survey[df_survey['これまでに、学校の授業（図画工作・国語・書写）以外で、毛筆による書道を習ったことはありますか？'] == 'いいえ']['ID'].tolist()

print(f"抽出対象となる経験者ID: {exp_ids}") 
# 結果は [2, 3, 4] になるはずです（9が消える）
print(f"抽出対象となる初学者ID: {beg_ids}")

# ==========================================
# 2. 層化抽出のロジック (合計36枚)
# ==========================================
conditions = ['A', 'B', 'C']
characters = ['刁 (チョウ)', '爻 (コウ)', '乎 (コ)']

selected_rows = []

for cond in conditions:
    for char in characters:
        target_rows = df_progress[
            (df_progress['条件'] == cond) & 
            (df_progress['文字'] == char)
        ]
        
        candidates_exp = target_rows[target_rows['被験者ID(記入用)'].isin(exp_ids)]
        candidates_beg = target_rows[target_rows['被験者ID(記入用)'].isin(beg_ids)]
        
        # 経験者から2枚抽出
        try:
            sample_exp = candidates_exp.sample(n=2, random_state=RANDOM_SEED)
        except ValueError:
            sample_exp = candidates_exp 
            
        # 初学者から2枚抽出
        try:
            sample_beg = candidates_beg.sample(n=2, random_state=RANDOM_SEED)
        except ValueError:
            sample_beg = candidates_beg
            
        selected_rows.extend(sample_exp.to_dict('records'))
        selected_rows.extend(sample_beg.to_dict('records'))

df_result = pd.DataFrame(selected_rows)

# ==========================================
# 3. リネームとファイルコピー
# ==========================================
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

results_log = []

print("\n--- 画像抽出・コピー処理開始 ---")

for index, row in df_result.iterrows():
    user_id = row['被験者ID(記入用)']
    seq_num = row['通し番号']
    
    # 元ファイル名 (3桁ゼロ埋め)
    src_filename = f"img_{seq_num:03d}{img_extension}"
    
    # フォルダ名 (3桁ゼロ埋め)
    folder_name = f"{user_id:03d}"
    src_path = os.path.join(source_root, folder_name, src_filename)
    
    # 保存ファイル名 (IDも通し番号も3桁)
    new_filename = f"img_{user_id:03d}_{seq_num:03d}{img_extension}"
    dst_path = os.path.join(output_folder, new_filename)
    
    log_entry = {
        'Original_ID': user_id,
        'Original_Seq': seq_num,
        'Condition': row['条件'],
        'Character': row['文字'],
        'New_Filename': new_filename,
        'Status': ''
    }

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        log_entry['Status'] = 'Success'
        print(f"OK: .../{folder_name}/{src_filename} -> {new_filename}")
    else:
        log_entry['Status'] = 'File Not Found'
        print(f"NG: 見つかりません -> {src_path}")
    
    results_log.append(log_entry)

# ログ保存
df_log = pd.DataFrame(results_log)
df_log.to_csv('extraction_list.csv', index=False, encoding='utf-8-sig')

print(f"\n処理完了。合計 {len(df_result)} 枚を対象としました。")