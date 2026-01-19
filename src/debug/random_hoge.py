import os
import random
import shutil
import pandas as pd
import time

# ==========================================
# 設定エリア
# ==========================================
CSV_FILE = "実験進捗シート.csv"      # CSVファイル名
SOURCE_ROOT = "./実験データ"   # 元画像フォルダ
OUTPUT_DIR = "./evaluation_set_unique" # 保存先
SAMPLES_PER_COND = 7                # 各条件の枚数
MAX_ATTEMPTS = 10000                # 抽選の最大試行回数
RANDOM_SEED = None                  # Noneなら毎回結果が変わります

# 文字名の変換マップ
CHAR_MAP = {
    '刁 (チョウ)': 'Cho',
    '爻 (コウ)': 'Kou',
    '乎 (コ)': 'Ko'
}
# ==========================================

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # 1. データ読み込み
    if not os.path.exists(CSV_FILE):
        print(f"エラー: {CSV_FILE} が見つかりません。")
        return
    
    df = pd.read_csv(CSV_FILE)
    
    # 2. 画像ファイルの存在確認とリスト化
    all_candidates = []
    
    # CSVの各行に対して、実際のファイルがあるか確認
    # ※CSVの「被験者ID(記入用)」とフォルダ名が対応している前提
    for _, row in df.iterrows():
        subj_id = int(row['被験者ID(記入用)'])
        seq_num = int(row['通し番号'])
        cond = row['条件']
        char_raw = row['文字']
        
        # フォルダ名の探索（"02" や "2" などに対応）
        # 数字だけ取り出してマッチングさせる
        target_folder = None
        if os.path.exists(SOURCE_ROOT):
            for d in os.listdir(SOURCE_ROOT):
                d_path = os.path.join(SOURCE_ROOT, d)
                if os.path.isdir(d_path):
                    try:
                        if int(d) == subj_id:
                            target_folder = d_path
                            break
                    except ValueError:
                        continue
        
        if not target_folder:
            continue

        # ファイル探索 (img_00X.png と仮定)
        target_file = None
        for f in os.listdir(target_folder):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            # ファイル名から番号抽出ロジック
            name_body, _ = os.path.splitext(f)
            # "img_005" -> 5
            try:
                # 数字部分を探す
                import re
                nums = re.findall(r'\d+', name_body)
                if nums:
                    # 末尾の数字を通し番号とみなす
                    if int(nums[-1]) == seq_num:
                        target_file = os.path.join(target_folder, f)
                        break
            except:
                pass
        
        if target_file:
            all_candidates.append({
                "path": target_file,
                "subj_id": subj_id,
                "seq_num": seq_num,
                "cond": cond,
                "char_name": CHAR_MAP.get(char_raw, "Unknown"),
                "orig_filename": os.path.basename(target_file),
                # 重複チェック用のキー: (被験者ID, 文字種)
                "unique_key": (subj_id, CHAR_MAP.get(char_raw, "Unknown"))
            })

    print(f"有効な候補画像数: {len(all_candidates)}枚")

    # 3. 制約を満たす抽出（リトライ方式）
    print("条件を満たす組み合わせを探索中...")
    
    selected_set = []
    success = False

    for attempt in range(MAX_ATTEMPTS):
        # リストをシャッフル
        random.shuffle(all_candidates)
        
        current_selection = []
        used_keys = set() # (Subj, Char) の集合
        cond_counts = {'A': 0, 'B': 0, 'C': 0}
        
        for item in all_candidates:
            c = item['cond']
            k = item['unique_key']
            
            # 条件1: その条件(A/B/C)の枠がまだ空いているか
            if cond_counts[c] >= SAMPLES_PER_COND:
                continue
            
            # 条件2: 同じ被験者の同じ文字がすでに選ばれていないか
            if k in used_keys:
                continue
            
            # 採用
            current_selection.append(item)
            used_keys.add(k)
            cond_counts[c] += 1
            
            # 完了判定
            if (cond_counts['A'] == SAMPLES_PER_COND and 
                cond_counts['B'] == SAMPLES_PER_COND and 
                cond_counts['C'] == SAMPLES_PER_COND):
                selected_set = current_selection
                success = True
                break
        
        if success:
            print(f"成功！ {attempt + 1}回目の試行で見つかりました。")
            break
    
    if not success:
        print("エラー: 条件を満たす組み合わせが見つかりませんでした。")
        print("被験者数やデータ数が不足している可能性があります。条件を緩めるかデータを確認してください。")
        return

    # 4. ファイル保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_data = []

    print(f"\n--- 選出結果 ({len(selected_set)}枚) ---")
    for item in selected_set:
        # 新ファイル名: img_被験者ID_通し番号_文字.png
        s_str = str(item['subj_id']).zfill(3)
        n_str = str(item['seq_num']).zfill(3)
        char_label = item['char_name']
        _, ext = os.path.splitext(item['orig_filename'])
        
        new_name = f"img_{s_str}_{n_str}_{char_label}{ext}"
        dst_path = os.path.join(OUTPUT_DIR, new_name)
        
        shutil.copy2(item['path'], dst_path)
        
        log_data.append([new_name, item['cond'], char_label, item['subj_id']])
        print(f"保存: {new_name} (Cond:{item['cond']}, Subj:{item['subj_id']})")

    # リスト出力
    df_log = pd.DataFrame(log_data, columns=["Filename", "Condition", "Character", "SubjectID"])
    df_log = df_log.sort_values(by=["Condition", "SubjectID"])
    df_log.to_csv(os.path.join(OUTPUT_DIR, "selection_list_unique.csv"), index=False)
    
    # 検証結果の表示
    print("\n--- 検証 ---")
    dup_check = df_log.duplicated(subset=['SubjectID', 'Character'])
    if dup_check.any():
        print("警告: 重複が発生しています！")
    else:
        print("OK: 全ての画像において (被験者, 文字) の組み合わせはユニークです。")
        print(f"フォルダ '{OUTPUT_DIR}' を確認してください。")

if __name__ == "__main__":
    main()