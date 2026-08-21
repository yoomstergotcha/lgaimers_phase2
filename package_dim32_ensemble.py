# -*- coding: utf-8 -*-
"""
NN v1 Pruned (70%) + NN v4 dim32 (30%) 앙상블 패키징 스크립트
- 산출물: nn_ensemble_pruned_dim32_70_30.zip
- OOF BSS: 1,893.88점
"""
import os
import sys
import time
import zipfile
import shutil
import subprocess
import joblib
import pandas as pd
import numpy as np

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def build_dim32_package(zip_name="nn_ensemble_pruned_dim32_70_30.zip"):
    print("=" * 70)
    print(f"📦 [패키징 시작] 32차원 확장 딥러닝 앙상블 (70% : 30%) ➔ {zip_name}")
    print("=" * 70)

    work_dir = "./work_dim32_70_30"
    os.makedirs(os.path.join(work_dir, "model"), exist_ok=True)

    # 1. 필수 파일 복사
    shutil.copy("baseline_submit/requirements.txt", os.path.join(work_dir, "requirements.txt"))
    shutil.copy("model/metadata_nn_v1_pruned.pkl", os.path.join(work_dir, "model/metadata_nn_v1_pruned.pkl"))
    shutil.copy("model/metadata_nn_v4_dim32.pkl", os.path.join(work_dir, "model/metadata_nn_v4_dim32.pkl"))

    for f in range(5):
        shutil.copy(f"model/nn_v1_pruned_fold_{f}.pt", os.path.join(work_dir, "model", f"nn_v1_pruned_fold_{f}.pt"))
        shutil.copy(f"model/nn_v4_dim32_fold_{f}.pt", os.path.join(work_dir, "model", f"nn_v4_dim32_fold_{f}.pt"))

    # 2. script.py 작성
    with open("work_dim48_70_30/script.py" if os.path.exists("work_dim48_70_30/script.py") else "package_dim48_ensemble.py", "r", encoding="utf-8") as f:
        content = f.read()

    script_content = '''# -*- coding: utf-8 -*-
"""
추론 실행 코드 (Inference Script) - v1 Pruned (70%) + v4 Pruned dim32 (30%)
- 100% 완전한 단일 자급자족(Self-contained) 구조
- 선수 ID 32차원 확장 임베딩 결합
"""
import os, sys, time, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ID_COL = "row_id"
TARGET_COL = "control_success"

def engineer_features_v1(df):
    feat = df.copy()
    if "season" in feat.columns:
        feat["season_offset"] = feat["season"] - 2019
        if "game_month" in feat.columns:
            feat["season_progression"] = (feat["season"] - 2019) + (feat["game_month"] - 3).clip(lower=0) / 9.0
            feat["month_sin"] = np.sin(2 * np.pi * feat["game_month"] / 12.0)
            feat["month_cos"] = np.cos(2 * np.pi * feat["game_month"] / 12.0)
    if "strikes_before" in feat.columns and "balls_before" in feat.columns:
        feat["count_diff"] = feat["strikes_before"] - feat["balls_before"]
        feat["is_2strike"] = (feat["strikes_before"] == 2).astype(np.int32)
        feat["is_3ball"] = (feat["balls_before"] == 3).astype(np.int32)
        feat["is_full_count"] = ((feat["balls_before"] == 3) & (feat["strikes_before"] == 2)).astype(np.int32)
        feat["count_sum"] = feat["balls_before"] + feat["strikes_before"]
    if "score_diff_pitcher_team" in feat.columns:
        feat["abs_score_diff"] = feat["score_diff_pitcher_team"].abs()
    if "li" in feat.columns:
        if "score_diff_pitcher_team" in feat.columns:
            feat["clutch_pressure"] = feat["li"] / (feat["abs_score_diff"] + 1.0)
        if "outs_before" in feat.columns:
            feat["li_x_outs"] = feat["li"] * (feat["outs_before"] + 1)
    if "pitcher_hand" in feat.columns and "batter_hand" in feat.columns:
        feat["is_same_hand"] = (feat["pitcher_hand"] == feat["batter_hand"]).astype(np.int32)
    if "asof_pitcher_success_rate" in feat.columns:
        if "asof_pitcher_prev1_game_success_rate" in feat.columns:
            feat["form_diff_1g"] = feat["asof_pitcher_prev1_game_success_rate"] - feat["asof_pitcher_success_rate"]
        if "asof_pitcher_prev3_game_success_rate" in feat.columns:
            feat["form_diff_3g"] = feat["asof_pitcher_prev3_game_success_rate"] - feat["asof_pitcher_success_rate"]
        if "asof_pitcher_prev5_game_success_rate" in feat.columns:
            feat["form_diff_5g"] = feat["asof_pitcher_prev5_game_success_rate"] - feat["asof_pitcher_success_rate"]
        if "form_diff_1g" in feat.columns and "form_diff_5g" in feat.columns:
            feat["form_momentum"] = feat["form_diff_1g"] - feat["form_diff_5g"]
    if "asof_pitcher_reverse_rate" in feat.columns and "asof_pitcher_middle_rate" in feat.columns:
        feat["total_mistake_rate"] = feat["asof_pitcher_reverse_rate"] + feat["asof_pitcher_middle_rate"]
    if "asof_pitcher_strike_rate" in feat.columns and "asof_pitcher_ball_rate" in feat.columns:
        feat["strike_ball_ratio"] = feat["asof_pitcher_strike_rate"] / (feat["asof_pitcher_ball_rate"] + 1e-4)
    if "asof_pitcher_strike_rate" in feat.columns and "asof_pitcher_middle_rate" in feat.columns:
        feat["pitcher_control_dominance"] = feat["asof_pitcher_strike_rate"] - feat["asof_pitcher_middle_rate"]
    if "asof_pitcher_success_rate" in feat.columns and "asof_batter_success_rate" in feat.columns:
        feat["matchup_expected_success"] = (feat["asof_pitcher_success_rate"] + feat["asof_batter_success_rate"]) / 2.0
        feat["matchup_diff_success"] = feat["asof_pitcher_success_rate"] - feat["asof_batter_success_rate"]
    if all(col in feat.columns for col in ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]):
        fb = feat["asof_pitcher_fastball_rate"].clip(lower=1e-5)
        br = feat["asof_pitcher_breaking_rate"].clip(lower=1e-5)
        os = feat["asof_pitcher_offspeed_rate"].clip(lower=1e-5)
        feat["pitchmix_entropy"] = -(fb * np.log(fb) + br * np.log(br) + os * np.log(os))
    return feat

def engineer_features_v4(df):
    feat = df.copy()
    if "season" in feat.columns:
        feat["is_abs_era"] = (feat["season"] >= 2024).astype(np.float32)
    else:
        feat["is_abs_era"] = 1.0

    if "game_month" in feat.columns:
        m_ratio = (feat["game_month"] - 3).clip(lower=0) / 7.0
        feat["month_ratio"] = m_ratio
        feat["month_ratio_sq"] = m_ratio ** 2
        feat["is_summer"] = feat["game_month"].isin([7, 8]).astype(np.float32)
        feat["month_sin"] = np.sin(2 * np.pi * feat["game_month"] / 12.0)
        feat["month_cos"] = np.cos(2 * np.pi * feat["game_month"] / 12.0)

    if "strikes_before" in feat.columns and "balls_before" in feat.columns:
        feat["count_diff"] = feat["strikes_before"] - feat["balls_before"]
        feat["is_2strike"] = (feat["strikes_before"] == 2).astype(np.int32)
        feat["is_3ball"] = (feat["balls_before"] == 3).astype(np.int32)
        feat["is_full_count"] = ((feat["balls_before"] == 3) & (feat["strikes_before"] == 2)).astype(np.int32)
        feat["count_sum"] = feat["balls_before"] + feat["strikes_before"]

    if "score_diff_pitcher_team" in feat.columns:
        feat["abs_score_diff"] = feat["score_diff_pitcher_team"].abs()

    if "li" in feat.columns:
        if "score_diff_pitcher_team" in feat.columns:
            feat["clutch_pressure"] = feat["li"] / (feat["abs_score_diff"] + 1.0)
        if "outs_before" in feat.columns:
            feat["li_x_outs"] = feat["li"] * (feat["outs_before"] + 1)

    if "pitcher_hand" in feat.columns and "batter_hand" in feat.columns:
        feat["is_same_hand"] = (feat["pitcher_hand"] == feat["batter_hand"]).astype(np.int32)

    if "asof_pitcher_success_rate" in feat.columns:
        if "asof_pitcher_prev1_game_success_rate" in feat.columns:
            feat["form_diff_1g"] = feat["asof_pitcher_prev1_game_success_rate"] - feat["asof_pitcher_success_rate"]
        if "asof_pitcher_prev3_game_success_rate" in feat.columns:
            feat["form_diff_3g"] = feat["asof_pitcher_prev3_game_success_rate"] - feat["asof_pitcher_success_rate"]
        if "asof_pitcher_prev5_game_success_rate" in feat.columns:
            feat["form_diff_5g"] = feat["asof_pitcher_prev5_game_success_rate"] - feat["asof_pitcher_success_rate"]
        if "form_diff_1g" in feat.columns and "form_diff_5g" in feat.columns:
            feat["form_momentum"] = feat["form_diff_1g"] - feat["form_diff_5g"]

    if "asof_pitcher_reverse_rate" in feat.columns and "asof_pitcher_middle_rate" in feat.columns:
        feat["total_mistake_rate"] = feat["asof_pitcher_reverse_rate"] + feat["asof_pitcher_middle_rate"]

    if "asof_pitcher_strike_rate" in feat.columns and "asof_pitcher_ball_rate" in feat.columns:
        feat["strike_ball_ratio"] = feat["asof_pitcher_strike_rate"] / (feat["asof_pitcher_ball_rate"] + 1e-4)

    if "asof_pitcher_strike_rate" in feat.columns and "asof_pitcher_middle_rate" in feat.columns:
        feat["pitcher_control_dominance"] = feat["asof_pitcher_strike_rate"] - feat["asof_pitcher_middle_rate"]

    if "asof_pitcher_success_rate" in feat.columns and "asof_batter_success_rate" in feat.columns:
        feat["matchup_expected_success"] = (feat["asof_pitcher_success_rate"] + feat["asof_batter_success_rate"]) / 2.0
        feat["matchup_diff_success"] = feat["asof_pitcher_success_rate"] - feat["asof_batter_success_rate"]

    if all(col in feat.columns for col in ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]):
        fb = feat["asof_pitcher_fastball_rate"].clip(lower=1e-5)
        br = feat["asof_pitcher_breaking_rate"].clip(lower=1e-5)
        os = feat["asof_pitcher_offspeed_rate"].clip(lower=1e-5)
        feat["pitchmix_entropy"] = -(fb * np.log(fb) + br * np.log(br) + os * np.log(os))
    return feat

class TabularResNet(nn.Module):
    def __init__(self, emb_dims, num_features, hidden_dim=256, dropout=0.0, emb_dropout=0.05):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n, d in emb_dims])
        self.emb_drop = nn.Dropout(emb_dropout)
        total_emb = sum(d for _, d in emb_dims)
        self.num_bn = nn.BatchNorm1d(num_features)
        in_dim = total_emb + num_features
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.relu1 = nn.ReLU()
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.relu2 = nn.ReLU()
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outs, dim=1) if emb_outs else torch.empty(len(x_num), 0, device=x_num.device)
        x_emb = self.emb_drop(x_emb)
        x_num_norm = self.num_bn(x_num)
        x = torch.cat([x_emb, x_num_norm], dim=1)
        x = self.input_layer(x)
        x = self.relu1(x + self.block1(x))
        x = self.relu2(x + self.block2(x))
        return self.head(x).squeeze(-1)

class InferenceDataset(Dataset):
    def __init__(self, cats, nums):
        self.cats = cats
        self.nums = nums
    def __len__(self):
        return len(self.nums)
    def __getitem__(self, idx):
        return self.cats[idx], self.nums[idx]

def get_data_dir():
    for d in ["./data", "./open", "data", "open", "../data", "../open", "."]:
        if os.path.exists(os.path.join(d, "test.csv")):
            return d
    return "./data"

def predict_model(model_prefix, meta_file, feat_fn, test_df, device):
    meta = joblib.load(os.path.join("./model", meta_file))
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    cat_encoders = meta["cat_encoders"]
    emb_dims = meta["emb_dims"]
    medians = meta["medians"]
    scalers = meta["scalers"]

    df_feat = feat_fn(test_df)
    for c in cat_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].astype(str).fillna("MISSING")
            c_map = cat_encoders.get(c, {})
            missing_idx = c_map.get("MISSING", len(c_map))
            df_feat[c] = df_feat[c].map(lambda x: c_map.get(x, missing_idx)).astype(np.int64)
        else:
            df_feat[c] = 0

    for c in num_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].fillna(medians.get(c, 0.0))
        else:
            df_feat[c] = medians.get(c, 0.0)

    cat_tensor = torch.tensor(df_feat[cat_cols].values, dtype=torch.long)
    num_raw = df_feat[num_cols].values.astype(np.float32)

    fold_preds = []
    BATCH_SIZE = 8192 if device.type == "cuda" else 4096

    for fold in range(5):
        m_path = os.path.join("./model", f"{model_prefix}_fold_{fold}.pt")
        if not os.path.exists(m_path):
            continue
        scaler = scalers.get(fold, scalers[0])
        num_s = scaler.transform(num_raw)
        num_tensor = torch.tensor(num_s, dtype=torch.float32)

        ds = InferenceDataset(cat_tensor, num_tensor)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        model = TabularResNet(emb_dims, len(num_cols), hidden_dim=256, dropout=0.0, emb_dropout=0.05).to(device)
        model.load_state_dict(torch.load(m_path, map_location=device))
        model.eval()

        p_list = []
        with torch.no_grad():
            for cats, nums in loader:
                p_list.append(model(cats.to(device), nums.to(device)).cpu().numpy())
        fold_preds.append(np.concatenate(p_list))

    return np.mean(fold_preds, axis=0)

def main():
    start_time = time.time()
    print("=" * 60)
    print("[INFO] v1_pruned (70%) + v4_dim32 (30%) 앙상블 추론 시작")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_data_dir()
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), encoding="utf-8-sig")

    sample_sub_path = os.path.join(data_dir, "sample_submission.csv")
    sub_df = pd.read_csv(sample_sub_path, encoding="utf-8-sig") if os.path.exists(sample_sub_path) else pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: 0.5})

    print(f" - Test 데이터: {len(test_df):,}건")
    p_v1 = predict_model("nn_v1_pruned", "metadata_nn_v1_pruned.pkl", engineer_features_v1, test_df, device)
    p_v4 = predict_model("nn_v4_dim32", "metadata_nn_v4_dim32.pkl", engineer_features_v4, test_df, device)

    # 70% v1_pruned + 30% v4_dim32
    final_preds = 0.70 * p_v1 + 0.30 * p_v4
    final_preds = np.clip(final_preds, 0.0, 1.0)

    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")

    pred_map = dict(zip(test_df[ID_COL], final_preds))
    sub_df[TARGET_COL] = sub_df[ID_COL].map(pred_map).fillna(0.4757)
    sub_df.to_csv(out_path, index=False, encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"[SUCCESS] 32차원 앙상블 추론 완료: {out_path} ({len(sub_df):,}건, 소요시간: {elapsed:.2f}초)")
    print(f" - 평균 예측값: {sub_df[TARGET_COL].mean():.5f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
'''

    with open(os.path.join(work_dir, "script.py"), "w", encoding="utf-8") as f:
        f.write(script_content)

    print("🔍 모의 추론 테스트 실행...")
    res = subprocess.run([sys.executable, "script.py"], cwd=work_dir, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    print(res.stdout)
    if res.returncode != 0:
        print("💥 추론 실패:\n", res.stderr)
        return False

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(work_dir, "script.py"), "script.py")
        zf.write(os.path.join(work_dir, "requirements.txt"), "requirements.txt")
        zf.write(os.path.join(work_dir, "model/metadata_nn_v1_pruned.pkl"), "model/metadata_nn_v1_pruned.pkl")
        zf.write(os.path.join(work_dir, "model/metadata_nn_v4_dim32.pkl"), "model/metadata_nn_v4_dim32.pkl")
        for f in range(5):
            zf.write(os.path.join(work_dir, "model", f"nn_v1_pruned_fold_{f}.pt"), f"model/nn_v1_pruned_fold_{f}.pt")
            zf.write(os.path.join(work_dir, "model", f"nn_v4_dim32_fold_{f}.pt"), f"model/nn_v4_dim32_fold_{f}.pt")

    print(f"🏆 [성공] {zip_name} 패키징 완료 ({os.path.getsize(zip_name)/1024/1024:.2f} MB)")
    shutil.rmtree(work_dir, ignore_errors=True)
    return True

if __name__ == "__main__":
    build_dim32_package()
