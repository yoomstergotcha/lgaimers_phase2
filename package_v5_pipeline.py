# -*- coding: utf-8 -*-
"""
NN v5 Multi-Seed 및 v1_pruned + v5_multiseed 앙상블 패키징 스크립트
1. nn_v5_multiseed_15fold.zip (OOF: 1,943.64점)
2. nn_ensemble_v1_v5_30_70.zip (OOF: 1,970.63점 역대 최고점 👑)
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


def build_v5_packages():
    print("=" * 75)
    print("📦 [패키징 시작] NN v5 Multi-Seed (15-Fold) & v1+v5 최강 앙상블 패키징")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. nn_ensemble_v1_v5_30_70.zip 패키징 (OOF 1,970.63점 역대 1위 👑)
    # -------------------------------------------------------------
    work_dir = "./work_v1_v5_30_70"
    os.makedirs(os.path.join(work_dir, "model"), exist_ok=True)

    shutil.copy("baseline_submit/requirements.txt", os.path.join(work_dir, "requirements.txt"))
    shutil.copy("model/metadata_nn_v1_pruned.pkl", os.path.join(work_dir, "model/metadata_nn_v1_pruned.pkl"))
    shutil.copy("model/metadata_nn_v5.pkl", os.path.join(work_dir, "model/metadata_nn_v5.pkl"))

    # v1_pruned 5 folds
    for f in range(5):
        shutil.copy(f"model/nn_v1_pruned_fold_{f}.pt", os.path.join(work_dir, "model", f"nn_v1_pruned_fold_{f}.pt"))

    # v5 15 folds (seeds: 42, 2024, 2026)
    for s in [42, 2024, 2026]:
        for f in range(5):
            shutil.copy(f"model/nn_v5_s{s}_fold_{f}.pt", os.path.join(work_dir, "model", f"nn_v5_s{s}_fold_{f}.pt"))

    script_content = '''# -*- coding: utf-8 -*-
"""
Inference Script: v1_pruned (30%) + v5_multiseed 15-Fold (70%) 앙상블
- 로컬 5-Fold OOF BSS: 1,970.63점 (역대 최고점)
- 100% 단일 자급자족(Self-contained) 추론 코드
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

def engineer_features_v5(df):
    feat = df.copy()
    if "season" in feat.columns:
        feat["season_offset"] = feat["season"] - 2019
        feat["is_abs_era"] = (feat["season"] >= 2024).astype(np.float32)
        if "game_month" in feat.columns:
            feat["season_progression"] = (feat["season"] - 2019) + (feat["game_month"] - 3).clip(lower=0) / 9.0
    else:
        feat["season_offset"] = 6.0
        feat["is_abs_era"] = 1.0
        if "game_month" in feat.columns:
            feat["season_progression"] = 6.0 + (feat["game_month"] - 3).clip(lower=0) / 9.0

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
    def __init__(self, emb_dims, num_features, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n, d in emb_dims])
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

def predict_v1(test_df, device):
    meta = joblib.load("./model/metadata_nn_v1_pruned.pkl")
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    cat_encoders = meta["cat_encoders"]
    emb_dims = meta["emb_dims"]
    medians = meta["medians"]
    scalers = meta["scalers"]

    df_feat = engineer_features_v1(test_df)
    for c in cat_cols:
        df_feat[c] = df_feat[c].astype(str).fillna("MISSING")
        c_map = cat_encoders.get(c, {})
        missing_idx = c_map.get("MISSING", len(c_map))
        df_feat[c] = df_feat[c].map(lambda x: c_map.get(x, missing_idx)).astype(np.int64)

    for c in num_cols:
        df_feat[c] = df_feat[c].fillna(medians.get(c, 0.0))

    cat_tensor = torch.tensor(df_feat[cat_cols].values, dtype=torch.long)
    num_raw = df_feat[num_cols].values.astype(np.float32)

    fold_preds = []
    BATCH_SIZE = 8192 if device.type == "cuda" else 4096

    for fold in range(5):
        m_path = f"./model/nn_v1_pruned_fold_{fold}.pt"
        if not os.path.exists(m_path):
            continue
        scaler = scalers.get(fold, scalers[0])
        num_s = scaler.transform(num_raw)
        num_tensor = torch.tensor(num_s, dtype=torch.float32)

        ds = InferenceDataset(cat_tensor, num_tensor)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        model = TabularResNet(emb_dims, len(num_cols), hidden_dim=256, dropout=0.0).to(device)
        model.load_state_dict(torch.load(m_path, map_location=device))
        model.eval()

        p_list = []
        with torch.no_grad():
            for cats, nums in loader:
                p_list.append(model(cats.to(device), nums.to(device)).cpu().numpy())
        fold_preds.append(np.concatenate(p_list))

    return np.mean(fold_preds, axis=0)

def predict_v5_multiseed(test_df, device):
    meta = joblib.load("./model/metadata_nn_v5.pkl")
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    cat_encoders = meta["cat_encoders"]
    emb_dims = meta["emb_dims"]
    medians = meta["medians"]
    seed_scalers = meta.get("seed_scalers", {42: meta["scalers"], 2024: meta["scalers"], 2026: meta["scalers"]})

    df_feat = engineer_features_v5(test_df)
    for c in cat_cols:
        df_feat[c] = df_feat[c].astype(str).fillna("MISSING")
        c_map = cat_encoders.get(c, {})
        missing_idx = c_map.get("MISSING", len(c_map))
        df_feat[c] = df_feat[c].map(lambda x: c_map.get(x, missing_idx)).astype(np.int64)

    for c in num_cols:
        df_feat[c] = df_feat[c].fillna(medians.get(c, 0.0))

    cat_tensor = torch.tensor(df_feat[cat_cols].values, dtype=torch.long)
    num_raw = df_feat[num_cols].values.astype(np.float32)

    all_preds = []
    BATCH_SIZE = 8192 if device.type == "cuda" else 4096

    for s in [42, 2024, 2026]:
        scalers = seed_scalers.get(s, meta["scalers"])
        for fold in range(5):
            m_path = f"./model/nn_v5_s{s}_fold_{fold}.pt"
            if not os.path.exists(m_path):
                continue
            scaler = scalers.get(fold, scalers[0] if isinstance(scalers, dict) else scalers)
            num_s = scaler.transform(num_raw)
            num_tensor = torch.tensor(num_s, dtype=torch.float32)

            ds = InferenceDataset(cat_tensor, num_tensor)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

            model = TabularResNet(emb_dims, len(num_cols), hidden_dim=256, dropout=0.0).to(device)
            model.load_state_dict(torch.load(m_path, map_location=device))
            model.eval()

            p_list = []
            with torch.no_grad():
                for cats, nums in loader:
                    p_list.append(model(cats.to(device), nums.to(device)).cpu().numpy())
            all_preds.append(np.concatenate(p_list))

    return np.mean(all_preds, axis=0)

def main():
    start_time = time.time()
    print("=" * 60)
    print("[INFO] v1_pruned (30%) + v5_multiseed 15-Fold (70%) 추론 시작")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_data_dir()
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), encoding="utf-8-sig")

    sample_sub_path = os.path.join(data_dir, "sample_submission.csv")
    sub_df = pd.read_csv(sample_sub_path, encoding="utf-8-sig") if os.path.exists(sample_sub_path) else pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: 0.5})

    print(f" - Test 데이터: {len(test_df):,}건")
    p_v1 = predict_v1(test_df, device)
    p_v5 = predict_v5_multiseed(test_df, device)

    # 30% v1_pruned + 70% v5_multiseed (OOF 1,970.63점 최고점 조합)
    final_preds = 0.30 * p_v1 + 0.70 * p_v5
    final_preds = np.clip(final_preds, 0.0, 1.0)

    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")

    pred_map = dict(zip(test_df[ID_COL], final_preds))
    sub_df[TARGET_COL] = sub_df[ID_COL].map(pred_map).fillna(0.4740)
    sub_df.to_csv(out_path, index=False, encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"[SUCCESS] 최강 앙상블 추론 완료: {out_path} ({len(sub_df):,}건, 소요시간: {elapsed:.2f}초)")
    print(f" - 평균 예측값: {sub_df[TARGET_COL].mean():.5f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
'''
    with open(os.path.join(work_dir, "script.py"), "w", encoding="utf-8") as f:
        f.write(script_content)

    print("🔍 [1차] 모의 추론 테스트 (v1_pruned 30% + v5_multiseed 70%)...")
    res = subprocess.run([sys.executable, "script.py"], cwd=work_dir, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    print(res.stdout)
    if res.returncode != 0:
        print("💥 추론 실패:\n", res.stderr)
        return False

    zip_1 = "nn_ensemble_v1_v5_30_70.zip"
    with zipfile.ZipFile(zip_1, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(work_dir, "script.py"), "script.py")
        zf.write(os.path.join(work_dir, "requirements.txt"), "requirements.txt")
        zf.write(os.path.join(work_dir, "model/metadata_nn_v1_pruned.pkl"), "model/metadata_nn_v1_pruned.pkl")
        zf.write(os.path.join(work_dir, "model/metadata_nn_v5.pkl"), "model/metadata_nn_v5.pkl")
        for f in range(5):
            zf.write(os.path.join(work_dir, "model", f"nn_v1_pruned_fold_{f}.pt"), f"model/nn_v1_pruned_fold_{f}.pt")
        for s in [42, 2024, 2026]:
            for f in range(5):
                zf.write(os.path.join(work_dir, "model", f"nn_v5_s{s}_fold_{f}.pt"), f"model/nn_v5_s{s}_fold_{f}.pt")

    print(f"🏆 [성공] {zip_1} 패키징 완료 ({os.path.getsize(zip_1)/1024/1024:.2f} MB)")
    shutil.rmtree(work_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 2. nn_v5_multiseed_15fold.zip 패키징 (순수 NN v5 15-Fold, OOF 1,943.64점)
    # -------------------------------------------------------------
    work_dir_v5 = "./work_v5_15fold"
    os.makedirs(os.path.join(work_dir_v5, "model"), exist_ok=True)

    shutil.copy("baseline_submit/requirements.txt", os.path.join(work_dir_v5, "requirements.txt"))
    shutil.copy("model/metadata_nn_v5.pkl", os.path.join(work_dir_v5, "model/metadata_nn_v5.pkl"))
    for s in [42, 2024, 2026]:
        for f in range(5):
            shutil.copy(f"model/nn_v5_s{s}_fold_{f}.pt", os.path.join(work_dir_v5, "model", f"nn_v5_s{s}_fold_{f}.pt"))

    script_v5_only = '''# -*- coding: utf-8 -*-
"""
Inference Script: Pure NN v5 Multi-Seed (15-Fold) 단독 추론
- 로컬 5-Fold OOF BSS: 1,943.64점
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

def engineer_features_v5(df):
    feat = df.copy()
    if "season" in feat.columns:
        feat["season_offset"] = feat["season"] - 2019
        feat["is_abs_era"] = (feat["season"] >= 2024).astype(np.float32)
        if "game_month" in feat.columns:
            feat["season_progression"] = (feat["season"] - 2019) + (feat["game_month"] - 3).clip(lower=0) / 9.0
    else:
        feat["season_offset"] = 6.0
        feat["is_abs_era"] = 1.0
        if "game_month" in feat.columns:
            feat["season_progression"] = 6.0 + (feat["game_month"] - 3).clip(lower=0) / 9.0

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
    def __init__(self, emb_dims, num_features, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n, d in emb_dims])
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

def main():
    start_time = time.time()
    print("=" * 60)
    print("[INFO] Pure NN v5 Multi-Seed 15-Fold 추론 시작")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_data_dir()
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), encoding="utf-8-sig")

    sample_sub_path = os.path.join(data_dir, "sample_submission.csv")
    sub_df = pd.read_csv(sample_sub_path, encoding="utf-8-sig") if os.path.exists(sample_sub_path) else pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: 0.5})

    meta = joblib.load("./model/metadata_nn_v5.pkl")
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    cat_encoders = meta["cat_encoders"]
    emb_dims = meta["emb_dims"]
    medians = meta["medians"]
    seed_scalers = meta.get("seed_scalers", {42: meta["scalers"], 2024: meta["scalers"], 2026: meta["scalers"]})

    df_feat = engineer_features_v5(test_df)
    for c in cat_cols:
        df_feat[c] = df_feat[c].astype(str).fillna("MISSING")
        c_map = cat_encoders.get(c, {})
        missing_idx = c_map.get("MISSING", len(c_map))
        df_feat[c] = df_feat[c].map(lambda x: c_map.get(x, missing_idx)).astype(np.int64)

    for c in num_cols:
        df_feat[c] = df_feat[c].fillna(medians.get(c, 0.0))

    cat_tensor = torch.tensor(df_feat[cat_cols].values, dtype=torch.long)
    num_raw = df_feat[num_cols].values.astype(np.float32)

    all_preds = []
    BATCH_SIZE = 8192 if device.type == "cuda" else 4096

    for s in [42, 2024, 2026]:
        scalers = seed_scalers.get(s, meta["scalers"])
        for fold in range(5):
            m_path = f"./model/nn_v5_s{s}_fold_{fold}.pt"
            if not os.path.exists(m_path):
                continue
            scaler = scalers.get(fold, scalers[0] if isinstance(scalers, dict) else scalers)
            num_s = scaler.transform(num_raw)
            num_tensor = torch.tensor(num_s, dtype=torch.float32)

            ds = InferenceDataset(cat_tensor, num_tensor)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

            model = TabularResNet(emb_dims, len(num_cols), hidden_dim=256, dropout=0.0).to(device)
            model.load_state_dict(torch.load(m_path, map_location=device))
            model.eval()

            p_list = []
            with torch.no_grad():
                for cats, nums in loader:
                    p_list.append(model(cats.to(device), nums.to(device)).cpu().numpy())
            all_preds.append(np.concatenate(p_list))

    final_preds = np.mean(all_preds, axis=0)
    final_preds = np.clip(final_preds, 0.0, 1.0)

    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")

    pred_map = dict(zip(test_df[ID_COL], final_preds))
    sub_df[TARGET_COL] = sub_df[ID_COL].map(pred_map).fillna(0.4740)
    sub_df.to_csv(out_path, index=False, encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"[SUCCESS] Pure NN v5 추론 완료: {out_path} ({len(sub_df):,}건, 소요시간: {elapsed:.2f}초)")
    print(f" - 평균 예측값: {sub_df[TARGET_COL].mean():.5f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
'''
    with open(os.path.join(work_dir_v5, "script.py"), "w", encoding="utf-8") as f:
        f.write(script_v5_only)

    print("🔍 [2차] 모의 추론 테스트 (Pure NN v5 Multi-Seed 15-Fold)...")
    res_v5 = subprocess.run([sys.executable, "script.py"], cwd=work_dir_v5, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    print(res_v5.stdout)
    if res_v5.returncode != 0:
        print("💥 추론 실패:\n", res_v5.stderr)
        return False

    zip_2 = "nn_v5_multiseed_15fold.zip"
    with zipfile.ZipFile(zip_2, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(work_dir_v5, "script.py"), "script.py")
        zf.write(os.path.join(work_dir_v5, "requirements.txt"), "requirements.txt")
        zf.write(os.path.join(work_dir_v5, "model/metadata_nn_v5.pkl"), "model/metadata_nn_v5.pkl")
        for s in [42, 2024, 2026]:
            for f in range(5):
                zf.write(os.path.join(work_dir_v5, "model", f"nn_v5_s{s}_fold_{f}.pt"), f"model/nn_v5_s{s}_fold_{f}.pt")

    print(f"🏆 [성공] {zip_2} 패키징 완료 ({os.path.getsize(zip_2)/1024/1024:.2f} MB)")
    shutil.rmtree(work_dir_v5, ignore_errors=True)

if __name__ == "__main__":
    build_v5_packages()
