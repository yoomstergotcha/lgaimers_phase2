# -*- coding: utf-8 -*-
"""
추론 실행 코드 (Inference Script) - PyTorch Tabular ResNet v2 (선수 ID 임베딩 + SE 게이팅 + SiLU) 5-Fold
- 100% 완전한 단일 자급자족(Self-contained) 구조
"""
import os
import sys
import time
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 콘솔 인코딩 안전 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ID_COL = "row_id"
TARGET_COL = "control_success"

# ============================================================
# [1] 피처 엔지니어링 함수
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    if "season" in feat.columns:
        feat["season_offset"] = feat["season"] - 2019
        if "game_month" in feat.columns:
            feat["season_progression"] = (feat["season"] - 2019) + (feat["game_month"] - 3).clip(lower=0) / 9.0
            feat["month_sin"] = np.sin(2 * np.pi * feat["game_month"] / 12.0)
            feat["month_cos"] = np.cos(2 * np.pi * feat["game_month"] / 12.0)
        if "game_dayofweek" in feat.columns:
            feat["dayofweek_sin"] = np.sin(2 * np.pi * feat["game_dayofweek"] / 7.0)
            feat["dayofweek_cos"] = np.cos(2 * np.pi * feat["game_dayofweek"] / 7.0)

    if "strikes_before" in feat.columns and "balls_before" in feat.columns:
        feat["count_diff"] = feat["strikes_before"] - feat["balls_before"]
        feat["is_2strike"] = (feat["strikes_before"] == 2).astype(np.int32)
        feat["is_3ball"] = (feat["balls_before"] == 3).astype(np.int32)
        feat["is_full_count"] = ((feat["balls_before"] == 3) & (feat["strikes_before"] == 2)).astype(np.int32)
        feat["is_first_pitch"] = ((feat["balls_before"] == 0) & (feat["strikes_before"] == 0)).astype(np.int32)
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


# ============================================================
# [2] PyTorch Tabular ResNet v2 아키텍처
# ============================================================
class SqueezeExcitation(nn.Module):
    def __init__(self, num_features, reduction=4):
        super(SqueezeExcitation, self).__init__()
        reduced_dim = max(8, num_features // reduction)
        self.fc = nn.Sequential(
            nn.Linear(num_features, reduced_dim),
            nn.SiLU(),
            nn.Linear(reduced_dim, num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.fc(x)
        return x * weights


class TabularResNetV2(nn.Module):
    def __init__(self, emb_dims, num_features, hidden_dim=256, dropout=0.12):
        super(TabularResNetV2, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cats, emb_dim) for num_cats, emb_dim in emb_dims
        ])
        total_emb_dim = sum(emb_dim for _, emb_dim in emb_dims)
        
        self.num_bn = nn.BatchNorm1d(num_features)
        self.se_gate = SqueezeExcitation(num_features)
        
        in_dim = total_emb_dim + num_features
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.silu1 = nn.SiLU()
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.silu2 = nn.SiLU()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outs, dim=1) if emb_outs else torch.empty(len(x_num), 0, device=x_num.device)
        x_num_norm = self.num_bn(x_num)
        x_num_gated = self.se_gate(x_num_norm)
        
        x = torch.cat([x_emb, x_num_gated], dim=1)
        x = self.input_layer(x)
        x = self.silu1(x + self.block1(x))
        x = self.silu2(x + self.block2(x))
        out = self.head(x)
        return out.squeeze(-1)


class InferenceDataset(Dataset):
    def __init__(self, cats, nums):
        self.cats = cats
        self.nums = nums

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, idx):
        return self.cats[idx], self.nums[idx]


# ============================================================
# [3] 데이터 로드 유틸
# ============================================================
def get_data_dir():
    candidate_dirs = ["./data", "./open", "data", "open"]
    for d in candidate_dirs:
        if os.path.exists(os.path.join(d, "test.csv")):
            return d
    return "./data"


def load_test_data(data_dir: str):
    test_path = os.path.join(data_dir, "test.csv")
    sample_sub_path = os.path.join(data_dir, "sample_submission.csv")

    test_df = pd.read_csv(test_path, encoding="utf-8-sig")
    if os.path.exists(sample_sub_path):
        sub_df = pd.read_csv(sample_sub_path, encoding="utf-8-sig")
    else:
        sub_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: 0.5})

    return test_df, sub_df


# ============================================================
# [4] 메인 추론 함수
# ============================================================
def main():
    start_time = time.time()
    print("=" * 60)
    print("[INFO] PyTorch Tabular ResNet v2 5-Fold 추론 시작")
    print("=" * 60)

    model_dir = "./model"
    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" - 추론 디바이스: {device}")

    # 1. 메타데이터 로드
    meta_path = os.path.join(model_dir, "metadata_nn_v2.pkl")
    meta = joblib.load(meta_path)

    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    cat_encoders = meta["cat_encoders"]
    emb_dims = meta["emb_dims"]
    medians = meta["medians"]
    scalers = meta["scalers"]
    fallback_mean = meta.get("global_mean_success", 0.5238)

    # 2. 데이터 로드 및 피처 생성
    data_dir = get_data_dir()
    test_df, sub_df = load_test_data(data_dir)
    print(f" - Test 데이터 건수: {len(test_df):,}건")

    df_feat = engineer_features(test_df)

    # 범주형 인코딩
    for c in cat_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].astype(str).fillna("MISSING")
            c_map = cat_encoders.get(c, {})
            missing_idx = c_map.get("MISSING", len(c_map))
            df_feat[c] = df_feat[c].map(lambda x: c_map.get(x, missing_idx)).astype(np.int64)
        else:
            df_feat[c] = 0

    # 수치형 결측치 대치
    for c in num_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].fillna(medians.get(c, 0.0))
        else:
            df_feat[c] = medians.get(c, 0.0)

    cat_tensor = torch.tensor(df_feat[cat_cols].values, dtype=torch.long)
    num_raw_array = df_feat[num_cols].values.astype(np.float32)

    # 3. 5-Fold 추론 수행
    all_fold_preds = []
    BATCH_SIZE = 8192 if device.type == "cuda" else 4096

    for fold in range(5):
        fold_model_path = os.path.join(model_dir, f"nn_v2_fold_{fold}.pt")
        if not os.path.exists(fold_model_path):
            continue

        scaler = scalers.get(fold, scalers[0])
        num_scaled = scaler.transform(num_raw_array)
        num_tensor = torch.tensor(num_scaled, dtype=torch.float32)

        ds = InferenceDataset(cat_tensor, num_tensor)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        model = TabularResNetV2(emb_dims, len(num_cols), hidden_dim=256, dropout=0.0).to(device)
        model.load_state_dict(torch.load(fold_model_path, map_location=device))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for cats, nums in loader:
                cats, nums = cats.to(device), nums.to(device)
                preds = model(cats, nums)
                fold_preds.append(preds.cpu().numpy())

        all_fold_preds.append(np.concatenate(fold_preds))

    # 4. 5-Fold 평균 산출
    avg_preds = np.mean(all_fold_preds, axis=0)
    avg_preds = np.clip(avg_preds, 0.0, 1.0)
    avg_preds = np.nan_to_num(avg_preds, nan=fallback_mean)

    # 5. submission.csv 저장
    pred_map = dict(zip(test_df[ID_COL], avg_preds))
    sub_df[TARGET_COL] = sub_df[ID_COL].map(pred_map).fillna(fallback_mean)

    sub_df.to_csv(out_path, index=False, encoding="utf-8")
    elapsed = time.time() - start_time
    print(f"[SUCCESS] 추론 완료 및 저장: {out_path} ({len(sub_df):,}건, 소요시간: {elapsed:.2f}초)")
    print("=" * 60)


if __name__ == "__main__":
    main()
