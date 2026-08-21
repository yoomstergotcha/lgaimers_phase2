# -*- coding: utf-8 -*-
"""
NN v5 통합 백본 (v1 거시트렌드 + v4 선수ID 24차원 + 비선형 월 곡률) + Multi-Seed (15-Fold) 학습 스크립트
- 단일 모델에 v1과 v4의 모든 검증된 신호 통합 (52개 정제 피처)
- 3개 랜덤 시드(42, 2024, 2026) x 5-Fold = 총 15개 딥러닝 모델 학습
- 시드별 OOF 및 15-Fold 평균 OOF 산출
- 자동 앙상블 및 패키징
"""
import os
import sys
import time
import zipfile
import shutil
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

torch.set_num_threads(min(16, os.cpu_count() or 4))

ID_COL = "row_id"
TARGET_COL = "control_success"

# 11개 추가 제거 피처 목록
PRUNED_COLS = [
    "asof_pitcher_pitchmix_n",
    "away_win_expectancy",
    "game_month",
    "runner_on_1b",
    "num_runners_on",
    "game_dayofweek",
    "dayofweek_sin",
    "dayofweek_cos",
    "score_diff_home",
    "is_first_pitch",
    "outs_before",
]

BASE_DROP_COLS = [
    "row_id",
    "control_success",
    "runner_on_2b",
    "runner_on_3b",
    "is_scoring_position",
    "is_winning",
    "is_losing",
    "is_late_inning",
    "is_empty_bases",
    "is_high_leverage",
    "is_clutch_score",
    "is_bases_loaded",
    "season",
]

ALL_DROP_COLS = BASE_DROP_COLS + PRUNED_COLS


# --- NN v5 Feature Engineering (통합 백본) ---
def engineer_features_v5(df):
    feat = df.copy()

    # 1. 연도 거시 트렌드 (v1 핵심 신호)
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

    # 2. 시즌 내 비선형 U자형 월 곡률 (v4 핵심 신호)
    if "game_month" in feat.columns:
        m_ratio = (feat["game_month"] - 3).clip(lower=0) / 7.0
        feat["month_ratio"] = m_ratio
        feat["month_ratio_sq"] = m_ratio ** 2
        feat["is_summer"] = feat["game_month"].isin([7, 8]).astype(np.float32)
        feat["month_sin"] = np.sin(2 * np.pi * feat["game_month"] / 12.0)
        feat["month_cos"] = np.cos(2 * np.pi * feat["game_month"] / 12.0)

    # 3. 볼카운트 및 승부처 상황
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

    # 4. 최근 폼 및 실수율 통계
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

    # 5. 매치업 기대 제구율 및 다양성
    if "asof_pitcher_success_rate" in feat.columns and "asof_batter_success_rate" in feat.columns:
        feat["matchup_expected_success"] = (feat["asof_pitcher_success_rate"] + feat["asof_batter_success_rate"]) / 2.0
        feat["matchup_diff_success"] = feat["asof_pitcher_success_rate"] - feat["asof_batter_success_rate"]

    if all(col in feat.columns for col in ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]):
        fb = feat["asof_pitcher_fastball_rate"].clip(lower=1e-5)
        br = feat["asof_pitcher_breaking_rate"].clip(lower=1e-5)
        os = feat["asof_pitcher_offspeed_rate"].clip(lower=1e-5)
        feat["pitchmix_entropy"] = -(fb * np.log(fb) + br * np.log(br) + os * np.log(os))

    return feat


class TabularDataset(Dataset):
    def __init__(self, cats, nums, targets=None, weights=None):
        self.cats = cats
        self.nums = nums
        self.targets = targets
        self.weights = weights

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.cats[idx], self.nums[idx], self.targets[idx], self.weights[idx]
        return self.cats[idx], self.nums[idx]


class TabularResNet(nn.Module):
    def __init__(self, emb_dims, num_features, hidden_dim=256, dropout=0.15):
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


def train_multiseed_v5():
    start_time = time.time()
    print("=" * 75)
    print("🚀 [NN v5 통합 백본] 거시트렌드 + 선수ID 24차원 + 비선형 월 Multi-Seed 학습 시작")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv("data/train.csv")
    sample_weights = (1.0 + 0.15 * (train_df["season"].values - 2019).clip(min=0)).astype(np.float32)

    cat_cols = [
        "pitcher_id", "batter_id", "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"
    ]

    df_feat = engineer_features_v5(train_df)

    cat_encoders = {}
    emb_dims = []

    for c in cat_cols:
        df_feat[c] = df_feat[c].astype(str).fillna("MISSING")
        unique_vals = sorted(df_feat[c].unique().tolist())
        if "MISSING" not in unique_vals:
            unique_vals.append("MISSING")
        c_map = {val: idx for idx, val in enumerate(unique_vals)}
        cat_encoders[c] = c_map
        df_feat[c] = df_feat[c].map(c_map).astype(np.int64)

        num_cats = len(c_map)
        if c in ["pitcher_id", "batter_id"]:
            emb_dim = 24
        elif num_cats <= 4:
            emb_dim = 4
        elif num_cats <= 15:
            emb_dim = 8
        else:
            emb_dim = min(32, max(4, int(np.sqrt(num_cats) * 2)))
        emb_dims.append((num_cats, emb_dim))

    num_cols = [c for c in df_feat.columns if c not in ALL_DROP_COLS and c not in cat_cols]
    print(f" - 확정 피처: 범주형 {len(cat_cols)}개 + 수치형 {len(num_cols)}개 (v1 거시트렌드 + v4 선수ID 24차원 통합)")

    medians = {}
    for c in num_cols:
        med = float(df_feat[c].median())
        medians[c] = med
        df_feat[c] = df_feat[c].fillna(med)

    cat_array = df_feat[cat_cols].values
    num_raw_array = df_feat[num_cols].values.astype(np.float32)
    y_array = df_feat[TARGET_COL].values.astype(np.float32)

    r_all = float(y_array.mean())
    ref_bs_all = r_all * (1.0 - r_all)

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    BATCH_SIZE = 4096
    EPOCHS = 7

    SEEDS = [42, 2024, 2026]
    seed_oofs = {}
    seed_scalers = {}

    for seed in SEEDS:
        print("\n" + "=" * 75)
        print(f"🌱 [Seed {seed}] 5-Fold 학습 시작 (EPOCHS: {EPOCHS})")
        print("=" * 75)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(df_feat), dtype=np.float32)
        scalers = {}

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_feat, df_feat[TARGET_COL])):
            fold_path = os.path.join(model_dir, f"nn_v5_s{seed}_fold_{fold}.pt")
            
            scaler = StandardScaler()
            num_train_scaled = scaler.fit_transform(num_raw_array[train_idx])
            num_val_scaled = scaler.transform(num_raw_array[val_idx])
            scalers[fold] = scaler

            train_ds = TabularDataset(
                torch.tensor(cat_array[train_idx], dtype=torch.long),
                torch.tensor(num_train_scaled, dtype=torch.float32),
                torch.tensor(y_array[train_idx], dtype=torch.float32),
                torch.tensor(sample_weights[train_idx], dtype=torch.float32)
            )
            val_ds = TabularDataset(
                torch.tensor(cat_array[val_idx], dtype=torch.long),
                torch.tensor(num_val_scaled, dtype=torch.float32),
                torch.tensor(y_array[val_idx], dtype=torch.float32),
                torch.tensor(sample_weights[val_idx], dtype=torch.float32)
            )

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

            r_val = float(y_array[val_idx].mean())
            ref_bs_val = r_val * (1.0 - r_val)

            # 이미 완료된 체크포인트 확인
            if os.path.exists(fold_path):
                print(f"  [v5_s{seed}] Fold {fold+1} 기존 완료 파일 발견 ➔ 로드하여 OOF 산출...")
                model = TabularResNet(emb_dims, len(num_cols), hidden_dim=256, dropout=0.15).to(device)
                model.load_state_dict(torch.load(fold_path, map_location=device))
                model.eval()

                val_preds_list = []
                with torch.no_grad():
                    for cats, nums, targets, weights in val_loader:
                        preds = model(cats.to(device), nums.to(device))
                        val_preds_list.append(preds.cpu().numpy())

                fold_val_preds = np.concatenate(val_preds_list)
                val_bs = np.mean((fold_val_preds - y_array[val_idx]) ** 2)
                val_bss = max(0.0, 100000.0 * (1.0 - val_bs / ref_bs_val))
                oof_preds[val_idx] = fold_val_preds
                print(f"  [v5_s{seed}] Fold {fold+1} 복구 완료! Val BSS: {val_bss:.2f}점 | MSE: {val_bs:.6f}")
                continue

            model = TabularResNet(emb_dims, len(num_cols), hidden_dim=256, dropout=0.15).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-5
            )

            best_val_bss = -99999.0
            best_val_preds = None
            best_state = None

            for epoch in range(1, EPOCHS + 1):
                model.train()
                train_losses = []
                for cats, nums, targets, weights in train_loader:
                    cats, nums, targets, weights = cats.to(device), nums.to(device), targets.to(device), weights.to(device)
                    optimizer.zero_grad()
                    preds = model(cats, nums)
                    loss = (weights * ((preds - targets) ** 2)).mean()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_losses.append(loss.item())

                model.eval()
                val_preds_list = []
                with torch.no_grad():
                    for cats, nums, targets, weights in val_loader:
                        preds = model(cats.to(device), nums.to(device))
                        val_preds_list.append(preds.cpu().numpy())

                fold_val_preds = np.concatenate(val_preds_list)
                val_bs = np.mean((fold_val_preds - y_array[val_idx]) ** 2)
                val_bss = max(0.0, 100000.0 * (1.0 - val_bs / ref_bs_val))
                val_mean = float(np.mean(fold_val_preds))

                is_best = ""
                if val_bss > best_val_bss:
                    best_val_bss = val_bss
                    best_val_preds = fold_val_preds
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    is_best = " ⭐ [NEW BEST!]"

                print(f"  [v5_s{seed}] Fold {fold+1} Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {np.mean(train_losses):.5f} | Val BSS: {val_bss:.2f}점 | Mean: {val_mean:.4f}{is_best}")

            oof_preds[val_idx] = best_val_preds
            torch.save(best_state, fold_path)

        seed_bs = np.mean((oof_preds - y_array) ** 2)
        seed_bss = max(0.0, 100000.0 * (1.0 - seed_bs / ref_bs_all))
        print(f"🏆 [Seed {seed} 완료] 5-Fold OOF BSS: {seed_bss:.2f}점 (MSE: {seed_bs:.6f}, Mean: {np.mean(oof_preds):.5f})")
        seed_oofs[seed] = oof_preds
        seed_scalers[seed] = scalers

    # Multi-Seed 15-Fold 앙상블 OOF 계산
    multi_oof = np.mean([seed_oofs[s] for s in SEEDS], axis=0)
    multi_bs = np.mean((multi_oof - y_array) ** 2)
    multi_bss = max(0.0, 100000.0 * (1.0 - multi_bs / ref_bs_all))

    print("\n" + "=" * 75)
    print(f"👑 [NN v5 Multi-Seed (15-Fold) 최종 결과] OOF BSS: {multi_bss:.2f}점 (MSE: {multi_bs:.6f}, Mean: {np.mean(multi_oof):.5f})")
    print("=" * 75)

    meta = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_encoders": cat_encoders,
        "emb_dims": emb_dims,
        "medians": medians,
        "scalers": seed_scalers[42], # 대표 scaler
        "seed_scalers": seed_scalers,
        "seeds": SEEDS,
        "global_mean_success": r_all,
        "final_bss": multi_bss,
        "pruned_cols": PRUNED_COLS
    }
    joblib.dump(meta, os.path.join(model_dir, "metadata_nn_v5.pkl"))

    oof_df = pd.DataFrame({
        "row_id": train_df[ID_COL],
        "control_success": y_array,
        "pred_nn_v5_s42": seed_oofs[42],
        "pred_nn_v5_s2024": seed_oofs[2024],
        "pred_nn_v5_s2026": seed_oofs[2026],
        "pred_nn_v5_multiseed": multi_oof
    })
    oof_df.to_csv(os.path.join(model_dir, "oof_nn_v5.csv"), index=False)

    # 3-Way 앙상블 그리드 서치 (v1_pruned + v5_multiseed)
    oof_v1_df = pd.read_csv("model/oof_nn_v1_pruned.csv")
    oof_v1_p = oof_v1_df["pred_nn_v1_pruned"].values

    print("\n" + "=" * 75)
    print("🏆 [v1_pruned + v5_multiseed 앙상블 그리드 서치]")
    print("=" * 75)
    best_blend_bss = 0.0
    best_w = 0.50

    for w1 in [1.0, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.0]:
        w5 = 1.0 - w1
        blend = w1 * oof_v1_p + w5 * multi_oof
        bs = np.mean((blend - y_array) ** 2)
        bss = 100000.0 * (1.0 - bs / ref_bs_all)
        star = " 👑 [BEST]" if bss > best_blend_bss else ""
        print(f" - (v1_pruned {w1*100:02.0f}% : v5_multiseed {w5*100:02.0f}%) ➔ OOF BSS: {bss:.2f}점 | MSE: {bs:.6f} | Mean: {np.mean(blend):.5f}{star}")
        if bss > best_blend_bss:
            best_blend_bss = bss
            best_w = w1

    elapsed = (time.time() - start_time) / 60
    print(f"\n[전체 완료] 총 소요 시간: {elapsed:.2f}분 | 최적 앙상블 OOF 점수: {best_blend_bss:.2f}점")


if __name__ == "__main__":
    train_multiseed_v5()
