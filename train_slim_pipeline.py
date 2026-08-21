# -*- coding: utf-8 -*-
"""
다중공선성(r > 0.80) 9개 변수를 추가 제거한 초압축(Slim) 피처 세트 본 학습 파이프라인
1. NN v1 Slim (5-Fold, 수치형 41개 + 범주형 9개) 학습 및 OOF 산출
2. NN v4 Slim (5-Fold, 수치형 43개 + 범주형 9개) 학습 및 OOF 산출
3. Slim OOF 앙상블 그리드 서치 및 2025 테스트셋 캘리브레이션 무결성 검증
"""
import os
import sys
import time
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
SEED = 42

def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

ID_COL = "row_id"
TARGET_COL = "control_success"

# 1. 기존 제거 피처 목록 (기본 식별자 + 중복 플래그 10종 + 1차 노이즈 11종)
BASE_AND_PRUNED_DROP = [
    "row_id", "control_success", "season",
    # 중복 플래그 10종
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "is_scoring_position", "is_winning", "is_losing", "is_late_inning",
    "is_empty_bases", "is_high_leverage", "is_clutch_score", "is_bases_loaded",
    # 1차 노이즈 및 중복
    "asof_pitcher_pitchmix_n", "away_win_expectancy", "game_month", "game_dayofweek",
    "dayofweek_sin", "dayofweek_cos", "score_diff_home", "is_first_pitch", "outs_before",
]

# 2. 다중공선성(r > 0.80) 해소를 위한 추가 9개 제거 피처
SLIM_EXTRA_DROP = [
    "run_total_before",                     # top + bot 단순 합 (r = 0.8049)
    "count_sum",                            # S + B 단순 합 (r = 0.8583)
    "asof_pitcher_fastball_rate",           # 1.0 - (BR + OS) 선형 종속 및 r = 0.00022 노이즈
    "asof_pitcher_prev1_game_success_rate", # form_diff_1g와 r = 0.9106 중복
    "asof_pitcher_prev5_game_success_rate", # prev3_game과 r = 0.8819 중복
    "asof_pitcher_prev5_game_middle_rate",  # prev3_middle과 r = 0.8460 중복
    "asof_pitcher_reverse_rate",            # total_mistake_rate와 r = 0.9047 중복
    "month_cos",                            # month_sin 및 month_ratio_sq와 중복
    "abs_score_diff",                       # clutch_pressure 분모에 이미 반영
]

ALL_DROP_COLS_SLIM = BASE_AND_PRUNED_DROP + SLIM_EXTRA_DROP


# --- Feature Engineering v1 Slim (수치형 41개) ---
def engineer_features_v1_slim(df):
    feat = df.copy()
    if "season" in feat.columns:
        feat["season_offset"] = feat["season"] - 2019
        if "game_month" in feat.columns:
            feat["season_progression"] = (feat["season"] - 2019) + (feat["game_month"] - 3).clip(lower=0) / 9.0
            feat["month_sin"] = np.sin(2 * np.pi * feat["game_month"] / 12.0)
    else:
        feat["season_offset"] = 6.0
        if "game_month" in feat.columns:
            feat["season_progression"] = 6.0 + (feat["game_month"] - 3).clip(lower=0) / 9.0
            feat["month_sin"] = np.sin(2 * np.pi * feat["game_month"] / 12.0)

    if "strikes_before" in feat.columns and "balls_before" in feat.columns:
        feat["count_diff"] = feat["strikes_before"] - feat["balls_before"]
        feat["is_2strike"] = (feat["strikes_before"] == 2).astype(np.int32)
        feat["is_3ball"] = (feat["balls_before"] == 3).astype(np.int32)
        feat["is_full_count"] = ((feat["balls_before"] == 3) & (feat["strikes_before"] == 2)).astype(np.int32)

    if "li" in feat.columns and "score_diff_pitcher_team" in feat.columns:
        feat["clutch_pressure"] = feat["li"] / (feat["score_diff_pitcher_team"].abs() + 1.0)
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


# --- Feature Engineering v4 Slim (수치형 43개) ---
def engineer_features_v4_slim(df):
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

    if "strikes_before" in feat.columns and "balls_before" in feat.columns:
        feat["count_diff"] = feat["strikes_before"] - feat["balls_before"]
        feat["is_2strike"] = (feat["strikes_before"] == 2).astype(np.int32)
        feat["is_3ball"] = (feat["balls_before"] == 3).astype(np.int32)
        feat["is_full_count"] = ((feat["balls_before"] == 3) & (feat["strikes_before"] == 2)).astype(np.int32)

    if "li" in feat.columns and "score_diff_pitcher_team" in feat.columns:
        feat["clutch_pressure"] = feat["li"] / (feat["score_diff_pitcher_team"].abs() + 1.0)
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


def train_5fold_model(model_name, feat_fn, cat_cols, df_raw, sample_weights, epochs=7):
    print("\n" + "=" * 75)
    print(f"🚀 [{model_name}] 5-Fold 정밀 학습 시작 (다중공선성 제거 초압축 버전, {epochs} Epochs)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" - 연산 디바이스: {device}")

    df_feat = feat_fn(df_raw)

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

    num_cols = [c for c in df_feat.columns if c not in ALL_DROP_COLS_SLIM and c not in cat_cols]
    print(f" - 범주형 {len(cat_cols)}개 + 수치형 {len(num_cols)}개 피처 확정 (초압축 {len(num_cols)}개 피처 세트)")

    medians = {}
    for c in num_cols:
        med = float(df_feat[c].median())
        medians[c] = med
        df_feat[c] = df_feat[c].fillna(med)

    cat_array = df_feat[cat_cols].values
    num_raw_array = df_feat[num_cols].values.astype(np.float32)
    y_array = df_feat[TARGET_COL].values.astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(df_feat), dtype=np.float32)
    scalers = {}

    model_dir = f"model/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    BATCH_SIZE = 4096

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_feat, df_feat[TARGET_COL])):
        f_start = time.time()
        print(f"\n--- [Fold {fold+1}/5] 학습 시작 ---")

        scaler = StandardScaler()
        num_train_s = scaler.fit_transform(num_raw_array[train_idx])
        num_val_s = scaler.transform(num_raw_array[val_idx])
        scalers[fold] = scaler

        train_ds = TabularDataset(
            torch.tensor(cat_array[train_idx], dtype=torch.long),
            torch.tensor(num_train_s, dtype=torch.float32),
            torch.tensor(y_array[train_idx], dtype=torch.float32),
            torch.tensor(sample_weights[train_idx], dtype=torch.float32)
        )
        val_ds = TabularDataset(
            torch.tensor(cat_array[val_idx], dtype=torch.long),
            torch.tensor(num_val_s, dtype=torch.float32)
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

        model = TabularResNet(emb_dims, len(num_cols), hidden_dim=256, dropout=0.15).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_val_bss = -1e9
        best_val_preds = None

        y_val_actual = y_array[val_idx]
        ref_bs_val = np.mean(y_val_actual) * (1.0 - np.mean(y_val_actual))

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_cats, batch_nums, batch_ys, batch_ws in train_loader:
                batch_cats = batch_cats.to(device)
                batch_nums = batch_nums.to(device)
                batch_ys = batch_ys.to(device)
                batch_ws = batch_ws.to(device)

                optimizer.zero_grad()
                preds = model(batch_cats, batch_nums)
                loss = torch.mean(batch_ws * (preds - batch_ys) ** 2)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_ys)

            scheduler.step()
            train_loss /= len(train_idx)

            # Validation
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for v_cats, v_nums in val_loader:
                    v_cats = v_cats.to(device)
                    v_nums = v_nums.to(device)
                    vp = model(v_cats, v_nums)
                    val_preds_list.append(vp.cpu().numpy())

            val_preds = np.concatenate(val_preds_list)
            val_bs = np.mean((val_preds - y_val_actual) ** 2)
            val_bss = max(0.0, 100000.0 * (1.0 - val_bs / ref_bs_val))

            if val_bss > best_val_bss:
                best_val_bss = val_bss
                best_val_preds = val_preds
                torch.save(model.state_dict(), os.path.join(model_dir, f"model_fold_{fold}.pt"))
                is_best = " 🌟 [Best]"
            else:
                is_best = ""

            print(f" [Epoch {epoch:02d}/{epochs:02d}] Train MSE: {train_loss:.6f} | Val BS: {val_bs:.6f} | Val BSS: {val_bss:7.2f}점 | Val Mean: {np.mean(val_preds):.5f}{is_best}")

        oof_preds[val_idx] = best_val_preds
        f_time = time.time() - f_start
        print(f" -> Fold {fold+1} 완료 (소요 시간: {f_time:.1f}초, Best Val BSS: {best_val_bss:.2f}점)")

    # 전체 OOF BSS 계산
    global_mean = np.mean(y_array)
    ref_bs_global = global_mean * (1.0 - global_mean)
    oof_bs = np.mean((oof_preds - y_array) ** 2)
    oof_bss = max(0.0, 100000.0 * (1.0 - oof_bs / ref_bs_global))

    print("\n" + "=" * 75)
    print(f"🏆 [{model_name}] 전체 5-Fold OOF BSS 결과: {oof_bss:.2f}점 (Brier MSE: {oof_bs:.6f})")
    print(f" - 전체 OOF 예측 평균: {np.mean(oof_preds):.5f} (정답 평균: {global_mean:.5f})")
    print(f" - 전체 OOF 예측 표준편차: {np.std(oof_preds):.5f}")
    print("=" * 75)

    # 아티팩트 저장
    joblib.dump(cat_encoders, os.path.join(model_dir, "cat_encoders.pkl"))
    joblib.dump(scalers, os.path.join(model_dir, "scalers.pkl"))
    joblib.dump(medians, os.path.join(model_dir, "medians.pkl"))
    joblib.dump(num_cols, os.path.join(model_dir, "num_cols.pkl"))
    joblib.dump(emb_dims, os.path.join(model_dir, "emb_dims.pkl"))
    np.save(os.path.join(model_dir, "oof_preds.npy"), oof_preds)

    return oof_preds, oof_bss


def run_slim_pipeline():
    total_start = time.time()
    print("=" * 75)
    print("🌟 [다중공선성 완전 제거 초압축(Slim) 파이프라인 학습 시작]")
    print("=" * 75)

    df_train = pd.read_csv("data/train.csv")
    print(f" - 학습 데이터 로드 완료: {len(df_train):,}건")

    # 최근성 샘플 가중치
    sample_weights = (1.0 + 0.15 * (df_train["season"].values - 2019).clip(min=0)).astype(np.float32)

    cat_cols = [
        "pitcher_id", "batter_id", "top_bottom", "game_type", "base_state",
        "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"
    ]

    # 1. NN v1 Slim 학습 (41개 수치형)
    oof_v1, bss_v1 = train_5fold_model("nn_v1_slim", engineer_features_v1_slim, cat_cols, df_train, sample_weights, epochs=7)

    # 2. NN v4 Slim 학습 (43개 수치형)
    oof_v4, bss_v4 = train_5fold_model("nn_v4_slim", engineer_features_v4_slim, cat_cols, df_train, sample_weights, epochs=7)

    # 3. Slim 앙상블 그리드 서치
    print("\n" + "=" * 75)
    print("📊 [Slim 앙상블 가중치 그리드 서치 (v1 Slim + v4 Slim)]")
    print("=" * 75)

    y_true = df_train[TARGET_COL].values.astype(np.float32)
    ref_bs = np.mean(y_true) * (1.0 - np.mean(y_true))

    best_w = None
    best_ens_bss = -1e9

    for w_v1 in np.linspace(0.0, 1.0, 21):
        w_v4 = 1.0 - w_v1
        ens_pred = w_v1 * oof_v1 + w_v4 * oof_v4
        ens_bs = np.mean((ens_pred - y_true) ** 2)
        ens_bss = max(0.0, 100000.0 * (1.0 - ens_bs / ref_bs))
        is_opt = ""
        if ens_bss > best_ens_bss:
            best_ens_bss = ens_bss
            best_w = (w_v1, w_v4)
            is_opt = " 🌟 [Best]"
        print(f" • v1 Slim {w_v1*100:4.1f}% + v4 Slim {w_v4*100:4.1f}% ➔ OOF BSS: {ens_bss:7.2f}점 | Mean: {np.mean(ens_pred):.5f} | Std: {np.std(ens_pred):.5f}{is_opt}")

    print("\n" + "=" * 75)
    print(f"👑 [Slim 파이프라인 최적 앙상블 완료]")
    print(f" - 최적 결합 비율: v1 Slim {best_w[0]*100:.1f}% + v4 Slim {best_w[1]*100:.1f}%")
    print(f" - 최고 OOF BSS: {best_ens_bss:.2f}점 (v1 단독: {bss_v1:.2f}점 | v4 단독: {bss_v4:.2f}점)")
    print(f" - 총 소요 시간: {(time.time() - total_start)/60:.2f}분")
    print("=" * 75)


if __name__ == "__main__":
    run_slim_pipeline()
