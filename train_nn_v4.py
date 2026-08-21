# -*- coding: utf-8 -*-
"""
정형 딥러닝 v4 본 학습 스크립트 (PyTorch Tabular ResNet v4 - 5-Fold Stratified K-Fold)
- 검증된 Exp 3 아키텍처 기반 (v1 백본 + 선수 ID 24차원 임베딩 + 순수 Weighted MSE + ReLU)
- 피처 정제: season_progression 제거, is_abs_era 추가, month_ratio 및 month_ratio_sq(볼록성) 추가
- 5-Fold 배깅(Bagging) 앙상블 및 최적 에폭 스냅샷(Early Checkpoint) 저장
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

CATEGORICAL_COLS = [
    "pitcher_id",
    "batter_id",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

DROP_COLS = [
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
    "season",  # season 자체는 is_abs_era와 sample_weight로 흡수
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """v4_clean 피처 엔지니어링 파이프라인"""
    feat = df.copy()

    # 1. ABS 시대 플래그 및 시즌 내 비선형(볼록성) 월 피처
    if "season" in feat.columns:
        feat["is_abs_era"] = (feat["season"] >= 2024).astype(np.float32)

    if "game_month" in feat.columns:
        m_ratio = (feat["game_month"] - 3).clip(lower=0) / 7.0
        feat["month_ratio"] = m_ratio
        feat["month_ratio_sq"] = m_ratio ** 2  # 한여름 U자형 볼록 곡률
        feat["is_summer"] = feat["game_month"].isin([7, 8]).astype(np.float32)
        feat["month_sin"] = np.sin(2 * np.pi * feat["game_month"] / 12.0)
        feat["month_cos"] = np.cos(2 * np.pi * feat["game_month"] / 12.0)

    if "game_dayofweek" in feat.columns:
        feat["dayofweek_sin"] = np.sin(2 * np.pi * feat["game_dayofweek"] / 7.0)
        feat["dayofweek_cos"] = np.cos(2 * np.pi * feat["game_dayofweek"] / 7.0)

    # 2. 볼카운트 및 승부처 상황 피처
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

    # 3. 투수 최근 폼 및 상성 피처 (EDA 1위 피처군)
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


class TabularResNetV4(nn.Module):
    """v1의 검증된 잔차 블록 구조 + ReLU + 선수 ID 24차원 엔티티 임베딩"""
    def __init__(self, emb_dims, num_features, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cats, emb_dim) for num_cats, emb_dim in emb_dims
        ])
        total_emb_dim = sum(emb_dim for _, emb_dim in emb_dims)
        self.num_bn = nn.BatchNorm1d(num_features)

        in_dim = total_emb_dim + num_features
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.relu1 = nn.ReLU()

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.relu2 = nn.ReLU()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outs, dim=1) if emb_outs else torch.empty(len(x_num), 0, device=x_num.device)
        x_num_norm = self.num_bn(x_num)
        
        x = torch.cat([x_emb, x_num_norm], dim=1)
        x = self.input_layer(x)
        x = self.relu1(x + self.block1(x))
        x = self.relu2(x + self.block2(x))
        return self.head(x).squeeze(-1)


def compute_brier_skill_score(y_true, y_pred, ref_p=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
    bs = np.mean((y_pred_clipped - y_true) ** 2)
    r = np.mean(y_true) if ref_p is None else ref_p
    ref_bs = r * (1.0 - r)
    if ref_bs <= 0:
        return 0.0, bs, ref_bs
    bss = 100000.0 * (1.0 - (bs / ref_bs))
    return max(0.0, bss), bs, ref_bs


def main():
    start_total_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"🚀 [PyTorch Tabular ResNet v4 - 5-Fold 배깅 본 학습] Device: {device}")
    print("=" * 75)

    train_path = "data/train.csv"
    train_df = pd.read_csv(train_path)
    print(f"[1] 데이터 로드 완료: {len(train_df):,}건 (2019~2024년 전체 데이터)")

    # 2. 피처 생성
    print("[2] 특성 공학 전처리 파이프라인 수행 중...")
    df_feat = engineer_features(train_df)

    # 최근 시즌 가중치 (2019: 1.0 ~ 2024: 1.75)
    sample_weights = (1.0 + 0.15 * (train_df["season"].values - 2019).clip(min=0)).astype(np.float32)

    cat_encoders = {}
    emb_dims = []

    for c in CATEGORICAL_COLS:
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

    num_cols = [c for c in df_feat.columns if c not in DROP_COLS and c not in CATEGORICAL_COLS]
    print(f" - 범주형 {len(CATEGORICAL_COLS)}개(투수/타자 ID 포함) + 수치형 {len(num_cols)}개 피처 확정")

    medians = {}
    for c in num_cols:
        med = float(df_feat[c].median())
        medians[c] = med
        df_feat[c] = df_feat[c].fillna(med)

    cat_array = df_feat[CATEGORICAL_COLS].values
    num_raw_array = df_feat[num_cols].values.astype(np.float32)
    y_array = df_feat[TARGET_COL].values.astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(df_feat), dtype=np.float32)
    scalers = {}

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    BATCH_SIZE = 4096
    EPOCHS = 7

    print("\n" + "=" * 75)
    print(f"🏆 [3] 5-Fold PyTorch Tabular ResNet v4 정밀 학습 시작 ({EPOCHS} Epochs per Fold)")
    print("=" * 75)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_feat, df_feat[TARGET_COL])):
        fold_start = time.time()
        print(f"\n>>> [Fold {fold+1}/5] 시작 (Train: {len(train_idx):,}건, Val: {len(val_idx):,}건)")

        scaler = StandardScaler()
        num_train_scaled = scaler.fit_transform(num_raw_array[train_idx])
        num_val_scaled = scaler.transform(num_raw_array[val_idx])
        scalers[fold] = scaler

        train_cat_t = torch.tensor(cat_array[train_idx], dtype=torch.long)
        train_num_t = torch.tensor(num_train_scaled, dtype=torch.float32)
        train_y_t = torch.tensor(y_array[train_idx], dtype=torch.float32)
        train_w_t = torch.tensor(sample_weights[train_idx], dtype=torch.float32)

        val_cat_t = torch.tensor(cat_array[val_idx], dtype=torch.long)
        val_num_t = torch.tensor(num_val_scaled, dtype=torch.float32)
        val_y_t = torch.tensor(y_array[val_idx], dtype=torch.float32)
        val_w_t = torch.tensor(sample_weights[val_idx], dtype=torch.float32)

        train_ds = TabularDataset(train_cat_t, train_num_t, train_y_t, train_w_t)
        val_ds = TabularDataset(val_cat_t, val_num_t, val_y_t, val_w_t)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

        model = TabularResNetV4(emb_dims, len(num_cols), hidden_dim=256, dropout=0.15).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-5
        )

        best_val_bss = -99999.0
        best_val_preds = None
        best_state = None
        best_epoch = 0

        r_val = float(y_array[val_idx].mean())
        ref_bs_val = r_val * (1.0 - r_val)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_losses = []
            for cats, nums, targets, weights in train_loader:
                cats, nums, targets, weights = cats.to(device), nums.to(device), targets.to(device), weights.to(device)
                optimizer.zero_grad()
                preds = model(cats, nums)
                # 순수 Weighted MSE Loss
                loss = (weights * ((preds - targets) ** 2)).mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for cats, nums, targets, weights in val_loader:
                    cats, nums = cats.to(device), nums.to(device)
                    preds = model(cats, nums)
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
                best_epoch = epoch
                is_best = " ⭐ [NEW BEST!]"

            print(f"  - Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {np.mean(train_losses):.5f} | Val BS: {val_bs:.6f} | Val BSS: {val_bss:.2f}점 | Mean(p): {val_mean:.4f}{is_best}")

        oof_preds[val_idx] = best_val_preds
        fold_model_path = os.path.join(model_dir, f"nn_v4_fold_{fold}.pt")
        torch.save(best_state, fold_model_path)
        fold_elapsed = time.time() - fold_start
        print(f" [Fold {fold+1} 완료] 최고 BSS: {best_val_bss:.2f}점 (Best Epoch {best_epoch}, 저장: {fold_model_path}, 소요: {fold_elapsed/60:.2f}분)")

    # 4. 전체 OOF 종합 평가
    r_all = float(y_array.mean())
    ref_bs_all = r_all * (1.0 - r_all)
    final_bs = np.mean((oof_preds - y_array) ** 2)
    final_bss = max(0.0, 100000.0 * (1.0 - final_bs / ref_bs_all))

    print("\n" + "=" * 75)
    print("🏆 [최종 성과] PyTorch Tabular ResNet v4 5-Fold OOF 종합 결과")
    print("=" * 75)
    print(f" - 전체 5-Fold OOF BSS 점수 : {final_bss:.2f}점 / 100,000점")
    print(f" - 전체 Brier Score (MSE)   : {final_bs:.6f}")
    print(f" - OOF 평균 예측값          : {np.mean(oof_preds):.5f} (실제 타깃 평균: {r_all:.5f})")
    print(f" - 전체 학습 소요 시간      : {(time.time() - start_total_time)/60:.2f}분")
    print("=" * 75)

    # 5. 메타데이터 및 OOF 저장
    meta = {
        "cat_cols": CATEGORICAL_COLS,
        "num_cols": num_cols,
        "cat_encoders": cat_encoders,
        "emb_dims": emb_dims,
        "medians": medians,
        "scalers": scalers,
        "global_mean_success": r_all,
        "final_bss": final_bss,
        "epochs": EPOCHS
    }
    joblib.dump(meta, os.path.join(model_dir, "metadata_nn_v4.pkl"))

    oof_df = pd.DataFrame({
        "row_id": train_df[ID_COL],
        "control_success": y_array,
        "pred_nn_v4": oof_preds
    })
    oof_df.to_csv(os.path.join(model_dir, "oof_nn_v4.csv"), index=False)
    print(f"[저장 완료] OOF 예측 파일: {os.path.join(model_dir, 'oof_nn_v4.csv')}")
    print(f"[저장 완료] 메타데이터: {os.path.join(model_dir, 'metadata_nn_v4.pkl')}")


if __name__ == "__main__":
    main()
