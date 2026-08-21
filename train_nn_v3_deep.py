# -*- coding: utf-8 -*-
"""
정형 딥러닝 v3 심층 집중 학습 (PyTorch Tabular ResNet v3 - 12 Epochs Deep Training)
핵심 강화 전략:
1. 1,600여 명 선수 ID 임베딩의 완전 수렴을 위한 12 에폭 집중 학습 (기존 5에폭 대비 2.4배)
2. Cosine Annealing 완만 감속 스케줄러 (최적 평탄점 Flat Minima 안착)
3. Fold별 최고 검증 점수(Best Validation BSS) 에폭 가중치 스냅샷 자동 저장 (Early Checkpointing)
4. Squeeze-and-Excitation(SE) 동적 피처 게이팅 + SiLU 활성화 함수
5. 라벨 스무딩(0.04) 가중 MSE 손실함수
"""
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 콘솔 인코딩 안전 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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
]


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


class TabularResNetV3(nn.Module):
    def __init__(self, emb_dims, num_features, hidden_dim=256, dropout=0.12):
        super(TabularResNetV3, self).__init__()
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


def label_smoothed_weighted_mse_loss(preds, targets, weights, smoothing=0.04):
    smoothed_targets = targets * (1.0 - smoothing) + 0.5 * smoothing
    loss = weights * ((preds - smoothed_targets) ** 2)
    return loss.mean()


def main():
    start_total_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"🔥 [PyTorch Tabular ResNet v3 12-Epochs 심층 집중 학습] Device: {device}")
    print("=" * 70)

    # 1. 데이터 로드
    train_path = "data/train.csv"
    train_df = pd.read_csv(train_path)
    print(f"[1] 데이터 로드 완료: {len(train_df):,}건")

    # 2. 피처 생성
    print("[2] 특성 공학 전처리 파이프라인 수행 중...")
    df_feat = engineer_features(train_df)

    sample_weights = (1.0 + 0.15 * (df_feat["season"].values - 2019).clip(min=0)).astype(np.float32)

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
            emb_dim = min(32, int(round(1.6 * (num_cats ** 0.56))))
        emb_dims.append((num_cats, emb_dim))

    num_cols = [c for c in df_feat.columns if c not in DROP_COLS and c not in CATEGORICAL_COLS]
    print(f" - 범주형 9개(선수 ID 포함) + 수치형 {len(num_cols)}개 피처 확정")

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

    BATCH_SIZE = 2048
    EPOCHS = 12  # 12 에폭 집중 학습!

    print("\n" + "=" * 70)
    print(f"🏆 [3] 5-Fold PyTorch Tabular ResNet v3 심층 학습 시작 ({EPOCHS} Epochs per Fold)")
    print("=" * 70)

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
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False)

        model = TabularResNetV3(emb_dims, len(num_cols), hidden_dim=256, dropout=0.12).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-5)

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
                loss = label_smoothed_weighted_mse_loss(preds, targets, weights, smoothing=0.04)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
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
            
            is_best = ""
            if val_bss > best_val_bss:
                best_val_bss = val_bss
                best_val_preds = fold_val_preds
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                is_best = " ⭐ [NEW BEST!]"
            
            print(f"  - Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {np.mean(train_losses):.5f} | Val BS (MSE): {val_bs:.6f} | Val BSS: {val_bss:.2f}점{is_best}")

        oof_preds[val_idx] = best_val_preds
        fold_model_path = os.path.join(model_dir, f"nn_v3_fold_{fold}.pt")
        torch.save(best_state, fold_model_path)
        fold_elapsed = time.time() - fold_start
        print(f" [Fold {fold+1} 완료] 최고 BSS: {best_val_bss:.2f}점 (Best Epoch {best_epoch}, 저장: {fold_model_path}, 소요: {fold_elapsed/60:.2f}분)")

    # 6. 전체 OOF 종합 평가
    r_all = float(y_array.mean())
    ref_bs_all = r_all * (1.0 - r_all)
    final_bs = np.mean((oof_preds - y_array) ** 2)
    final_bss = max(0.0, 100000.0 * (1.0 - final_bs / ref_bs_all))

    print("\n" + "=" * 70)
    print("🏆 [결과] PyTorch Tabular ResNet v3 심층 12-Epochs 5-Fold OOF 성과")
    print("=" * 70)
    print(f" - 전체 5-Fold OOF BSS 점수 : {final_bss:.2f}점")
    print(f" - 전체 Brier Score (MSE)   : {final_bs:.6f}")
    print(f" - 전체 학습 소요 시간      : {(time.time() - start_total_time)/60:.2f}분")
    print("=" * 70)

    # 7. 메타데이터 및 OOF 저장
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
    joblib.dump(meta, os.path.join(model_dir, "metadata_nn_v3.pkl"))
    
    oof_df = pd.DataFrame({
        "row_id": train_df[ID_COL],
        "control_success": y_array,
        "pred_nn_v3": oof_preds
    })
    oof_df.to_csv(os.path.join(model_dir, "oof_nn_v3.csv"), index=False)
    print(f"[저장 완료] OOF 결과: {os.path.join(model_dir, 'oof_nn_v3.csv')}")
    print(f"[저장 완료] 메타데이터: {os.path.join(model_dir, 'metadata_nn_v3.pkl')}")


if __name__ == "__main__":
    main()
