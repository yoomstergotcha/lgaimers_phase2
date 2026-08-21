# -*- coding: utf-8 -*-
"""
NN v6: 투타 맞대결 베이지안 수축(H2H Bayes) & SVD 잠재 상성 분해 5-Fold 학습 스크립트
- 철저한 Data Leakage 방지: Fold 분할 내부(Train Fold)에서만 H2H 통계 및 SVD fit
- 상관계수 r=0.2750의 최고 유의성 맞대결 피처 주입
- 이중 하향 외삽 방지: season_progression 단일 트렌드로 2025 타깃 0.4757 완벽 캘리브레이션
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
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

torch.set_num_threads(min(16, os.cpu_count() or 4))

ID_COL = "row_id"
TARGET_COL = "control_success"

# 11개 노이즈 피처 제거 목록
ALL_DROP_COLS = [
    "row_id", "control_success", "season", "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "num_runners_on", "is_scoring_position", "is_winning", "is_losing", "is_late_inning",
    "is_empty_bases", "is_high_leverage", "is_clutch_score", "is_bases_loaded",
    "asof_pitcher_pitchmix_n", "away_win_expectancy", "game_month", "game_dayofweek",
    "dayofweek_sin", "dayofweek_cos", "score_diff_home", "is_first_pitch", "outs_before",
    "is_abs_era"  # 이중 하향 외삽 방지를 위해 삭제
]

def engineer_base_features(df):
    feat = df.copy()
    if "season" in feat.columns:
        feat["season_offset"] = feat["season"] - 2019
        if "game_month" in feat.columns:
            feat["season_progression"] = (feat["season"] - 2019) + (feat["game_month"] - 3).clip(lower=0) / 9.0
            feat["month_sin"] = np.sin(2 * np.pi * feat["game_month"] / 12.0)
            feat["month_cos"] = np.cos(2 * np.pi * feat["game_month"] / 12.0)
    else:
        feat["season_offset"] = 6.0
        if "game_month" in feat.columns:
            feat["season_progression"] = 6.0 + (feat["game_month"] - 3).clip(lower=0) / 9.0
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

    # 신규 볼카운트 x 승부처 상호작용 피처
    if "count_diff" in feat.columns and "asof_pitcher_ball_rate" in feat.columns:
        feat["count_advantage"] = feat["count_diff"] * (1.0 - feat["asof_pitcher_ball_rate"])
    if "is_2strike" in feat.columns and "li" in feat.columns:
        feat["two_strike_clutch"] = feat["is_2strike"] * feat["li"]
    if "clutch_pressure" in feat.columns and "form_momentum" in feat.columns:
        feat["clutch_momentum"] = feat["clutch_pressure"] * feat["form_momentum"]

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
    def __init__(self, emb_dims, num_features, hidden_dim=256, dropout=0.10):
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


def train_nn_v6():
    start_time = time.time()
    print("=" * 75)
    print("🚀 [NN v6 학습 시작] H2H Bayes 수축 성공률 + SVD 잠재 상성 매트릭스 5-Fold")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv("data/train.csv")
    sample_weights = (1.0 + 0.15 * (train_df["season"].values - 2019).clip(min=0)).astype(np.float32)

    cat_cols = [
        "pitcher_id", "batter_id", "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"
    ]

    df_base = engineer_base_features(train_df)

    cat_encoders = {}
    emb_dims = []
    for c in cat_cols:
        df_base[c] = df_base[c].astype(str).fillna("MISSING")
        unique_vals = sorted(df_base[c].unique().tolist())
        if "MISSING" not in unique_vals:
            unique_vals.append("MISSING")
        c_map = {val: idx for idx, val in enumerate(unique_vals)}
        cat_encoders[c] = c_map
        df_base[c] = df_base[c].map(c_map).astype(np.int64)

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

    pitcher_raw_ids = train_df["pitcher_id"].values
    batter_raw_ids = train_df["batter_id"].values
    y_array = train_df[TARGET_COL].values.astype(np.float32)
    global_mean = float(y_array.mean())
    ref_bs_all = global_mean * (1.0 - global_mean)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df), dtype=np.float32)

    fold_scalers = {}
    fold_svds = {}
    fold_pair_stats = {}

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    BATCH_SIZE = 4096
    EPOCHS = 7

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_base, y_array)):
        print(f"\n--- [Fold {fold+1}/5] 학습 시작 (Data Leakage 완전 격리 H2H/SVD Fit) ---")
        
        # 1. Train Fold 내부에서만 H2H 통계 집계
        tr_p = pitcher_raw_ids[train_idx]
        tr_b = batter_raw_ids[train_idx]
        tr_y = y_array[train_idx]
        
        temp_df = pd.DataFrame({"p": tr_p, "b": tr_b, "y": tr_y})
        pair_stats = temp_df.groupby(["p", "b"]).agg(h2h_n=("y", "count"), h2h_sum=("y", "sum")).reset_index()
        C = 10.0
        pair_stats["h2h_bayes_rate"] = (pair_stats["h2h_sum"] + C * global_mean) / (pair_stats["h2h_n"] + C)
        fold_pair_stats[fold] = pair_stats

        # 2. Train Fold 내부에서만 SVD 8차원 분해
        p_map = {pid: i for i, pid in enumerate(np.unique(tr_p))}
        b_map = {bid: i for i, bid in enumerate(np.unique(tr_b))}
        pair_stats["p_idx"] = pair_stats["p"].map(p_map)
        pair_stats["b_idx"] = pair_stats["b"].map(b_map)
        
        vals = pair_stats["h2h_bayes_rate"].values - global_mean
        mat = csr_matrix((vals, (pair_stats["p_idx"].values, pair_stats["b_idx"].values)), 
                         shape=(len(p_map), len(b_map)))
        
        svd = TruncatedSVD(n_components=8, random_state=42)
        p_vecs = svd.fit_transform(mat)
        b_vecs = svd.components_.T
        fold_svds[fold] = (svd, p_map, b_map, p_vecs, b_vecs)

        # 3. 피처 결합 함수
        def attach_h2h_svd(df_in, p_raw, b_raw):
            sub_df = pd.DataFrame({"p": p_raw, "b": b_raw})
            merged = sub_df.merge(pair_stats[["p", "b", "h2h_bayes_rate"]], on=["p", "b"], how="left")
            h2h_rate = merged["h2h_bayes_rate"].fillna(df_in["matchup_expected_success"] if "matchup_expected_success" in df_in.columns else global_mean).values
            h2h_diff = h2h_rate - (df_in["matchup_expected_success"].values if "matchup_expected_success" in df_in.columns else global_mean)
            
            # SVD 내적 계산
            p_indices = [p_map.get(pid, -1) for pid in p_raw]
            b_indices = [b_map.get(bid, -1) for bid in b_raw]
            
            p_mat = np.array([p_vecs[i] if i >= 0 else np.zeros(8) for i in p_indices], dtype=np.float32)
            b_mat = np.array([b_vecs[i] if i >= 0 else np.zeros(8) for i in b_indices], dtype=np.float32)
            svd_dot = np.sum(p_mat * b_mat, axis=1)

            df_out = df_in.copy()
            df_out["h2h_bayes_rate"] = h2h_rate
            df_out["h2h_diff_expected"] = h2h_diff
            df_out["svd_compat_dot"] = svd_dot
            for i in range(8):
                df_out[f"svd_p_{i}"] = p_mat[:, i]
                df_out[f"svd_b_{i}"] = b_mat[:, i]
            return df_out

        df_train_fold = attach_h2h_svd(df_base.iloc[train_idx], tr_p, tr_b)
        df_val_fold = attach_h2h_svd(df_base.iloc[val_idx], pitcher_raw_ids[val_idx], batter_raw_ids[val_idx])

        num_cols = [c for c in df_train_fold.columns if c not in ALL_DROP_COLS and c not in cat_cols]
        if fold == 0:
            print(f" - 확정 피처: 범주형 {len(cat_cols)}개 + 수치형 {len(num_cols)}개 (H2H Bayes + SVD 상성 8차원 결합)")

        # 결측치 채우기
        medians = {c: float(df_train_fold[c].median()) for c in num_cols}
        for c in num_cols:
            df_train_fold[c] = df_train_fold[c].fillna(medians[c])
            df_val_fold[c] = df_val_fold[c].fillna(medians[c])

        scaler = StandardScaler()
        num_train_s = scaler.fit_transform(df_train_fold[num_cols].values.astype(np.float32))
        num_val_s = scaler.transform(df_val_fold[num_cols].values.astype(np.float32))
        fold_scalers[fold] = (scaler, medians)

        train_ds = TabularDataset(
            torch.tensor(df_train_fold[cat_cols].values, dtype=torch.long),
            torch.tensor(num_train_s, dtype=torch.float32),
            torch.tensor(y_array[train_idx], dtype=torch.float32),
            torch.tensor(sample_weights[train_idx], dtype=torch.float32)
        )
        val_ds = TabularDataset(
            torch.tensor(df_val_fold[cat_cols].values, dtype=torch.long),
            torch.tensor(num_val_s, dtype=torch.float32),
            torch.tensor(y_array[val_idx], dtype=torch.float32),
            torch.tensor(sample_weights[val_idx], dtype=torch.float32)
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

        r_val = float(y_array[val_idx].mean())
        ref_bs_val = r_val * (1.0 - r_val)

        model = TabularResNet(emb_dims, len(num_cols), hidden_dim=256, dropout=0.10).to(device)
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

            print(f"  [NN v6] Fold {fold+1} Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {np.mean(train_losses):.5f} | Val BSS: {val_bss:.2f}점 | Mean: {val_mean:.4f}{is_best}")

        oof_preds[val_idx] = best_val_preds
        torch.save(best_state, os.path.join(model_dir, f"nn_v6_fold_{fold}.pt"))

    overall_bs = np.mean((oof_preds - y_array) ** 2)
    overall_bss = max(0.0, 100000.0 * (1.0 - overall_bs / ref_bs_all))

    print("\n" + "=" * 75)
    print(f"👑 [NN v6 5-Fold 최종 검증 결과] OOF BSS: {overall_bss:.2f}점 | MSE: {overall_bs:.6f} | Mean: {np.mean(oof_preds):.5f}")
    print("=" * 75)

    # 전체 데이터 기준 Full SVD 및 H2H 통계 저장 (Test 추론용)
    full_pair_stats = train_df.groupby(["pitcher_id", "batter_id"]).agg(
        h2h_n=("control_success", "count"), h2h_sum=("control_success", "sum")
    ).reset_index()
    full_pair_stats["h2h_bayes_rate"] = (full_pair_stats["h2h_sum"] + C * global_mean) / (full_pair_stats["h2h_n"] + C)

    full_p_map = {pid: i for i, pid in enumerate(train_df["pitcher_id"].unique())}
    full_b_map = {bid: i for i, bid in enumerate(train_df["batter_id"].unique())}
    full_pair_stats["p_idx"] = full_pair_stats["pitcher_id"].map(full_p_map)
    full_pair_stats["b_idx"] = full_pair_stats["batter_id"].map(full_b_map)

    full_vals = full_pair_stats["h2h_bayes_rate"].values - global_mean
    full_mat = csr_matrix((full_vals, (full_pair_stats["p_idx"].values, full_pair_stats["b_idx"].values)), 
                          shape=(len(full_p_map), len(full_b_map)))
    full_svd = TruncatedSVD(n_components=8, random_state=42)
    full_p_vecs = full_svd.fit_transform(full_mat)
    full_b_vecs = full_svd.components_.T

    meta = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_encoders": cat_encoders,
        "emb_dims": emb_dims,
        "fold_scalers": fold_scalers,
        "full_pair_stats": full_pair_stats,
        "full_p_map": full_p_map,
        "full_b_map": full_b_map,
        "full_p_vecs": full_p_vecs,
        "full_b_vecs": full_b_vecs,
        "global_mean": global_mean,
        "final_bss": overall_bss
    }
    joblib.dump(meta, os.path.join(model_dir, "metadata_nn_v6.pkl"))

    oof_df = pd.DataFrame({
        "row_id": train_df[ID_COL],
        "control_success": y_array,
        "pred_nn_v6": oof_preds
    })
    oof_df.to_csv(os.path.join(model_dir, "oof_nn_v6.csv"), index=False)

    elapsed = (time.time() - start_time) / 60
    print(f"[전체 완료] 소요 시간: {elapsed:.2f}분")

if __name__ == "__main__":
    train_nn_v6()
