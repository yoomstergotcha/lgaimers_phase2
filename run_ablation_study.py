# -*- coding: utf-8 -*-
"""
엄격한 가설 기반 어블레이션 실험 러너 (Ablation Study Runner)
- 검증 셋: 2024년 시즌 단독 검증 (Time-Series Split: 2019~2023 학습 -> 2024 검증)
- 목적:
  1. season_progression 제거 vs is_abs_era + 비선형 월 피처 (Exp 1)
  2. Pure Weighted MSE vs Label Smoothed MSE (Exp 2)
  3. 선수 ID 직접 임베딩 유무 (Exp 3)
  4. SE Block 피처 게이팅 유무 (Exp 4)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

seed_everything(SEED)

ID_COL = "row_id"
TARGET_COL = "control_success"

META_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

DROP_BASE = [
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

def build_features(df: pd.DataFrame, mode: str = "v1") -> pd.DataFrame:
    """
    mode options:
      - 'v1': season_progression 포함 (선형 외삽 버전)
      - 'v4_clean': season_progression 삭제, is_abs_era 추가, month_ratio + month_ratio_sq + is_summer 추가
    """
    feat = df.copy()

    if mode == "v1":
        if "season" in feat.columns:
            feat["season_offset"] = feat["season"] - 2019
            if "game_month" in feat.columns:
                feat["season_progression"] = (feat["season"] - 2019) + (feat["game_month"] - 3).clip(lower=0) / 9.0
                feat["month_sin"] = np.sin(2 * np.pi * feat["game_month"] / 12.0)
                feat["month_cos"] = np.cos(2 * np.pi * feat["game_month"] / 12.0)
            if "game_dayofweek" in feat.columns:
                feat["dayofweek_sin"] = np.sin(2 * np.pi * feat["game_dayofweek"] / 7.0)
                feat["dayofweek_cos"] = np.cos(2 * np.pi * feat["game_dayofweek"] / 7.0)
    elif mode == "v4_clean":
        if "season" in feat.columns:
            feat["is_abs_era"] = (feat["season"] >= 2024).astype(np.float32)
            if "game_month" in feat.columns:
                # 0.0 ~ 1.0 범위로 고정 (연도 누적 선형 증가 완전 제거)
                m_ratio = (feat["game_month"] - 3).clip(lower=0) / 7.0
                feat["month_ratio"] = m_ratio
                feat["month_ratio_sq"] = m_ratio ** 2  # 한여름 U자형 볼록 곡률
                feat["is_summer"] = feat["game_month"].isin([7, 8]).astype(np.float32)
                feat["month_sin"] = np.sin(2 * np.pi * feat["game_month"] / 12.0)
                feat["month_cos"] = np.cos(2 * np.pi * feat["game_month"] / 12.0)
            if "game_dayofweek" in feat.columns:
                feat["dayofweek_sin"] = np.sin(2 * np.pi * feat["game_dayofweek"] / 7.0)
                feat["dayofweek_cos"] = np.cos(2 * np.pi * feat["game_dayofweek"] / 7.0)

    # 공통 야구 도메인 파생 피처
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


class SqueezeExcitation(nn.Module):
    def __init__(self, num_features, reduction=4):
        super().__init__()
        reduced_dim = max(8, num_features // reduction)
        self.fc = nn.Sequential(
            nn.Linear(num_features, reduced_dim),
            nn.SiLU(),
            nn.Linear(reduced_dim, num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class TabularResNetModel(nn.Module):
    def __init__(self, emb_dims, num_features, hidden_dim=256, dropout=0.15, use_se=False, act_type="relu"):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cats, emb_dim) for num_cats, emb_dim in emb_dims
        ])
        total_emb_dim = sum(emb_dim for _, emb_dim in emb_dims)
        self.num_bn = nn.BatchNorm1d(num_features)
        self.use_se = use_se
        if use_se:
            self.se_gate = SqueezeExcitation(num_features)

        self.act_fn = nn.ReLU() if act_type == "relu" else nn.SiLU()

        in_dim = total_emb_dim + num_features
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.act_fn,
            nn.Dropout(dropout)
        )

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            self.act_fn,
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outs, dim=1) if emb_outs else torch.empty(len(x_num), 0, device=x_num.device)
        
        x_num_norm = self.num_bn(x_num)
        if self.use_se:
            x_num_norm = self.se_gate(x_num_norm)

        x = torch.cat([x_emb, x_num_norm], dim=1)
        x = self.input_layer(x)
        x = self.act_fn(x + self.block1(x))
        x = self.act_fn(x + self.block2(x))
        return self.head(x).squeeze(-1)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    bs = np.mean((y_pred - y_true) ** 2)
    r = np.mean(y_true)
    ref_bs = r * (1.0 - r)
    bss = max(0.0, 100000.0 * (1.0 - (bs / ref_bs)))
    pred_mean = np.mean(y_pred)
    bias_sq = (pred_mean - r) ** 2
    bias_penalty_bss = 100000.0 * (bias_sq / ref_bs)
    return bss, bs, pred_mean, r, bias_penalty_bss


def run_experiment(exp_name, feat_mode, loss_type, use_player_id, use_se, act_type, df_raw, epochs=6, batch_size=4096):
    print("\n" + "=" * 75)
    print(f"🔬 [실험 실행] {exp_name}")
    print(f" - Feat: {feat_mode} | Loss: {loss_type} | PlayerID: {use_player_id} | SE: {use_se} | Act: {act_type}")
    print("=" * 75)

    seed_everything(SEED)
    df_feat = build_features(df_raw, mode=feat_mode)

    # Train / Val Split (시계열 2024 검증)
    train_mask = (df_feat["season"] < 2024).values
    val_mask = (df_feat["season"] == 2024).values

    # 최근 시즌 가중치
    sample_weights = (1.0 + 0.15 * (df_feat["season"].values - 2019).clip(min=0)).astype(np.float32)

    cat_cols = list(META_CAT_COLS)
    if use_player_id:
        cat_cols = ["pitcher_id", "batter_id"] + cat_cols

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

    num_cols = [c for c in df_feat.columns if c not in DROP_BASE and c not in cat_cols and c != "season"]

    medians = df_feat.loc[train_mask, num_cols].median().to_dict()
    for c in num_cols:
        df_feat[c] = df_feat[c].fillna(medians[c])

    cat_arr = df_feat[cat_cols].values
    num_raw_arr = df_feat[num_cols].values.astype(np.float32)
    y_arr = df_feat[TARGET_COL].values.astype(np.float32)

    scaler = StandardScaler()
    num_train = scaler.fit_transform(num_raw_arr[train_mask])
    num_val = scaler.transform(num_raw_arr[val_mask])

    train_ds = TabularDataset(
        torch.tensor(cat_arr[train_mask], dtype=torch.long),
        torch.tensor(num_train, dtype=torch.float32),
        torch.tensor(y_arr[train_mask], dtype=torch.float32),
        torch.tensor(sample_weights[train_mask], dtype=torch.float32)
    )
    val_ds = TabularDataset(
        torch.tensor(cat_arr[val_mask], dtype=torch.long),
        torch.tensor(num_val, dtype=torch.float32),
        torch.tensor(y_arr[val_mask], dtype=torch.float32),
        torch.tensor(sample_weights[val_mask], dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    model = TabularResNetModel(emb_dims, len(num_cols), hidden_dim=256, dropout=0.15, use_se=use_se, act_type=act_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_bss = -9999.0
    best_bs = 999.0
    best_pred_mean = 0.5
    best_preds = None
    best_epoch = 0

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for cats, nums, targets, weights in train_loader:
            optimizer.zero_grad()
            preds = model(cats, nums)
            if loss_type == "pure_mse":
                loss = (weights * ((preds - targets) ** 2)).mean()
            elif loss_type == "smooth_mse":
                targets_smooth = targets * 0.96 + 0.02
                loss = (weights * ((preds - targets_smooth) ** 2)).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
        
        scheduler.step()
        train_loss /= len(train_ds)

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for cats, nums, targets, weights in val_loader:
                preds = model(cats, nums)
                val_preds_list.append(preds.cpu().numpy())
        
        val_preds = np.concatenate(val_preds_list)
        bss, bs, p_mean, y_mean, bias_pen = compute_metrics(y_arr[val_mask], val_preds)

        if bss > best_bss:
            best_bss = bss
            best_bs = bs
            best_pred_mean = p_mean
            best_preds = val_preds
            best_epoch = ep

        print(f"  [Epoch {ep:02d}/{epochs:02d}] Train Loss: {train_loss:.5f} | Val 2024 BSS: {bss:.2f}점 | MSE: {bs:.6f} | Mean(p): {p_mean:.4f} (True: {y_mean:.4f})")

    elapsed = time.time() - t_start
    bss, bs, p_mean, y_mean, bias_pen = compute_metrics(y_arr[val_mask], best_preds)
    print(f"🏆 [실험 결과 - {exp_name}]")
    print(f" -> Best 2024 Val BSS     : {bss:.2f}점 (Best Epoch: {best_epoch})")
    print(f" -> Brier Score (MSE)     : {bs:.6f}")
    print(f" -> 2024 예측 평균 (p_bar): {p_mean:.5f} (실제 2024 정답률: {y_mean:.5f})")
    print(f" -> Calibration 감점(Bias): -{bias_pen:.2f}점")
    print(f" -> 소요 시간             : {elapsed:.1f}초")

    return {
        "exp_name": exp_name,
        "feat_mode": feat_mode,
        "loss_type": loss_type,
        "use_player_id": use_player_id,
        "use_se": use_se,
        "act_type": act_type,
        "val_2024_bss": bss,
        "val_2024_bs": bs,
        "pred_mean_2024": p_mean,
        "true_mean_2024": y_mean,
        "bias_penalty_bss": bias_pen,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "preds": best_preds
    }


def main():
    print("=" * 80)
    print("🚀 [엄격한 가설 검증] PyTorch Tabular ResNet 5대 핵심 요인 Ablation Study 시작")
    print("=" * 80)

    train_path = "data/train.csv"
    df_raw = pd.read_csv(train_path)
    print(f"데이터 로드 완료: 전체 {len(df_raw):,}건 (2019~2023: {(df_raw['season'] < 2024).sum():,}건, 2024: {(df_raw['season'] == 2024).sum():,}건)")

    results = []

    # 1. Exp 0: Baseline v1 원본 (season_progression 포함, pure mse, ReLU, no ID, no SE)
    res_0 = run_experiment("Exp 0: Baseline v1 (원본 선형외삽)", feat_mode="v1", loss_type="pure_mse", use_player_id=False, use_se=False, act_type="relu", df_raw=df_raw, epochs=6)
    results.append(res_0)

    # 2. Exp 1: 피처 개선 v4_clean (외삽 삭제 + is_abs_era + 2차 비선형 월 피처)
    res_1 = run_experiment("Exp 1: v4_clean 피처 (외삽삭제+ABS단절+볼록월)", feat_mode="v4_clean", loss_type="pure_mse", use_player_id=False, use_se=False, act_type="relu", df_raw=df_raw, epochs=6)
    results.append(res_1)

    # 3. Exp 2: 라벨 스무딩의 영향 검증 (v4_clean 피처 + Label Smoothing 0.04)
    res_2 = run_experiment("Exp 2: Label Smoothing (0.04) 비교", feat_mode="v4_clean", loss_type="smooth_mse", use_player_id=False, use_se=False, act_type="relu", df_raw=df_raw, epochs=6)
    results.append(res_2)

    # 4. Exp 3: 선수 고유 ID 직접 임베딩 영향 (v4_clean 피처 + Player ID 24-dim)
    res_3 = run_experiment("Exp 3: 선수 ID 직접 임베딩 (24차원)", feat_mode="v4_clean", loss_type="pure_mse", use_player_id=True, use_se=False, act_type="relu", df_raw=df_raw, epochs=6)
    results.append(res_3)

    # 5. Exp 4: SE Block 피처 게이팅 영향 (v4_clean 피처 + SE Module)
    res_4 = run_experiment("Exp 4: SE Block (피처 게이팅) 비교", feat_mode="v4_clean", loss_type="pure_mse", use_player_id=False, use_se=True, act_type="silu", df_raw=df_raw, epochs=6)
    results.append(res_4)

    # 6. Exp 5: v3 전체 조합 (v4_clean + ID 임베딩 + SE + SiLU + 라벨스무딩)
    res_5 = run_experiment("Exp 5: v3 복합 아키텍처 (ID+SE+SiLU+스무딩)", feat_mode="v4_clean", loss_type="smooth_mse", use_player_id=True, use_se=True, act_type="silu", df_raw=df_raw, epochs=6)
    results.append(res_5)

    print("\n" + "=" * 80)
    print("📊 [최종 종합 결과표] 2024 시계열 단독 검증 Ablation Study 비교")
    print("=" * 80)
    summary_df = pd.DataFrame([{
        "실험명": r["exp_name"],
        "2024 검증 BSS": f"{r['val_2024_bss']:.2f}점",
        "MSE (Brier)": f"{r['val_2024_bs']:.6f}",
        "2024 예측 평균": f"{r['pred_mean_2024']:.4f}",
        "편향 감점": f"-{r['bias_penalty_bss']:.2f}점",
        "소요시간": f"{r['elapsed_sec']:.1f}초"
    } for r in results])
    print(summary_df.to_string(index=False))

    summary_df.to_csv("model/ablation_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] 결과 저장: model/ablation_summary.csv")


if __name__ == "__main__":
    main()
