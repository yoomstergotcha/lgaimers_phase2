# -*- coding: utf-8 -*-
"""
초압축(Slim) v1 + v4 앙상블 모델 패키징 스크립트
1. NN v1 Slim (5-Fold) + NN v4 Slim (5-Fold) 가중 앙상블
2. 단독 추론 스크립트 script.py 생성
3. nn_ens_pr_slim_70_30.zip 및 nn_ens_pr_slim_65_35.zip 생성
4. 압축 파일 무결성 및 모의 추론 검증
"""
import os
import sys
import time
import shutil
import zipfile
import subprocess
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


SCRIPT_TEMPLATE = '''# -*- coding: utf-8 -*-
"""
KBO 제구 성공 확률 예측 AI - 초압축(Slim) v1 + v4 앙상블 추론 스크립트
- v1 Slim ({w_v1:.2f}) + v4 Slim ({w_v4:.2f}) 가중 결합
- 다중공선성 완전 제거 초압축 41개 / 43개 피처 파이프라인
"""
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(min(16, os.cpu_count() or 4))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID_COL = "row_id"
TARGET_COL = "control_success"

BASE_AND_PRUNED_DROP = [
    "row_id", "control_success", "season",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "is_scoring_position", "is_winning", "is_losing", "is_late_inning",
    "is_empty_bases", "is_high_leverage", "is_clutch_score", "is_bases_loaded",
    "asof_pitcher_pitchmix_n", "away_win_expectancy", "game_month", "game_dayofweek",
    "dayofweek_sin", "dayofweek_cos", "score_diff_home", "is_first_pitch", "outs_before",
]

SLIM_EXTRA_DROP = [
    "run_total_before", "count_sum", "asof_pitcher_fastball_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_pitcher_reverse_rate",
    "month_cos", "abs_score_diff",
]

ALL_DROP_COLS_SLIM = BASE_AND_PRUNED_DROP + SLIM_EXTRA_DROP


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


def predict_model_family(model_dir, feat_fn, df_raw, cat_cols):
    cat_encoders = joblib.load(os.path.join(model_dir, "cat_encoders.pkl"))
    scalers = joblib.load(os.path.join(model_dir, "scalers.pkl"))
    medians = joblib.load(os.path.join(model_dir, "medians.pkl"))
    num_cols = joblib.load(os.path.join(model_dir, "num_cols.pkl"))
    emb_dims = joblib.load(os.path.join(model_dir, "emb_dims.pkl"))

    df_feat = feat_fn(df_raw)

    for c in cat_cols:
        c_map = cat_encoders[c]
        default_val = c_map.get("MISSING", 0)
        df_feat[c] = df_feat[c].astype(str).map(lambda v: c_map.get(v, default_val)).astype(np.int64)

    for c in num_cols:
        df_feat[c] = df_feat[c].fillna(medians[c])

    cat_tensor = torch.tensor(df_feat[cat_cols].values, dtype=torch.long).to(device)
    num_raw = df_feat[num_cols].values.astype(np.float32)

    fold_preds = []
    for fold in range(5):
        scaler = scalers[fold]
        num_s = scaler.transform(num_raw)
        num_tensor = torch.tensor(num_s, dtype=torch.float32).to(device)

        model = TabularResNet(emb_dims, len(num_cols), hidden_dim=256, dropout=0.15).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, f"model_fold_{{fold}}.pt"), map_location=device))
        model.eval()

        with torch.no_grad():
            preds = model(cat_tensor, num_tensor).cpu().numpy()
            fold_preds.append(preds)

    return np.mean(fold_preds, axis=0)


def main():
    start_time = time.time()
    test_path = "data/test.csv"
    if not os.path.exists(test_path):
        print(f"Error: {{test_path}} not found.")
        sys.exit(1)

    df_test = pd.read_csv(test_path)
    cat_cols = [
        "pitcher_id", "batter_id", "top_bottom", "game_type", "base_state",
        "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"
    ]

    p_v1 = predict_model_family("model/nn_v1_slim", engineer_features_v1_slim, df_test, cat_cols)
    p_v4 = predict_model_family("model/nn_v4_slim", engineer_features_v4_slim, df_test, cat_cols)

    p_final = {w_v1:.4f} * p_v1 + {w_v4:.4f} * p_v4
    p_final = np.clip(p_final, 0.0, 1.0)

    os.makedirs("output", exist_ok=True)
    sub = pd.DataFrame({{ID_COL: df_test[ID_COL], TARGET_COL: p_final}})
    sub.to_csv("output/submission.csv", index=False)

    elapsed = time.time() - start_time
    print(f"[추론 완료] {{len(sub):,}}건 예측 | 평균: {{np.mean(p_final):.5f}} | 표준편차: {{np.std(p_final):.5f}} | 소요: {{elapsed:.2f}}초")


if __name__ == "__main__":
    main()
'''


def package_ensemble(w_v1=0.70, w_v4=0.30, output_zip_name="nn_ens_pr_slim_70_30.zip"):
    print("\n" + "=" * 75)
    print(f"📦 [Slim 앙상블 패키징] v1 Slim ({w_v1*100:.1f}%) + v4 Slim ({w_v4*100:.1f}%) ➔ {output_zip_name}")
    print("=" * 75)

    work_dir = "temp_slim_pkg"
    os.makedirs(work_dir, exist_ok=True)

    # 1. 모델 디렉토리 복사
    os.makedirs(os.path.join(work_dir, "model"), exist_ok=True)
    shutil.copytree("model/nn_v1_slim", os.path.join(work_dir, "model/nn_v1_slim"), dirs_exist_ok=True)
    shutil.copytree("model/nn_v4_slim", os.path.join(work_dir, "model/nn_v4_slim"), dirs_exist_ok=True)

    # 2. requirements.txt 복사
    if os.path.exists("requirements.txt"):
        shutil.copy("requirements.txt", os.path.join(work_dir, "requirements.txt"))

    # 3. script.py 생성
    script_content = SCRIPT_TEMPLATE.format(w_v1=w_v1, w_v4=w_v4)
    with open(os.path.join(work_dir, "script.py"), "w", encoding="utf-8") as f:
        f.write(script_content)

    # 4. 단독 실행 테스트 (모의 추론)
    print(" - 단독 실행 모의 추론 검증 시작...")
    os.makedirs(os.path.join(work_dir, "data"), exist_ok=True)
    shutil.copy("data/test.csv", os.path.join(work_dir, "data/test.csv"))

    t0 = time.time()
    res = subprocess.run([sys.executable, "script.py"], cwd=work_dir, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    t1 = time.time()

    if res.returncode != 0:
        print(f"❌ 추론 실행 실패! Stderr:\n{res.stderr}")
        shutil.rmtree(work_dir, ignore_errors=True)
        return False

    print(f" - 추론 성공! Stdout: {res.stdout.strip()}")
    print(f" - 추론 소요 시간: {t1 - t0:.2f}초")

    # 5. submission.csv 검증
    sub_path = os.path.join(work_dir, "output/submission.csv")
    sub = pd.read_csv(sub_path)
    print(f" - 생성된 submission.csv 확인: {len(sub):,}행 | 예측 평균: {sub['control_success'].mean():.5f} | 표준편차: {sub['control_success'].std():.5f}")

    # data 디렉토리 및 output 정리 후 zip 생성
    shutil.rmtree(os.path.join(work_dir, "data"), ignore_errors=True)
    shutil.rmtree(os.path.join(work_dir, "output"), ignore_errors=True)

    with zipfile.ZipFile(output_zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(work_dir):
            for file in files:
                abs_p = os.path.join(root, file)
                rel_p = os.path.relpath(abs_p, work_dir)
                zf.write(abs_p, rel_p)

    shutil.rmtree(work_dir, ignore_errors=True)
    size_mb = os.path.getsize(output_zip_name) / (1024 * 1024)
    print(f"🌟 [패키징 완료] {output_zip_name} ({size_mb:.2f} MB)")
    return True


if __name__ == "__main__":
    package_ensemble(w_v1=0.60, w_v4=0.40, output_zip_name="nn_ens_pr_slim_60_40.zip")
    package_ensemble(w_v1=0.65, w_v4=0.35, output_zip_name="nn_ens_pr_slim_65_35.zip")
    package_ensemble(w_v1=0.70, w_v4=0.30, output_zip_name="nn_ens_pr_slim_70_30.zip")
