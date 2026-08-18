#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_pipeline_v4_notebook_aligned.py
=====================================

Notebook-aligned, leakage-safe implementation of the revised model structure.

Source of truth for raw columns / preprocessing / Base263
---------------------------------------------------------
current_submission_2024_validation_temporal_ensemble.ipynb

This script preserves the notebook's:
- project layout: <project>/data/{train.csv,test.csv,sample_submission.csv,trackman_history.csv}
- target: control_success / id: row_id
- Base263 feature order and 11 categorical features
- base situation features
- reconstructed pitch flags
- prior-season Flat history
- Drift / Context Effect / Safe Context
- Hierarchical Logit
- Current Season decomposition
- walk-forward Latent Pitcher Skill
- CatBoost baseline hyperparameters and fixed 558-tree submission convention

Revised-model additions
-----------------------
1) 2024 locked validation protocol:
     dev: 2022 + 2023 only for feature/architecture/epoch screening
     locked: 2024 exactly once for final candidate/blend decision
     final: train through 2024 -> predict 2025
2) Exact Prev1 / Prev2 / Prev3 season profiles derived from reconstructed train flags.
3) CatBoost ablations:
     T0_BASE
     T1_ABS_LEVEL
     T2_REL_LEVEL
     T3_REL_TREND
     T4_REL_LEVEL_PLUS_TREND
     T5_RELIABILITY_ONLY
     T6_LEVEL_ONLY_CONTROL
4) Structured Transformer state tokens from the actual notebook feature universe:
     6 contexts x {Career, Prev3, Prev2, Prev1} = 24 historical tokens.
   Current/Recent information is placed in the CURRENT QUERY token rather than
   fabricating unavailable full Context x Time cells.
5) CURRENT QUERY token instead of generic [CLS].
6) Learned missing-cell embedding, plus n / availability.
7) Same-state-matrix MLP sanity baseline.
8) Fixed 80/20 CatBoost/Transformer first. Only if it improves, evaluate
   coarse 90/10, 80/20, 70/30 and freeze the best coarse weight.
9) Prediction/error correlation + error covariance diagnostics.
10) Raw-file/code/config fingerprints and immutable dev/locked manifests.

Important discrepancy intentionally NOT hidden
----------------------------------------------
The guide document mentions old-F sample weights (old F = 0.25), while the uploaded
current-submission notebook fits the main CatBoost without sample_weight.
To keep T0_BASE truly notebook-reproducible and avoid changing temporal representation
and sample weighting simultaneously, the DEFAULT is sample_weight_mode="none".
The guide weight is available explicitly with:
    --sample-weight-mode guide_old_f

TrackMan is loaded by the notebook but not used by Base263, and the revised V1 guide
explicitly excludes TrackMan from the Transformer. This script therefore fingerprints
trackman_history.csv if present but does not use it as a model input.

Typical usage
-------------
python model_pipeline_v4_notebook_aligned.py --stage prepare --project-dir /Users/chunyoomin/lgaimers --output-dir /Users/chunyoomin/lgaimers/output/model_v4
python model_pipeline_v4_notebook_aligned.py --stage dev     --project-dir /Users/chunyoomin/lgaimers --output-dir /Users/chunyoomin/lgaimers/output/model_v4
python model_pipeline_v4_notebook_aligned.py --stage locked  --project-dir /Users/chunyoomin/lgaimers --output-dir /Users/chunyoomin/lgaimers/output/model_v4
python model_pipeline_v4_notebook_aligned.py --stage final   --project-dir /Users/chunyoomin/lgaimers --output-dir /Users/chunyoomin/lgaimers/output/model_v4

Dependencies
------------
numpy, pandas, scikit-learn, catboost, torch
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Notebook constants / exact Base263 contract
# -----------------------------------------------------------------------------
ID = 'row_id'

TARGET = 'control_success'

VALID_SEASON = 2024

SUBMISSION_FIXED_ITERATIONS = 558

FLAT_ALPHA = 20.0

CONTEXT_EFFECT_ALPHA = 20.0

LOGIT_EPS = 1e-05

CURRENT_SEASON_ALPHA = 100.0

SKILL_OBS_ALPHA = 80.0

SKILL_TARGET_ALPHA = 100.0

SKILL_MIN_FUTURE_PITCHES = 50

SKILL_SNAPSHOT_POSITIONS = (0, 10, 25, 50, 100, 200, 400, 800)

FLAG_COLS = ['flag_success',
 'flag_middle',
 'flag_reverse',
 'flag_ball',
 'flag_strike',
 'flag_fastball',
 'flag_breaking',
 'flag_offspeed']

ASOF_FLAG_MAP = {'flag_success': 'asof_pitcher_success_rate',
 'flag_middle': 'asof_pitcher_middle_rate',
 'flag_reverse': 'asof_pitcher_reverse_rate',
 'flag_ball': 'asof_pitcher_ball_rate',
 'flag_strike': 'asof_pitcher_strike_rate',
 'flag_fastball': 'asof_pitcher_fastball_rate',
 'flag_breaking': 'asof_pitcher_breaking_rate',
 'flag_offspeed': 'asof_pitcher_offspeed_rate'}

FLAT_GROUPS = {'league_count': ['count_state'],
 'league_count_hand': ['count_state', 'batter_hand'],
 'pitcher': ['pitcher_id'],
 'pitcher_hand': ['pitcher_id', 'batter_hand'],
 'pitcher_count': ['pitcher_id', 'count_state'],
 'pitcher_count_hand': ['pitcher_id', 'count_state', 'batter_hand'],
 'pitcher_baseout': ['pitcher_id', 'base_out_state'],
 'pitcher_count_baseout': ['pitcher_id', 'count_state', 'base_out_state'],
 'handmatch_count': ['hand_matchup', 'count_state']}

PITCHER_CONTEXTS = ['pitcher_count', 'pitcher_hand', 'pitcher_count_hand', 'pitcher_baseout']

CAT_COLS = ['game_dayofweek',
 'top_bottom',
 'game_type',
 'base_state',
 'pitcher_hand',
 'batter_hand',
 'pitcher_team_id',
 'batter_team_id',
 'count_state',
 'base_out_state',
 'hand_matchup']

SKILL_NUMERIC_COLUMNS = ['asof_pitcher_n',
 'asof_pitcher_success_rate',
 'asof_pitcher_middle_rate',
 'asof_pitcher_reverse_rate',
 'asof_pitcher_ball_rate',
 'asof_pitcher_strike_rate',
 'asof_pitcher_pitchmix_n',
 'asof_pitcher_fastball_rate',
 'asof_pitcher_breaking_rate',
 'asof_pitcher_offspeed_rate',
 'asof_pitcher_prev1_game_success_rate',
 'asof_pitcher_prev3_game_success_rate',
 'asof_pitcher_prev5_game_success_rate',
 'asof_pitcher_prev1_game_middle_rate',
 'asof_pitcher_prev3_game_middle_rate',
 'asof_pitcher_prev5_game_middle_rate',
 'preseason_pitcher_missing']

SKILL_CAT_COLUMNS = ['pitcher_hand']

SKILL_MODEL_PARAMS = {'loss_function': 'RMSE',
 'iterations': 450,
 'learning_rate': 0.035,
 'depth': 5,
 'l2_leaf_reg': 8.0,
 'random_strength': 0.5,
 'random_seed': 42,
 'task_type': 'CPU',
 'thread_count': -1,
 'allow_writing_files': False,
 'verbose': False}

MAIN_MODEL_PARAMS = {'loss_function': 'Logloss',
 'eval_metric': 'BrierScore',
 'iterations': 600,
 'learning_rate': 0.04,
 'depth': 6,
 'l2_leaf_reg': 7.0,
 'random_strength': 1.0,
 'max_ctr_complexity': 1,
 'border_count': 64,
 'random_seed': 42,
 'task_type': 'CPU',
 'thread_count': -1,
 'allow_writing_files': False,
 'verbose': 50}

FEATURES = ['season',
 'game_month',
 'inning',
 'balls_before',
 'strikes_before',
 'outs_before',
 'run_top_before',
 'run_bot_before',
 'run_total_before',
 'score_diff_home',
 'score_diff_pitcher_team',
 'home_win_expectancy',
 'away_win_expectancy',
 'li',
 'asof_pitcher_n',
 'asof_pitcher_success_rate',
 'asof_pitcher_reverse_rate',
 'asof_pitcher_middle_rate',
 'asof_pitcher_ball_rate',
 'asof_pitcher_strike_rate',
 'asof_pitcher_prev1_game_success_rate',
 'asof_pitcher_prev3_game_success_rate',
 'asof_pitcher_prev5_game_success_rate',
 'asof_pitcher_prev1_game_middle_rate',
 'asof_pitcher_prev3_game_middle_rate',
 'asof_pitcher_prev5_game_middle_rate',
 'asof_batter_n',
 'asof_batter_success_rate',
 'asof_pitcher_fastball_rate',
 'asof_pitcher_breaking_rate',
 'asof_pitcher_offspeed_rate',
 'pitcher_is_home',
 'pitcher_team_win_expectancy',
 'pitcher_history_missing',
 'previous_game_history_missing',
 'pitcher_success_prev1_delta',
 'pitcher_success_prev3_delta',
 'pitcher_success_prev5_delta',
 'pitcher_success_count500',
 'pitcher_reverse_count500',
 'pitcher_ball_count500',
 'pitcher_strike_count500',
 'preseason_pitcher_missing',
 'flat_league_count_flag_success_rate',
 'flat_league_count_success_n',
 'flat_league_count_flag_middle_rate',
 'flat_league_count_n',
 'flat_league_count_flag_reverse_rate',
 'flat_league_count_flag_ball_rate',
 'flat_league_count_flag_strike_rate',
 'flat_league_count_flag_fastball_rate',
 'flat_league_count_flag_breaking_rate',
 'flat_league_count_flag_offspeed_rate',
 'flat_league_count_hand_flag_success_rate',
 'flat_league_count_hand_success_n',
 'flat_league_count_hand_flag_middle_rate',
 'flat_league_count_hand_n',
 'flat_league_count_hand_flag_reverse_rate',
 'flat_league_count_hand_flag_ball_rate',
 'flat_league_count_hand_flag_strike_rate',
 'flat_league_count_hand_flag_fastball_rate',
 'flat_league_count_hand_flag_breaking_rate',
 'flat_league_count_hand_flag_offspeed_rate',
 'flat_pitcher_flag_success_rate',
 'flat_pitcher_success_n',
 'flat_pitcher_flag_middle_rate',
 'flat_pitcher_n',
 'flat_pitcher_flag_reverse_rate',
 'flat_pitcher_flag_ball_rate',
 'flat_pitcher_flag_strike_rate',
 'flat_pitcher_flag_fastball_rate',
 'flat_pitcher_flag_breaking_rate',
 'flat_pitcher_flag_offspeed_rate',
 'flat_pitcher_hand_flag_success_rate',
 'flat_pitcher_hand_success_n',
 'flat_pitcher_hand_flag_middle_rate',
 'flat_pitcher_hand_n',
 'flat_pitcher_hand_flag_reverse_rate',
 'flat_pitcher_hand_flag_ball_rate',
 'flat_pitcher_hand_flag_strike_rate',
 'flat_pitcher_hand_flag_fastball_rate',
 'flat_pitcher_hand_flag_breaking_rate',
 'flat_pitcher_hand_flag_offspeed_rate',
 'flat_pitcher_count_flag_success_rate',
 'flat_pitcher_count_success_n',
 'flat_pitcher_count_flag_middle_rate',
 'flat_pitcher_count_n',
 'flat_pitcher_count_flag_reverse_rate',
 'flat_pitcher_count_flag_ball_rate',
 'flat_pitcher_count_flag_strike_rate',
 'flat_pitcher_count_flag_fastball_rate',
 'flat_pitcher_count_flag_breaking_rate',
 'flat_pitcher_count_flag_offspeed_rate',
 'flat_pitcher_count_hand_flag_success_rate',
 'flat_pitcher_count_hand_success_n',
 'flat_pitcher_count_hand_flag_middle_rate',
 'flat_pitcher_count_hand_n',
 'flat_pitcher_count_hand_flag_reverse_rate',
 'flat_pitcher_count_hand_flag_ball_rate',
 'flat_pitcher_count_hand_flag_strike_rate',
 'flat_pitcher_count_hand_flag_fastball_rate',
 'flat_pitcher_count_hand_flag_breaking_rate',
 'flat_pitcher_count_hand_flag_offspeed_rate',
 'flat_pitcher_baseout_flag_success_rate',
 'flat_pitcher_baseout_success_n',
 'flat_pitcher_baseout_flag_middle_rate',
 'flat_pitcher_baseout_n',
 'flat_pitcher_baseout_flag_reverse_rate',
 'flat_pitcher_baseout_flag_ball_rate',
 'flat_pitcher_baseout_flag_strike_rate',
 'flat_pitcher_baseout_flag_fastball_rate',
 'flat_pitcher_baseout_flag_breaking_rate',
 'flat_pitcher_baseout_flag_offspeed_rate',
 'flat_pitcher_count_baseout_flag_success_rate',
 'flat_pitcher_count_baseout_success_n',
 'flat_pitcher_count_baseout_flag_middle_rate',
 'flat_pitcher_count_baseout_n',
 'flat_pitcher_count_baseout_flag_reverse_rate',
 'flat_pitcher_count_baseout_flag_ball_rate',
 'flat_pitcher_count_baseout_flag_strike_rate',
 'flat_pitcher_count_baseout_flag_fastball_rate',
 'flat_pitcher_count_baseout_flag_breaking_rate',
 'flat_pitcher_count_baseout_flag_offspeed_rate',
 'flat_handmatch_count_flag_success_rate',
 'flat_handmatch_count_success_n',
 'flat_handmatch_count_flag_middle_rate',
 'flat_handmatch_count_n',
 'flat_handmatch_count_flag_reverse_rate',
 'flat_handmatch_count_flag_ball_rate',
 'flat_handmatch_count_flag_strike_rate',
 'flat_handmatch_count_flag_fastball_rate',
 'flat_handmatch_count_flag_breaking_rate',
 'flat_handmatch_count_flag_offspeed_rate',
 'pitcher_drift_flag_success',
 'pitcher_drift_flag_middle',
 'pitcher_drift_flag_reverse',
 'pitcher_drift_flag_ball',
 'pitcher_drift_flag_strike',
 'pitcher_drift_flag_fastball',
 'pitcher_drift_flag_breaking',
 'pitcher_drift_flag_offspeed',
 'pitcher_context_effect_pitcher_count_flag_success',
 'pitcher_safe_context_pitcher_count_flag_success_rate',
 'pitcher_context_effect_pitcher_count_flag_middle',
 'pitcher_safe_context_pitcher_count_flag_middle_rate',
 'pitcher_context_effect_pitcher_count_flag_reverse',
 'pitcher_safe_context_pitcher_count_flag_reverse_rate',
 'pitcher_context_effect_pitcher_count_flag_ball',
 'pitcher_safe_context_pitcher_count_flag_ball_rate',
 'pitcher_context_effect_pitcher_count_flag_strike',
 'pitcher_safe_context_pitcher_count_flag_strike_rate',
 'pitcher_context_effect_pitcher_count_flag_fastball',
 'pitcher_safe_context_pitcher_count_flag_fastball_rate',
 'pitcher_context_effect_pitcher_count_flag_breaking',
 'pitcher_safe_context_pitcher_count_flag_breaking_rate',
 'pitcher_context_effect_pitcher_count_flag_offspeed',
 'pitcher_safe_context_pitcher_count_flag_offspeed_rate',
 'pitcher_context_effect_pitcher_hand_flag_success',
 'pitcher_safe_context_pitcher_hand_flag_success_rate',
 'pitcher_context_effect_pitcher_hand_flag_middle',
 'pitcher_safe_context_pitcher_hand_flag_middle_rate',
 'pitcher_context_effect_pitcher_hand_flag_reverse',
 'pitcher_safe_context_pitcher_hand_flag_reverse_rate',
 'pitcher_context_effect_pitcher_hand_flag_ball',
 'pitcher_safe_context_pitcher_hand_flag_ball_rate',
 'pitcher_context_effect_pitcher_hand_flag_strike',
 'pitcher_safe_context_pitcher_hand_flag_strike_rate',
 'pitcher_context_effect_pitcher_hand_flag_fastball',
 'pitcher_safe_context_pitcher_hand_flag_fastball_rate',
 'pitcher_context_effect_pitcher_hand_flag_breaking',
 'pitcher_safe_context_pitcher_hand_flag_breaking_rate',
 'pitcher_context_effect_pitcher_hand_flag_offspeed',
 'pitcher_safe_context_pitcher_hand_flag_offspeed_rate',
 'pitcher_context_effect_pitcher_count_hand_flag_success',
 'pitcher_safe_context_pitcher_count_hand_flag_success_rate',
 'pitcher_context_effect_pitcher_count_hand_flag_middle',
 'pitcher_safe_context_pitcher_count_hand_flag_middle_rate',
 'pitcher_context_effect_pitcher_count_hand_flag_reverse',
 'pitcher_safe_context_pitcher_count_hand_flag_reverse_rate',
 'pitcher_context_effect_pitcher_count_hand_flag_ball',
 'pitcher_safe_context_pitcher_count_hand_flag_ball_rate',
 'pitcher_context_effect_pitcher_count_hand_flag_strike',
 'pitcher_safe_context_pitcher_count_hand_flag_strike_rate',
 'pitcher_context_effect_pitcher_count_hand_flag_fastball',
 'pitcher_safe_context_pitcher_count_hand_flag_fastball_rate',
 'pitcher_context_effect_pitcher_count_hand_flag_breaking',
 'pitcher_safe_context_pitcher_count_hand_flag_breaking_rate',
 'pitcher_context_effect_pitcher_count_hand_flag_offspeed',
 'pitcher_safe_context_pitcher_count_hand_flag_offspeed_rate',
 'pitcher_context_effect_pitcher_baseout_flag_success',
 'pitcher_safe_context_pitcher_baseout_flag_success_rate',
 'pitcher_context_effect_pitcher_baseout_flag_middle',
 'pitcher_safe_context_pitcher_baseout_flag_middle_rate',
 'pitcher_context_effect_pitcher_baseout_flag_reverse',
 'pitcher_safe_context_pitcher_baseout_flag_reverse_rate',
 'pitcher_context_effect_pitcher_baseout_flag_ball',
 'pitcher_safe_context_pitcher_baseout_flag_ball_rate',
 'pitcher_context_effect_pitcher_baseout_flag_strike',
 'pitcher_safe_context_pitcher_baseout_flag_strike_rate',
 'pitcher_context_effect_pitcher_baseout_flag_fastball',
 'pitcher_safe_context_pitcher_baseout_flag_fastball_rate',
 'pitcher_context_effect_pitcher_baseout_flag_breaking',
 'pitcher_safe_context_pitcher_baseout_flag_breaking_rate',
 'pitcher_context_effect_pitcher_baseout_flag_offspeed',
 'pitcher_safe_context_pitcher_baseout_flag_offspeed_rate',
 'pitcher_hlogit_count_flag_success',
 'pitcher_hlogit_hand_flag_success',
 'pitcher_hlogit_count_hand_interaction_flag_success',
 'pitcher_hlogit_baseout_flag_success',
 'pitcher_hlogit_count_flag_middle',
 'pitcher_hlogit_hand_flag_middle',
 'pitcher_hlogit_count_hand_interaction_flag_middle',
 'pitcher_hlogit_baseout_flag_middle',
 'pitcher_hlogit_count_flag_reverse',
 'pitcher_hlogit_hand_flag_reverse',
 'pitcher_hlogit_count_hand_interaction_flag_reverse',
 'pitcher_hlogit_baseout_flag_reverse',
 'pitcher_hlogit_count_flag_ball',
 'pitcher_hlogit_hand_flag_ball',
 'pitcher_hlogit_count_hand_interaction_flag_ball',
 'pitcher_hlogit_baseout_flag_ball',
 'pitcher_hlogit_count_flag_strike',
 'pitcher_hlogit_hand_flag_strike',
 'pitcher_hlogit_count_hand_interaction_flag_strike',
 'pitcher_hlogit_baseout_flag_strike',
 'pitcher_hlogit_count_flag_fastball',
 'pitcher_hlogit_hand_flag_fastball',
 'pitcher_hlogit_count_hand_interaction_flag_fastball',
 'pitcher_hlogit_baseout_flag_fastball',
 'pitcher_hlogit_count_flag_breaking',
 'pitcher_hlogit_hand_flag_breaking',
 'pitcher_hlogit_count_hand_interaction_flag_breaking',
 'pitcher_hlogit_baseout_flag_breaking',
 'pitcher_hlogit_count_flag_offspeed',
 'pitcher_hlogit_hand_flag_offspeed',
 'pitcher_hlogit_count_hand_interaction_flag_offspeed',
 'pitcher_hlogit_baseout_flag_offspeed',
 'current_season_pitcher_n',
 'current_season_pitcher_success_rate',
 'current_season_pitcher_success_shrunk',
 'current_season_pitcher_reliability',
 'current_season_minus_career_success',
 'pitcher_latent_skill',
 'pitcher_latent_reliability',
 'pitcher_latent_vs_league',
 'pitcher_latent_vs_asof',
 'pitcher_success_latent_shrunk',
 'pitcher_coldstart_lt25',
 'pitcher_coldstart_lt50',
 'pitcher_coldstart_lt100',
 'current_season_pitcher_success_shrunk_latent',
 'current_season_success_vs_latent',
 'game_dayofweek',
 'top_bottom',
 'game_type',
 'base_state',
 'pitcher_hand',
 'batter_hand',
 'pitcher_team_id',
 'batter_team_id',
 'count_state',
 'base_out_state',
 'hand_matchup']


# -----------------------------------------------------------------------------
# Revised-pipeline configuration
# -----------------------------------------------------------------------------

TEMPORAL_CONTEXT_GROUPS = {
    "pitcher": ["pitcher_id"],
    "pitcher_hand": ["pitcher_id", "batter_hand"],
    "pitcher_count": ["pitcher_id", "count_state"],
    "pitcher_count_hand": ["pitcher_id", "count_state", "batter_hand"],
    "pitcher_baseout": ["pitcher_id", "base_out_state"],
    "pitcher_count_baseout": ["pitcher_id", "count_state", "base_out_state"],
}

TRANSFORMER_CONTEXT_MAP = {
    "Overall": "pitcher",
    "Hand": "pitcher_hand",
    "Count": "pitcher_count",
    "CountHand": "pitcher_count_hand",
    "BaseOut": "pitcher_baseout",
    "CountBaseOut": "pitcher_count_baseout",
}

TRANSFORMER_METRICS = (
    "success",
    "middle",
    "reverse",
    "ball",
    "strike",
    "fastball",
    "breaking",
    "offspeed",
)

TRANSFORMER_TIMES = ("Career", "Prev3", "Prev2", "Prev1")

QUERY_NUMERIC_COLUMNS = [
    "season",
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "pitcher_is_home",
    "pitcher_team_win_expectancy",
    "pitcher_history_missing",
    "previous_game_history_missing",
    "preseason_pitcher_missing",
    "current_season_pitcher_n",
    "current_season_pitcher_success_rate",
    "current_season_pitcher_success_shrunk",
    "current_season_pitcher_reliability",
    "current_season_minus_career_success",
]

QUERY_CATEGORICAL_COLUMNS = list(CAT_COLS)
QUERY_COLUMNS = QUERY_NUMERIC_COLUMNS + QUERY_CATEGORICAL_COLUMNS

RAW_REQUIRED_COMMON = [
    "row_id",
    "season",
    "pitcher_id",
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

@dataclass(frozen=True)
class PipelineConfig:
    row_id_col: str = ID
    season_col: str = "season"
    target_col: str = TARGET
    game_type_col: str = "game_type"

    train_start_year: int = 2019
    dev_val_year_1: int = 2022
    dev_val_year_2: int = 2023
    locked_val_year: int = 2024
    test_year: int = 2025

    dev_2023_weight: float = 2.0
    catboost_locked_candidates: int = 3
    seeds: Tuple[int, ...] = (41, 42, 43)

    # Exact current-submission fixed-tree convention.
    cb_iterations: int = SUBMISSION_FIXED_ITERATIONS
    cb_depth: int = int(MAIN_MODEL_PARAMS["depth"])
    cb_learning_rate: float = float(MAIN_MODEL_PARAMS["learning_rate"])
    cb_l2_leaf_reg: float = float(MAIN_MODEL_PARAMS["l2_leaf_reg"])
    cb_random_strength: float = float(MAIN_MODEL_PARAMS["random_strength"])
    cb_max_ctr_complexity: int = int(MAIN_MODEL_PARAMS["max_ctr_complexity"])
    cb_border_count: int = int(MAIN_MODEL_PARAMS["border_count"])

    # Keep the current notebook baseline unweighted by default.
    # "guide_old_f" applies 0.25 to F rows from 2019~2022.
    sample_weight_mode: str = "none"
    old_f_weight: float = 0.25

    # Exact-season temporal representation.
    temporal_history_scope: str = "all"  # "all" or "R"

    # DL V1.
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 128
    dropout: float = 0.15
    token_dropout: float = 0.10
    dl_batch_size: int = 512
    dl_lr: float = 1e-3
    dl_weight_decay: float = 1e-4
    dl_max_epochs: int = 60
    # Long-running DL controls. These affect only epoch selection in dev; locked/final
    # still train for the frozen best epoch chosen in dev.
    dl_min_epochs: int = 5
    dl_patience: int = 5
    dl_min_delta: float = 1e-6
    dl_eval_every: int = 1
    dl_log_every: int = 1
    dl_resume: bool = True

    # MLP sanity baseline.
    mlp_hidden_1: int = 128
    mlp_hidden_2: int = 64
    mlp_dropout: float = 0.15

    # Fixed blend first.
    blend_cb_weight: float = 0.80


# -----------------------------------------------------------------------------
# Notebook feature functions (copied from uploaded current-submission notebook)
# -----------------------------------------------------------------------------
def add_base_features(df):
    df["count_state"] = (
        df["balls_before"].astype(str)
        + "-"
        + df["strikes_before"].astype(str)
    )
    df["base_out_state"] = (
        df["base_state"].astype(str)
        + "_"
        + df["outs_before"].astype(str)
    )
    df["hand_matchup"] = (
        df["pitcher_hand"].astype(str)
        + "_"
        + df["batter_hand"].astype(str)
    )
    df["pitcher_is_home"] = df["top_bottom"].eq("T").astype("int8")
    df["pitcher_team_win_expectancy"] = np.where(
        df["pitcher_is_home"].eq(1),
        df["home_win_expectancy"],
        df["away_win_expectancy"],
    ).astype("float32")
    df["pitcher_history_missing"] = df["asof_pitcher_n"].eq(0).astype("int8")
    df["previous_game_history_missing"] = (
        df["asof_pitcher_prev1_game_success_rate"].isna().astype("int8")
    )

    for window in (1, 3, 5):
        df[f"pitcher_success_prev{window}_delta"] = (
            df[f"asof_pitcher_prev{window}_game_success_rate"]
            - df["asof_pitcher_success_rate"]
        ).astype("float32")

    pitcher_n_cap500 = df["asof_pitcher_n"].clip(upper=500).astype("float32")
    for metric in ("success", "reverse", "ball", "strike"):
        df[f"pitcher_{metric}_count500"] = (
            df[f"asof_pitcher_{metric}_rate"] * pitcher_n_cap500
        ).astype("float32")


def reconstruct_pitch_flags(data):
    pitcher_group = data.groupby(["pitcher_id", "season"], sort=False)

    def reconstruct_flag(rate_column, n_column):
        n_now = data[n_column].astype("float64")
        n_next = pitcher_group[n_column].shift(-1).astype("float64")
        rate_now = data[rate_column].fillna(0.0).astype("float64")
        rate_next = pitcher_group[rate_column].shift(-1).astype("float64")
        raw_flag = n_next * rate_next - n_now * rate_now
        valid_flag = n_next.eq(n_now + 1) & rate_next.notna()

        reconstructed = pd.Series(np.nan, index=data.index, dtype="float32")
        reconstructed.loc[valid_flag] = (
            np.rint(raw_flag.loc[valid_flag])
            .clip(0, 1)
            .astype("float32")
        )
        return reconstructed

    data["flag_success"] = data[TARGET].astype("float32")

    mapping = {
        "flag_middle": ("asof_pitcher_middle_rate", "asof_pitcher_n"),
        "flag_reverse": ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
        "flag_ball": ("asof_pitcher_ball_rate", "asof_pitcher_n"),
        "flag_strike": ("asof_pitcher_strike_rate", "asof_pitcher_n"),
        "flag_fastball": ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
        "flag_breaking": ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
        "flag_offspeed": ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
    }

    for flag, (rate_column, n_column) in mapping.items():
        data[flag] = reconstruct_flag(rate_column, n_column)


def flat_n_column(prefix, group_name, flag):
    if flag == "flag_success":
        return f"{prefix}_{group_name}_success_n"
    return f"{prefix}_{group_name}_n"


def calculate_flat_priors(history_data):
    if len(history_data) == 0:
        return {flag: np.nan for flag in FLAG_COLS}
    return {
        flag: float(history_data[flag].dropna().mean())
        for flag in FLAG_COLS
    }


def make_flat_features(target_data, history_data, groups, prefix="flat"):
    target_order = np.arange(len(target_data), dtype=np.int32)
    priors = calculate_flat_priors(history_data)
    feature_parts = []

    for group_name, group_columns in groups.items():
        feature_data = {}

        if len(history_data) == 0:
            for flag in FLAG_COLS:
                feature_data[f"{prefix}_{group_name}_{flag}_rate"] = np.full(
                    len(target_data), np.nan, dtype=np.float32
                )
                n_column = flat_n_column(prefix, group_name, flag)
                if n_column not in feature_data:
                    feature_data[n_column] = np.zeros(
                        len(target_data), dtype=np.int32
                    )
            group_features = pd.DataFrame(feature_data, index=target_order)

        else:
            aggregation = {}
            for flag in FLAG_COLS:
                aggregation[f"__{flag}__sum"] = (flag, "sum")
                aggregation[f"__{flag}__count"] = (flag, "count")

            history_summary = (
                history_data
                .groupby(
                    group_columns,
                    dropna=False,
                    observed=False,
                )
                .agg(**aggregation)
                .reset_index()
            )

            target_keys = target_data[group_columns].copy()
            target_keys["_flat_order"] = target_order
            merged = target_keys.merge(
                history_summary,
                on=group_columns,
                how="left",
                sort=False,
            )

            for flag in FLAG_COLS:
                n = merged[f"__{flag}__count"].fillna(0).astype("int32")
                flag_sum = merged[f"__{flag}__sum"].fillna(0.0).astype("float64")
                rate = (
                    flag_sum + FLAT_ALPHA * priors[flag]
                ) / (
                    n.astype("float64") + FLAT_ALPHA
                )

                feature_data[f"{prefix}_{group_name}_{flag}_rate"] = (
                    rate.astype("float32").to_numpy()
                )

                n_column = flat_n_column(prefix, group_name, flag)
                if n_column not in feature_data:
                    feature_data[n_column] = n.to_numpy()

            group_features = pd.DataFrame(
                feature_data,
                index=merged["_flat_order"].to_numpy(),
            )

        feature_parts.append(group_features)

    flat_features = pd.concat(feature_parts, axis=1).sort_index()
    flat_features.index = target_data.index
    return flat_features


def build_seasonal_flat_features(target_frame, history_frame):
    season_parts = []
    for target_season in sorted(target_frame["season"].unique()):
        target_rows = target_frame.loc[target_frame["season"].eq(target_season)]
        history_rows = history_frame.loc[history_frame["season"].lt(target_season)]
        season_parts.append(
            make_flat_features(
                target_rows,
                history_rows,
                FLAT_GROUPS,
                "flat",
            )
        )
    return pd.concat(season_parts, axis=0).reindex(target_frame.index)


def add_pitcher_context_features(df):
    feature_data = {}

    for flag, asof_column in ASOF_FLAG_MAP.items():
        current_rate = df[asof_column].astype("float32")
        historical_overall_rate = df[f"flat_pitcher_{flag}_rate"].astype("float32")
        feature_data[f"pitcher_drift_{flag}"] = (
            current_rate - historical_overall_rate
        ).astype("float32").to_numpy(copy=False)

    for context in PITCHER_CONTEXTS:
        for flag, asof_column in ASOF_FLAG_MAP.items():
            current_rate = df[asof_column].astype("float32")
            overall_rate = df[f"flat_pitcher_{flag}_rate"].astype("float32")
            context_rate = df[f"flat_{context}_{flag}_rate"].astype("float32")
            context_n = df[
                flat_n_column("flat", context, flag)
            ].astype("float32")

            reliability = (
                context_n / (context_n + CONTEXT_EFFECT_ALPHA)
            ).astype("float32")

            context_effect = (
                (context_rate - overall_rate) * reliability
            ).astype("float32")

            feature_data[f"pitcher_context_effect_{context}_{flag}"] = (
                context_effect.to_numpy(copy=False)
            )
            feature_data[f"pitcher_safe_context_{context}_{flag}_rate"] = (
                (current_rate + context_effect)
                .clip(lower=0.0, upper=1.0)
                .astype("float32")
                .to_numpy(copy=False)
            )

    features = pd.DataFrame(feature_data, index=df.index)
    df[features.columns] = features


def prob_to_logit(values):
    probabilities = np.asarray(values, dtype=np.float32)
    probabilities = np.clip(probabilities, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(probabilities / (1.0 - probabilities)).astype("float32")


def add_pitcher_hlogit_features(df):
    feature_data = {}

    for flag in ASOF_FLAG_MAP:
        overall_logit = prob_to_logit(df[f"flat_pitcher_{flag}_rate"])
        count_logit = prob_to_logit(df[f"flat_pitcher_count_{flag}_rate"])
        hand_logit = prob_to_logit(df[f"flat_pitcher_hand_{flag}_rate"])
        count_hand_logit = prob_to_logit(
            df[f"flat_pitcher_count_hand_{flag}_rate"]
        )
        baseout_logit = prob_to_logit(df[f"flat_pitcher_baseout_{flag}_rate"])

        count_n = df[
            flat_n_column("flat", "pitcher_count", flag)
        ].to_numpy(dtype=np.float32, copy=False)
        hand_n = df[
            flat_n_column("flat", "pitcher_hand", flag)
        ].to_numpy(dtype=np.float32, copy=False)
        count_hand_n = df[
            flat_n_column("flat", "pitcher_count_hand", flag)
        ].to_numpy(dtype=np.float32, copy=False)
        baseout_n = df[
            flat_n_column("flat", "pitcher_baseout", flag)
        ].to_numpy(dtype=np.float32, copy=False)

        count_rel = (count_n / (count_n + CONTEXT_EFFECT_ALPHA)).astype("float32")
        hand_rel = (hand_n / (hand_n + CONTEXT_EFFECT_ALPHA)).astype("float32")
        count_hand_rel = (
            count_hand_n / (count_hand_n + CONTEXT_EFFECT_ALPHA)
        ).astype("float32")
        baseout_rel = (
            baseout_n / (baseout_n + CONTEXT_EFFECT_ALPHA)
        ).astype("float32")

        feature_data[f"pitcher_hlogit_count_{flag}"] = (
            (count_logit - overall_logit) * count_rel
        ).astype("float32")
        feature_data[f"pitcher_hlogit_hand_{flag}"] = (
            (hand_logit - overall_logit) * hand_rel
        ).astype("float32")

        interaction_rel = np.minimum(
            np.minimum(count_rel, hand_rel),
            count_hand_rel,
        ).astype("float32")

        feature_data[f"pitcher_hlogit_count_hand_interaction_{flag}"] = (
            (
                count_hand_logit
                - count_logit
                - hand_logit
                + overall_logit
            )
            * interaction_rel
        ).astype("float32")

        feature_data[f"pitcher_hlogit_baseout_{flag}"] = (
            (baseout_logit - overall_logit) * baseout_rel
        ).astype("float32")

    features = pd.DataFrame(feature_data, index=df.index)
    df[features.columns] = features


CURRENT_SEASON_FEATURES = [
    "current_season_pitcher_n",
    "current_season_pitcher_success_rate",
    "current_season_pitcher_success_shrunk",
    "current_season_pitcher_reliability",
    "current_season_minus_career_success",
]


def make_current_season_features(target_data, history_data, prior_league_success_rate):
    prior_summary = (
        history_data
        .groupby("pitcher_id", dropna=False, observed=True)
        .agg(
            prior_pitcher_success_sum=("flag_success", "sum"),
            prior_pitcher_n=("flag_success", "count"),
        )
        .reset_index()
    )

    target_values = target_data[
        ["pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"]
    ].copy()
    target_values["_order"] = np.arange(len(target_data), dtype=np.int32)
    merged = target_values.merge(
        prior_summary,
        on="pitcher_id",
        how="left",
        sort=False,
    )

    prior_n = merged["prior_pitcher_n"].fillna(0).astype("float32")
    prior_sum = merged["prior_pitcher_success_sum"].fillna(0.0).astype("float32")

    asof_sum = (
        merged["asof_pitcher_n"].astype("float32")
        * merged["asof_pitcher_success_rate"].fillna(0.0).astype("float32")
    ).astype("float32")

    current_n = (
        merged["asof_pitcher_n"].astype("float32") - prior_n
    ).clip(lower=0.0).astype("float32")
    current_n_array = current_n.to_numpy(copy=False)

    current_sum = np.minimum(
        (asof_sum - prior_sum)
        .clip(lower=0.0)
        .astype("float32")
        .to_numpy(copy=False),
        current_n_array,
    ).astype("float32")

    current_rate = np.divide(
        current_sum,
        current_n_array,
        out=np.full(len(target_data), np.nan, dtype=np.float32),
        where=current_n_array >= 1.0,
    ).astype("float32")

    current_shrunk = (
        (
            current_sum
            + CURRENT_SEASON_ALPHA * prior_league_success_rate
        )
        / (current_n_array + CURRENT_SEASON_ALPHA)
    ).astype("float32")

    features = pd.DataFrame(
        {
            "current_season_pitcher_n": current_n_array,
            "current_season_pitcher_success_rate": current_rate,
            "current_season_pitcher_success_shrunk": current_shrunk,
            "current_season_pitcher_reliability": (
                current_n_array
                / (current_n_array + CURRENT_SEASON_ALPHA)
            ).astype("float32"),
            "current_season_minus_career_success": (
                current_shrunk
                - merged["asof_pitcher_success_rate"].to_numpy(
                    dtype=np.float32,
                    copy=False,
                )
            ).astype("float32"),
        },
        index=merged["_order"].to_numpy(),
    ).sort_index()

    features.index = target_data.index
    return features[CURRENT_SEASON_FEATURES]


def build_current_season_features(target_frame, preprocessing_history_data):
    season_parts = []
    first_season = int(preprocessing_history_data["season"].min())

    for target_season in sorted(target_frame["season"].unique()):
        target_rows = target_frame.loc[target_frame["season"].eq(target_season)]
        history_rows = preprocessing_history_data.loc[
            preprocessing_history_data["season"].lt(target_season)
        ]

        prior = (
            0.5
            if target_season == first_season
            else float(history_rows["flag_success"].mean())
        )

        season_parts.append(
            make_current_season_features(target_rows, history_rows, prior)
        )

    return pd.concat(season_parts, axis=0).reindex(target_frame.index)


def add_preseason_pitcher_missing(train_df):
    train_df["preseason_pitcher_missing"] = np.int8(0)
    seen_pitchers = set()

    for season in sorted(train_df["season"].dropna().astype(int).unique()):
        mask = train_df["season"].eq(season)
        pitchers = train_df.loc[mask, "pitcher_id"]
        train_df.loc[mask, "preseason_pitcher_missing"] = (
            ~pitchers.isin(seen_pitchers)
        ).astype("int8").to_numpy()
        seen_pitchers.update(pitchers.dropna().unique().tolist())

    train_df["preseason_pitcher_missing"] = (
        train_df["preseason_pitcher_missing"].astype("int8")
    )


def prepare_skill_x(df):
    x = df[SKILL_NUMERIC_COLUMNS + SKILL_CAT_COLUMNS].copy()

    n = (
        pd.to_numeric(x["asof_pitcher_n"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )

    x["skill_asof_n_log1p"] = np.log1p(n).astype("float32")
    x["skill_asof_n_sqrt"] = np.sqrt(n).astype("float32")

    for threshold in (10, 25, 50, 100, 200):
        x[f"skill_n_lt_{threshold}"] = n.lt(threshold).astype("int8")

    x["skill_success_rate_missing"] = (
        x["asof_pitcher_success_rate"].isna().astype("int8")
    )

    for window in (1, 3, 5):
        recent_column = f"asof_pitcher_prev{window}_game_success_rate"
        x[f"skill_prev{window}_success_delta"] = (
            x[recent_column] - x["asof_pitcher_success_rate"]
        ).astype("float32")

    pitchmix_columns = [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    pitchmix = x[pitchmix_columns].astype("float32")
    x["skill_pitchmix_max"] = pitchmix.max(axis=1).astype("float32")
    safe_pitchmix = pitchmix.clip(lower=1e-6)
    x["skill_pitchmix_entropy"] = (
        -(safe_pitchmix * np.log(safe_pitchmix))
        .sum(axis=1)
        .astype("float32")
    )

    for column in SKILL_CAT_COLUMNS:
        x[column] = x[column].astype("string").fillna("__NA__").astype(str)

    return x


def build_skill_snapshots(train_df):
    snapshot_parts = []
    work = train_df.copy()
    work["_skill_original_order"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["season", "pitcher_id", "_skill_original_order"],
        kind="stable",
    )

    season_priors = (
        train_df.groupby("season", observed=False)[TARGET].mean().to_dict()
    )

    grouped = work.groupby(
        ["season", "pitcher_id"],
        sort=False,
        observed=False,
    )

    for (season, pitcher_id), pitcher_season in grouped:
        size = len(pitcher_season)
        if size <= SKILL_MIN_FUTURE_PITCHES:
            continue

        target_values = (
            pd.to_numeric(pitcher_season[TARGET], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        reverse_future_sum = np.cumsum(target_values[::-1])[::-1]

        valid_positions = []
        for position in SKILL_SNAPSHOT_POSITIONS:
            future_n = size - position - 1
            if position < size - 1 and future_n >= SKILL_MIN_FUTURE_PITCHES:
                valid_positions.append(position)

        if not valid_positions:
            continue

        snapshot = pitcher_season.iloc[valid_positions].copy()
        future_n = np.asarray(
            [size - position - 1 for position in valid_positions],
            dtype=np.int32,
        )
        future_success_sum = np.asarray(
            [reverse_future_sum[position + 1] for position in valid_positions],
            dtype=np.float64,
        )
        season_prior = float(season_priors[season])

        snapshot["_skill_snapshot_season"] = int(season)
        snapshot["_skill_future_n"] = future_n
        snapshot["_skill_target"] = (
            (
                future_success_sum
                + SKILL_TARGET_ALPHA * season_prior
            )
            / (future_n + SKILL_TARGET_ALPHA)
        ).astype("float32")
        snapshot["_skill_weight"] = np.sqrt(
            np.minimum(future_n, 600)
        ).astype("float32")
        snapshot_parts.append(snapshot)

    return pd.concat(snapshot_parts, axis=0, ignore_index=True)


def fit_skill_model(snapshot_df):
    x = prepare_skill_x(snapshot_df)
    y = snapshot_df["_skill_target"].astype("float32")
    weight = snapshot_df["_skill_weight"].astype("float32")

    model = CatBoostRegressor(**SKILL_MODEL_PARAMS)
    model.fit(
        x,
        y,
        sample_weight=weight,
        cat_features=SKILL_CAT_COLUMNS,
    )
    return model


def historical_league_prior(train_df, target_season):
    history = train_df.loc[train_df["season"].lt(target_season), TARGET]
    return 0.5 if len(history) == 0 else float(history.mean())


def build_walkforward_latent_skill(train_df, skill_snapshots):
    latent_skill = pd.Series(np.nan, index=train_df.index, dtype="float32")
    latent_prior = pd.Series(np.nan, index=train_df.index, dtype="float32")

    for target_season in sorted(train_df["season"].dropna().astype(int).unique()):
        started = time.perf_counter()
        target_mask = train_df["season"].eq(target_season)
        prior = historical_league_prior(train_df, target_season)
        latent_prior.loc[target_mask] = prior

        history_snapshots = skill_snapshots.loc[
            skill_snapshots["_skill_snapshot_season"].lt(target_season)
        ]

        if len(history_snapshots) == 0:
            latent_skill.loc[target_mask] = prior
            print(
                f"Latent {target_season}: fallback={prior:.6f}"
                f" | {time.perf_counter() - started:.1f}s"
            )
            continue

        model = fit_skill_model(history_snapshots)
        x_target = prepare_skill_x(train_df.loc[target_mask])
        prediction = np.clip(
            model.predict(x_target),
            0.05,
            0.95,
        ).astype("float32")
        latent_skill.loc[target_mask] = prediction

        history_seasons = sorted(
            history_snapshots["_skill_snapshot_season"].unique().tolist()
        )
        print(
            f"Latent {target_season}: history={history_seasons}"
            f" | snapshots={len(history_snapshots):,}"
            f" | rows={int(target_mask.sum()):,}"
            f" | {time.perf_counter() - started:.1f}s"
        )

        del model, x_target, prediction, history_snapshots
        gc.collect()

    return latent_skill, latent_prior


def add_latent_blend_features(df, latent_skill, latent_prior):
    n = (
        pd.to_numeric(df["asof_pitcher_n"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    observed_rate = pd.to_numeric(
        df["asof_pitcher_success_rate"],
        errors="coerce",
    )

    latent = pd.Series(latent_skill, index=df.index, dtype="float32")
    prior = pd.Series(latent_prior, index=df.index, dtype="float32")
    latent = latent.fillna(prior).fillna(0.5)
    observed_filled = observed_rate.fillna(latent)
    reliability = n / (n + SKILL_OBS_ALPHA)

    df["pitcher_latent_skill"] = latent.astype("float32")
    df["pitcher_latent_reliability"] = reliability.astype("float32")
    df["pitcher_latent_vs_league"] = (latent - prior).astype("float32")
    df["pitcher_latent_vs_asof"] = (
        latent - observed_filled
    ).astype("float32")
    df["pitcher_success_latent_shrunk"] = (
        (
            n * observed_filled
            + SKILL_OBS_ALPHA * latent
        )
        / (n + SKILL_OBS_ALPHA)
    ).astype("float32")

    preseason_missing = df["preseason_pitcher_missing"].eq(1)
    for threshold in (25, 50, 100):
        df[f"pitcher_coldstart_lt{threshold}"] = (
            preseason_missing & n.lt(threshold)
        ).astype("int8")

    current_n = (
        pd.to_numeric(df["current_season_pitcher_n"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    current_rate = (
        pd.to_numeric(
            df["current_season_pitcher_success_rate"],
            errors="coerce",
        )
        .fillna(latent)
    )

    df["current_season_pitcher_success_shrunk_latent"] = (
        (
            current_n * current_rate
            + SKILL_OBS_ALPHA * latent
        )
        / (current_n + SKILL_OBS_ALPHA)
    ).astype("float32")
    df["current_season_success_vs_latent"] = (
        current_rate - latent
    ).astype("float32")


def normalize_categorical_columns(df):
    for column in CAT_COLS:
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].isna().any():
                df[column] = df[column].fillna(-999999).astype("int64")
        else:
            df[column] = df[column].fillna("__NA__").astype(str)


def evaluate_probability(probability, y_true, baseline_brier):
    brier = float(np.mean((probability - y_true) ** 2))
    score = max(0.0, 100000.0 * (1.0 - brier / baseline_brier))
    return brier, score


# -----------------------------------------------------------------------------
# Reproducibility / IO helpers
# -----------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_versions() -> Dict[str, str]:
    import sklearn
    import catboost

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "catboost": catboost.__version__,
        "torch": torch.__version__,
    }


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def validate_required_columns(df: pd.DataFrame, cfg: PipelineConfig, require_test: bool = False) -> None:
    required = [cfg.row_id_col, cfg.season_col, cfg.target_col, cfg.game_type_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[cfg.row_id_col].duplicated().any():
        dup = int(df[cfg.row_id_col].duplicated().sum())
        raise ValueError(f"{cfg.row_id_col} must be unique; found {dup} duplicates")
    if require_test and not (df[cfg.season_col] == cfg.test_year).any():
        raise ValueError(f"No rows found for test year {cfg.test_year}")


def input_fingerprint(input_path: Optional[Path], df: pd.DataFrame) -> Dict[str, Any]:
    if input_path is not None:
        return {
            "path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
        }
    # Demo/synthetic: hash stable serialized metadata + a deterministic sample.
    sample = df.head(100).to_json(orient="split", date_format="iso", double_precision=12)
    return {
        "path": "<synthetic-demo>",
        "sha256": hashlib.sha256(sample.encode("utf-8")).hexdigest(),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
    }


def current_code_sha256() -> str:
    try:
        return sha256_file(Path(__file__).resolve())
    except Exception:
        return "unknown"





# -----------------------------------------------------------------------------
# Raw-data validation and notebook-aligned feature preparation
# -----------------------------------------------------------------------------

def validate_raw_columns(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_required = RAW_REQUIRED_COMMON + [TARGET]
    test_required = RAW_REQUIRED_COMMON
    missing_train = [c for c in train_required if c not in train.columns]
    missing_test = [c for c in test_required if c not in test.columns]
    if missing_train:
        raise ValueError(f"train.csv missing required columns: {missing_train}")
    if missing_test:
        raise ValueError(f"test.csv missing required columns: {missing_test}")
    if train[ID].duplicated().any():
        raise ValueError(f"train {ID} must be unique.")
    if test[ID].duplicated().any():
        raise ValueError(f"test {ID} must be unique.")
    overlap = set(train[ID]).intersection(set(test[ID]))
    if overlap:
        raise ValueError(f"train/test {ID} overlap detected; examples={list(overlap)[:5]}")


def reconstruction_quality_report(data: pd.DataFrame) -> Dict[str, float]:
    report = {}
    for flag in FLAG_COLS:
        if flag in data.columns:
            report[flag] = float(data[flag].notna().mean())
    return report


def add_preseason_pitcher_missing_for_target(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.Series:
    out = pd.Series(np.zeros(len(target_df), dtype=np.int8), index=target_df.index)
    for target_season in sorted(target_df["season"].dropna().astype(int).unique()):
        mask = target_df["season"].eq(target_season)
        seen = set(
            history_df.loc[history_df["season"].lt(target_season), "pitcher_id"]
            .dropna()
            .unique()
            .tolist()
        )
        out.loc[mask] = (~target_df.loc[mask, "pitcher_id"].isin(seen)).astype("int8").to_numpy()
    return out.astype("int8")


def predict_latent_skill_for_target(
    target_df: pd.DataFrame,
    train_history_df: pd.DataFrame,
    skill_snapshots: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    pred = pd.Series(np.nan, index=target_df.index, dtype="float32")
    prior_series = pd.Series(np.nan, index=target_df.index, dtype="float32")

    for target_season in sorted(target_df["season"].dropna().astype(int).unique()):
        mask = target_df["season"].eq(target_season)
        prior = historical_league_prior(train_history_df, target_season)
        prior_series.loc[mask] = prior
        history_snapshots = skill_snapshots.loc[
            skill_snapshots["_skill_snapshot_season"].lt(target_season)
        ]
        if history_snapshots.empty:
            pred.loc[mask] = prior
            continue
        model = fit_skill_model(history_snapshots)
        x_target = prepare_skill_x(target_df.loc[mask])
        pred.loc[mask] = np.clip(model.predict(x_target), 0.05, 0.95).astype("float32")
        del model, x_target
        gc.collect()

    return pred, prior_series


def build_target_flat_features(
    target_frame: pd.DataFrame,
    history_frame: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    for target_season in sorted(target_frame["season"].dropna().astype(int).unique()):
        target_rows = target_frame.loc[target_frame["season"].eq(target_season)]
        history_rows = history_frame.loc[history_frame["season"].lt(target_season)]
        parts.append(make_flat_features(target_rows, history_rows, FLAT_GROUPS, "flat"))
    return pd.concat(parts, axis=0).reindex(target_frame.index)


def _temporal_source(
    history_frame: pd.DataFrame,
    source_season: int,
    scope: str,
) -> pd.DataFrame:
    source = history_frame.loc[history_frame["season"].eq(source_season)]
    if scope == "R":
        source = source.loc[source["game_type"].astype(str).eq("R")]
    elif scope != "all":
        raise ValueError("temporal_history_scope must be 'all' or 'R'")
    return source


def build_exact_season_temporal_features(
    target_frame: pd.DataFrame,
    history_frame: pd.DataFrame,
    scope: str = "all",
) -> pd.DataFrame:
    """Prev1/2/3 exact-season profiles; never uses target/test rows as history."""
    season_parts = []

    for target_season in sorted(target_frame["season"].dropna().astype(int).unique()):
        target_rows = target_frame.loc[target_frame["season"].eq(target_season)]
        lag_parts = []

        for lag in (1, 2, 3):
            source_season = int(target_season) - lag
            source = _temporal_source(history_frame, source_season, scope)
            prefix = f"temporal_prev{lag}"

            lag_df = make_flat_features(
                target_rows,
                source,
                TEMPORAL_CONTEXT_GROUPS,
                prefix,
            )

            priors = calculate_flat_priors(source)
            for flag in FLAG_COLS:
                prior_col = f"{prefix}_league_{flag}_rate"
                lag_df[prior_col] = np.float32(priors[flag])

                overall_col = f"{prefix}_pitcher_{flag}_rate"
                rel_col = f"{overall_col}_rel"
                lag_df[rel_col] = (
                    lag_df[overall_col].astype("float32") - np.float32(priors[flag])
                ).astype("float32")

            success_n = flat_n_column(prefix, "pitcher", "flag_success")
            lag_df[f"{prefix}_pitcher_available"] = (
                pd.to_numeric(lag_df[success_n], errors="coerce").fillna(0).gt(0)
            ).astype("int8")

            lag_parts.append(lag_df)

        season_feat = pd.concat(lag_parts, axis=1)

        for flag in FLAG_COLS:
            p1 = f"temporal_prev1_pitcher_{flag}_rate_rel"
            p2 = f"temporal_prev2_pitcher_{flag}_rate_rel"
            p3 = f"temporal_prev3_pitcher_{flag}_rate_rel"
            season_feat[f"temporal_trend12_pitcher_{flag}_rel"] = (
                season_feat[p1] - season_feat[p2]
            ).astype("float32")
            season_feat[f"temporal_trend23_pitcher_{flag}_rel"] = (
                season_feat[p2] - season_feat[p3]
            ).astype("float32")

        season_feat.index = target_rows.index
        season_parts.append(season_feat)

    return pd.concat(season_parts, axis=0).reindex(target_frame.index)


def discover_temporal_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    abs_level, rel_level, rel_trend, reliability, level_control = [], [], [], [], []

    for lag in (1, 2, 3):
        prefix = f"temporal_prev{lag}"
        for flag in FLAG_COLS:
            abs_col = f"{prefix}_pitcher_{flag}_rate"
            rel_col = f"{abs_col}_rel"
            league_col = f"{prefix}_league_{flag}_rate"
            if abs_col in df.columns:
                abs_level.append(abs_col)
            if rel_col in df.columns:
                rel_level.append(rel_col)
            if league_col in df.columns:
                level_control.append(league_col)

        n_col = flat_n_column(prefix, "pitcher", "flag_success")
        a_col = f"{prefix}_pitcher_available"
        if n_col in df.columns:
            reliability.append(n_col)
        if a_col in df.columns:
            reliability.append(a_col)

    for flag in FLAG_COLS:
        for pair in ("12", "23"):
            c = f"temporal_trend{pair}_pitcher_{flag}_rel"
            if c in df.columns:
                rel_trend.append(c)

    return {
        "abs_level": abs_level,
        "rel_level": rel_level,
        "rel_trend": rel_trend,
        "reliability": reliability,
        "level_control": level_control,
    }


def build_catboost_arms(df: pd.DataFrame) -> Dict[str, List[str]]:
    missing_base = [c for c in FEATURES if c not in df.columns]
    if missing_base:
        raise RuntimeError(f"Base263 columns missing: {missing_base[:20]}")
    groups = discover_temporal_feature_groups(df)
    base = list(FEATURES)
    return {
        "T0_BASE": base,
        "T1_ABS_LEVEL": list(dict.fromkeys(base + groups["abs_level"] + groups["reliability"])),
        "T2_REL_LEVEL": list(dict.fromkeys(base + groups["rel_level"] + groups["reliability"])),
        "T3_REL_TREND": list(dict.fromkeys(base + groups["rel_trend"] + groups["reliability"])),
        "T4_REL_LEVEL_PLUS_TREND": list(
            dict.fromkeys(base + groups["rel_level"] + groups["rel_trend"] + groups["reliability"])
        ),
        "T5_RELIABILITY_ONLY": list(dict.fromkeys(base + groups["reliability"])),
        "T6_LEVEL_ONLY_CONTROL": list(dict.fromkeys(base + groups["level_control"])),
    }


def prepare_feature_table(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Rebuild Base263 + exact-season temporal features from raw train/test."""
    validate_raw_columns(train_raw, test_raw)

    train = train_raw.copy()
    test = test_raw.copy()

    print("[prepare] base situation features")
    add_base_features(train)
    add_base_features(test)

    print("[prepare] reconstruct train pitch flags")
    reconstruct_pitch_flags(train)
    flag_quality = reconstruction_quality_report(train)
    print("[prepare] reconstructed flag non-null ratio:", flag_quality)

    # Source notebook uses these reconstructed flags only for historical aggregation.
    preprocessing_history_data = train[["season", "pitcher_id", "flag_success"]].copy()

    print("[prepare] prior-season Flat history")
    train_flat = build_seasonal_flat_features(train, train)
    test_flat = build_target_flat_features(test, train)
    train = pd.concat([train, train_flat], axis=1)
    test = pd.concat([test, test_flat], axis=1)
    del train_flat, test_flat
    gc.collect()

    print("[prepare] Drift / Context Effect / Safe Context")
    add_pitcher_context_features(train)
    add_pitcher_context_features(test)

    print("[prepare] Hierarchical Logit")
    add_pitcher_hlogit_features(train)
    add_pitcher_hlogit_features(test)

    print("[prepare] Current Season decomposition")
    train[CURRENT_SEASON_FEATURES] = build_current_season_features(
        train, preprocessing_history_data
    )
    test[CURRENT_SEASON_FEATURES] = build_current_season_features(
        test, preprocessing_history_data
    )

    print("[prepare] preseason pitcher missing")
    add_preseason_pitcher_missing(train)
    test["preseason_pitcher_missing"] = add_preseason_pitcher_missing_for_target(test, train)

    print("[prepare] Latent Skill snapshots / walk-forward")
    skill_snapshots = build_skill_snapshots(train)
    latent_train, prior_train = build_walkforward_latent_skill(train, skill_snapshots)
    add_latent_blend_features(train, latent_train, prior_train)

    latent_test, prior_test = predict_latent_skill_for_target(test, train, skill_snapshots)
    add_latent_blend_features(test, latent_test, prior_test)

    print("[prepare] exact Prev1/Prev2/Prev3 season profiles")
    train_temporal = build_exact_season_temporal_features(
        train, train, scope=cfg.temporal_history_scope
    )
    test_temporal = build_exact_season_temporal_features(
        test, train, scope=cfg.temporal_history_scope
    )
    train = pd.concat([train, train_temporal], axis=1)
    test = pd.concat([test, test_temporal], axis=1)
    del train_temporal, test_temporal, skill_snapshots
    gc.collect()

    # Preserve the notebook's exact Base263 feature contract.
    missing_train = [c for c in FEATURES if c not in train.columns]
    missing_test = [c for c in FEATURES if c not in test.columns]
    if missing_train or missing_test:
        raise RuntimeError(
            f"Base263 mismatch. train missing={missing_train[:10]} test missing={missing_test[:10]}"
        )
    if len(FEATURES) != 263 or len(CAT_COLS) != 11:
        raise RuntimeError("Notebook contract changed: expected 263 features / 11 categorical.")

    normalize_categorical_columns(train)
    normalize_categorical_columns(test)

    # Test target is intentionally unavailable.
    if TARGET not in test.columns:
        test[TARGET] = np.nan

    train["_dataset_split"] = "train"
    test["_dataset_split"] = "test"

    # Direct pitcher_id is retained only for audit/debug; no model feature list uses it.
    combined = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
    if combined[ID].duplicated().any():
        raise RuntimeError("Prepared row_id is not unique after train/test concat.")
    return combined


def raw_file_fingerprint(paths: Mapping[str, Path]) -> Dict[str, Any]:
    files = {}
    for name, path in paths.items():
        if path.exists():
            files[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
        else:
            files[name] = {"path": str(path.resolve()), "missing": True}
    signature = sha256_json(files)
    return {"files": files, "sha256": signature}


def prepared_table_fingerprint(df: pd.DataFrame, raw_fp: Mapping[str, Any]) -> Dict[str, Any]:
    # Hash stable metadata + raw fingerprint, avoiding a second full serialization hash.
    payload = {
        "raw_sha256": raw_fp["sha256"],
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "split_counts": df["_dataset_split"].value_counts(dropna=False).to_dict(),
    }
    return {**payload, "sha256": sha256_json(payload)}


def save_prepared_contract(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    out_dir: Path,
    raw_fp: Mapping[str, Any],
) -> Dict[str, Any]:
    arms = build_catboost_arms(df)
    schema = build_notebook_state_schema(df)
    query_missing = [c for c in QUERY_COLUMNS if c not in df.columns]
    if query_missing:
        raise RuntimeError(f"Transformer query columns missing: {query_missing}")

    contract = {
        "created_at": now_iso(),
        "raw_fingerprint": raw_fp,
        "prepared_fingerprint": prepared_table_fingerprint(df, raw_fp),
        "code_sha256": current_code_sha256(),
        "config": asdict(cfg),
        "base_feature_count": len(FEATURES),
        "base_features": FEATURES,
        "categorical_count": len(CAT_COLS),
        "categorical_features": CAT_COLS,
        "temporal_feature_groups": discover_temporal_feature_groups(df),
        "catboost_arms": arms,
        "state_schema": schema.to_dict(),
        "query_columns": QUERY_COLUMNS,
        "notes": {
            "pitcher_id_direct_input": False,
            "trackman_used": False,
            "state_token_count": len(schema.cells),
            "current_recent_handling": (
                "Current and recent/asof information are encoded in CURRENT QUERY. "
                "Historical state tokens use Career + exact Prev1/2/3 because the notebook "
                "does not expose full six-context current/recent metric cells."
            ),
        },
    }
    save_json(contract, out_dir / "column_contract.json")
    return contract



# -----------------------------------------------------------------------------
# Sample weights / metrics / CatBoost branch
# -----------------------------------------------------------------------------

def compute_sample_weights(df: pd.DataFrame, cfg: PipelineConfig) -> np.ndarray:
    if cfg.sample_weight_mode == "none":
        return np.ones(len(df), dtype=np.float32)
    if cfg.sample_weight_mode != "guide_old_f":
        raise ValueError("sample_weight_mode must be 'none' or 'guide_old_f'")

    season = pd.to_numeric(df[cfg.season_col], errors="raise").to_numpy()
    gt = df[cfg.game_type_col].astype(str).to_numpy()
    w = np.ones(len(df), dtype=np.float32)
    old_f = (gt == "F") & (season <= 2022)
    w[old_f] = np.float32(cfg.old_f_weight)
    return w


def safe_brier(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() == 0:
        return float("nan")
    pred = np.clip(pred[mask], 1e-7, 1 - 1e-7)
    return float(brier_score_loss(y[mask], pred))


def subgroup_brier(df: pd.DataFrame, cfg: PipelineConfig, y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out = {"overall": safe_brier(y, p)}
    gt = df[cfg.game_type_col].astype(str).to_numpy()
    for group in ("R", "F"):
        mask = gt == group
        out[group] = safe_brier(y[mask], p[mask]) if mask.any() else float("nan")
    return out


def ensemble_diagnostics(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    y: np.ndarray,
    p_cb: np.ndarray,
    p_dl: np.ndarray,
    cb_weight: float,
) -> Dict[str, Any]:
    p_blend = cb_weight * p_cb + (1.0 - cb_weight) * p_dl
    err_cb = p_cb - y
    err_dl = p_dl - y

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "cb": subgroup_brier(df, cfg, y, p_cb),
        "dl": subgroup_brier(df, cfg, y, p_dl),
        "blend": subgroup_brier(df, cfg, y, p_blend),
        "delta_blend_minus_cb": safe_brier(y, p_blend) - safe_brier(y, p_cb),
        "prediction_correlation": corr(p_cb, p_dl),
        "error_correlation": corr(err_cb, err_dl),
        "error_covariance": float(np.cov(err_cb, err_dl, ddof=1)[0, 1]) if len(y) > 1 else float("nan"),
        "blend_weight_cb": float(cb_weight),
    }


def coarse_blend_sensitivity(y: np.ndarray, p_cb: np.ndarray, p_dl: np.ndarray) -> Dict[str, float]:
    return {
        f"cb_{round(w*100)}_dl_{round((1-w)*100)}": safe_brier(y, w*p_cb + (1-w)*p_dl)
        for w in (0.90, 0.80, 0.70)
    }


def _prepare_cb_frame(df: pd.DataFrame, features: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    x = df.loc[:, list(features)].copy()
    cat_cols = [c for c in CAT_COLS if c in x.columns]
    for c in x.columns:
        if c in cat_cols:
            x[c] = x[c].astype("string").fillna("__NA__").astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x, cat_cols


def fit_predict_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: Sequence[str],
    cfg: PipelineConfig,
    seed: int,
) -> np.ndarray:
    x_train, cat_cols = _prepare_cb_frame(train_df, features)
    x_val, _ = _prepare_cb_frame(val_df, features)
    y_train = train_df[cfg.target_col].to_numpy(dtype=int)
    w_train = compute_sample_weights(train_df, cfg)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="BrierScore",
        iterations=cfg.cb_iterations,
        learning_rate=cfg.cb_learning_rate,
        depth=cfg.cb_depth,
        l2_leaf_reg=cfg.cb_l2_leaf_reg,
        random_strength=cfg.cb_random_strength,
        max_ctr_complexity=cfg.cb_max_ctr_complexity,
        border_count=cfg.cb_border_count,
        random_seed=seed,
        task_type="CPU",
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    train_pool = Pool(x_train, label=y_train, weight=w_train, cat_features=cat_cols)
    model.fit(train_pool)
    return model.predict_proba(x_val)[:, 1]


def dev_fold_masks(df: pd.DataFrame, cfg: PipelineConfig) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
    season = df[cfg.season_col].to_numpy()
    folds = []
    for val_year, fold_weight in (
        (cfg.dev_val_year_1, 1.0),
        (cfg.dev_val_year_2, cfg.dev_2023_weight),
    ):
        tr = (season >= cfg.train_start_year) & (season < val_year)
        va = season == val_year
        if tr.any() and va.any():
            folds.append((str(val_year), tr, va, float(fold_weight)))
    if not folds:
        raise ValueError("No valid development folds were found.")
    return folds


def evaluate_cb_arm_on_dev(
    df: pd.DataFrame,
    features: Sequence[str],
    cfg: PipelineConfig,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    fold_results = []
    weighted_scores = []
    weighted_weights = []

    for fold_name, tr_mask, va_mask, fold_weight in dev_fold_masks(df, cfg):
        tr_df = df.loc[tr_mask]
        va_df = df.loc[va_mask]
        y = va_df[cfg.target_col].to_numpy(dtype=float)
        seed_scores = []
        for seed in seeds:
            p = fit_predict_catboost(tr_df, va_df, features, cfg, seed)
            seed_scores.append(safe_brier(y, p))
        mean_score = float(np.mean(seed_scores))
        std_score = float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0
        fold_results.append(
            {
                "validation_year": int(fold_name),
                "weight": fold_weight,
                "brier_by_seed": {str(s): float(v) for s, v in zip(seeds, seed_scores)},
                "mean_brier": mean_score,
                "std_brier": std_score,
            }
        )
        weighted_scores.append(mean_score * fold_weight)
        weighted_weights.append(fold_weight)

    aggregate = float(np.sum(weighted_scores) / np.sum(weighted_weights))
    return {"folds": fold_results, "weighted_dev_brier": aggregate}


def catboost_dev_screen(
    df: pd.DataFrame,
    arms: Mapping[str, Sequence[str]],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    primary_seed = cfg.seeds[0]
    screening = {}
    for arm, features in arms.items():
        if arm != "T0_BASE" and len(features) == len(arms["T0_BASE"]):
            screening[arm] = {"skipped": True, "reason": "no additional features available"}
            continue
        screening[arm] = evaluate_cb_arm_on_dev(df, features, cfg, [primary_seed])

    ranked_screen = sorted(
        [
            (arm, res["weighted_dev_brier"])
            for arm, res in screening.items()
            if not res.get("skipped", False)
        ],
        key=lambda x: x[1],
    )
    # Keep a few more than the final locked candidate count for 3-seed confirmation.
    survivor_count = min(len(ranked_screen), max(cfg.catboost_locked_candidates + 1, cfg.catboost_locked_candidates))
    survivors = [a for a, _ in ranked_screen[:survivor_count]]

    confirmation = {}
    for arm in survivors:
        confirmation[arm] = evaluate_cb_arm_on_dev(df, arms[arm], cfg, cfg.seeds)

    ranked_confirmed = sorted(
        [(arm, res["weighted_dev_brier"]) for arm, res in confirmation.items()],
        key=lambda x: x[1],
    )
    locked_candidates = [a for a, _ in ranked_confirmed[: cfg.catboost_locked_candidates]]

    return {
        "one_seed_screening": screening,
        "screening_rank": ranked_screen,
        "three_seed_confirmation": confirmation,
        "confirmed_rank": ranked_confirmed,
        "locked_candidates": locked_candidates,
    }


def catboost_locked_eval(
    df: pd.DataFrame,
    arms: Mapping[str, Sequence[str]],
    candidates: Sequence[str],
    cfg: PipelineConfig,
) -> Tuple[Dict[str, Any], str, np.ndarray]:
    season = df[cfg.season_col].to_numpy()
    tr_df = df.loc[(season >= cfg.train_start_year) & (season < cfg.locked_val_year)]
    va_df = df.loc[season == cfg.locked_val_year]
    if va_df.empty:
        raise ValueError(f"No locked validation rows for {cfg.locked_val_year}")
    y = va_df[cfg.target_col].to_numpy(dtype=float)

    results: Dict[str, Any] = {}
    primary_predictions: Dict[str, np.ndarray] = {}
    for arm in candidates:
        seed_scores = []
        for seed in cfg.seeds:
            p = fit_predict_catboost(tr_df, va_df, arms[arm], cfg, seed)
            seed_scores.append(safe_brier(y, p))
            if seed == cfg.seeds[0]:
                primary_predictions[arm] = p
        results[arm] = {
            "brier_by_seed": {str(s): float(v) for s, v in zip(cfg.seeds, seed_scores)},
            "mean_brier": float(np.mean(seed_scores)),
            "std_brier": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0,
            "primary_seed_brier": float(seed_scores[0]),
        }

    # One-time locked decision among pre-registered candidates; no subsequent tuning.
    chosen = min(candidates, key=lambda a: results[a]["mean_brier"])
    return results, chosen, primary_predictions[chosen]





# -----------------------------------------------------------------------------
# Actual notebook-backed Transformer state schema
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class StateSchema:
    contexts: Tuple[str, ...]
    times: Tuple[str, ...]
    metrics: Tuple[str, ...]
    cells: Tuple[Tuple[str, str], ...]
    source_map: Mapping[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contexts": list(self.contexts),
            "times": list(self.times),
            "metrics": list(self.metrics),
            "cells": [list(x) for x in self.cells],
            "source_map": dict(self.source_map),
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "StateSchema":
        return cls(
            contexts=tuple(obj["contexts"]),
            times=tuple(obj["times"]),
            metrics=tuple(obj["metrics"]),
            cells=tuple(tuple(x) for x in obj["cells"]),
            source_map=dict(obj["source_map"]),
        )


def _state_key(context: str, time_name: str, field: str) -> str:
    return f"{context}|{time_name}|{field}"


def build_notebook_state_schema(df: pd.DataFrame) -> StateSchema:
    source_map: Dict[str, str] = {}
    cells: List[Tuple[str, str]] = []

    for context, group in TRANSFORMER_CONTEXT_MAP.items():
        for time_name in TRANSFORMER_TIMES:
            if time_name == "Career":
                prefix = "flat"
            else:
                lag = {"Prev1": 1, "Prev2": 2, "Prev3": 3}[time_name]
                prefix = f"temporal_prev{lag}"

            for metric in TRANSFORMER_METRICS:
                col = f"{prefix}_{group}_flag_{metric}_rate"
                if col not in df.columns:
                    raise RuntimeError(f"State source column missing: {col}")
                source_map[_state_key(context, time_name, metric)] = col

            n_col = flat_n_column(prefix, group, "flag_success")
            if n_col not in df.columns:
                raise RuntimeError(f"State n source column missing: {n_col}")
            source_map[_state_key(context, time_name, "n")] = n_col
            cells.append((context, time_name))

    return StateSchema(
        contexts=tuple(TRANSFORMER_CONTEXT_MAP.keys()),
        times=tuple(TRANSFORMER_TIMES),
        metrics=tuple(TRANSFORMER_METRICS),
        cells=tuple(cells),
        source_map=source_map,
    )


def build_state_arrays(
    df: pd.DataFrame,
    schema: StateSchema,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Values=[8 rates, log1p(n), availability]; missing cells use learned embedding."""
    context_to_id = {c: i for i, c in enumerate(schema.contexts)}
    time_to_id = {t: i for i, t in enumerate(schema.times)}
    n_rows = len(df)
    n_cells = len(schema.cells)
    dim = len(schema.metrics) + 2

    values = np.zeros((n_rows, n_cells, dim), dtype=np.float32)
    availability = np.zeros((n_rows, n_cells), dtype=np.float32)
    context_ids = np.zeros(n_cells, dtype=np.int64)
    time_ids = np.zeros(n_cells, dtype=np.int64)

    for j, (context, time_name) in enumerate(schema.cells):
        context_ids[j] = context_to_id[context]
        time_ids[j] = time_to_id[time_name]

        for k, metric in enumerate(schema.metrics):
            col = schema.source_map[_state_key(context, time_name, metric)]
            values[:, j, k] = (
                pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            )

        n_col = schema.source_map[_state_key(context, time_name, "n")]
        n_raw = (
            pd.to_numeric(df[n_col], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=np.float32)
        )
        a_raw = (n_raw > 0).astype(np.float32)
        values[:, j, len(schema.metrics)] = np.log1p(n_raw)
        values[:, j, len(schema.metrics) + 1] = a_raw
        availability[:, j] = a_raw

    return values, availability, context_ids, time_ids


@dataclass
class QueryPreprocessor:
    columns: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    medians: Dict[str, float]
    means: Dict[str, float]
    stds: Dict[str, float]
    encoder: Optional[OneHotEncoder]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: Sequence[str]) -> "QueryPreprocessor":
        cols = list(columns)
        if not cols:
            raise ValueError("No query features found. Add query__* current-context columns.")

        num = []
        cat = []
        for c in cols:
            if c in QUERY_CATEGORICAL_COLUMNS:
                cat.append(c)
            elif pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
                num.append(c)
            else:
                cat.append(c)

        medians: Dict[str, float] = {}
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for c in num:
            s = pd.to_numeric(df[c], errors="coerce")
            med = float(s.median()) if np.isfinite(s.median()) else 0.0
            filled = s.fillna(med).astype(float)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            medians[c], means[c], stds[c] = med, mean, std

        encoder = None
        if cat:
            try:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            except TypeError:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
            encoder.fit(df[cat].astype("string").fillna("__MISSING__"))

        return cls(cols, num, cat, medians, means, stds, encoder)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts: List[np.ndarray] = []
        if self.numeric_cols:
            arr = []
            for c in self.numeric_cols:
                s = pd.to_numeric(df[c], errors="coerce").fillna(self.medians[c]).astype(float)
                arr.append(((s.to_numpy() - self.means[c]) / self.stds[c]).astype(np.float32))
            parts.append(np.stack(arr, axis=1))
        if self.categorical_cols:
            assert self.encoder is not None
            cat_arr = self.encoder.transform(df[self.categorical_cols].astype("string").fillna("__MISSING__"))
            parts.append(np.asarray(cat_arr, dtype=np.float32))
        if not parts:
            raise RuntimeError("Query transform produced no features")
        return np.concatenate(parts, axis=1)


# -----------------------------------------------------------------------------
# DL models
# -----------------------------------------------------------------------------


class StructuredTransformer(nn.Module):
    def __init__(
        self,
        value_dim: int,
        query_dim: int,
        n_contexts: int,
        n_times: int,
        context_ids: np.ndarray,
        time_ids: np.ndarray,
        cfg: PipelineConfig,
    ) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.value_encoder = nn.Sequential(
            nn.Linear(value_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.context_embedding = nn.Embedding(n_contexts, d)
        self.time_embedding = nn.Embedding(n_times, d)
        self.missing_embedding = nn.Parameter(torch.zeros(d))
        self.query_embedding = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.missing_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.query_embedding, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, norm=nn.LayerNorm(d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        self.register_buffer("context_ids", torch.tensor(context_ids, dtype=torch.long), persistent=False)
        self.register_buffer("time_ids", torch.tensor(time_ids, dtype=torch.long), persistent=False)

    def forward(self, values: torch.Tensor, availability: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # values: [B,T,D], availability: [B,T]
        bsz, n_tokens, _ = values.shape
        value_emb = self.value_encoder(values)
        missing = self.missing_embedding.view(1, 1, -1).expand(bsz, n_tokens, -1)
        avail = availability.unsqueeze(-1)
        state = avail * value_emb + (1.0 - avail) * missing

        pos = self.context_embedding(self.context_ids) + self.time_embedding(self.time_ids)
        state = state + pos.unsqueeze(0)

        # Token Dropout: replace a state token with the semantic missing token at training time.
        if self.training and self.cfg.token_dropout > 0:
            drop = torch.rand((bsz, n_tokens), device=values.device) < self.cfg.token_dropout
            replacement = missing + pos.unsqueeze(0)
            state = torch.where(drop.unsqueeze(-1), replacement, state)

        # Current QUERY token replaces a generic CLS token.
        query_token = self.query_encoder(query) + self.query_embedding.view(1, -1)
        seq = torch.cat([state, query_token.unsqueeze(1)], dim=1)
        out = self.transformer(seq)
        query_out = out[:, -1, :]
        return self.head(query_out).squeeze(-1)


class FlattenMLP(nn.Module):
    def __init__(self, input_dim: int, cfg: PipelineConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.mlp_hidden_1),
            nn.GELU(),
            nn.Dropout(cfg.mlp_dropout),
            nn.Linear(cfg.mlp_hidden_1, cfg.mlp_hidden_2),
            nn.GELU(),
            nn.Dropout(cfg.mlp_dropout),
            nn.Linear(cfg.mlp_hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _weighted_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * w).sum() / torch.clamp(w.sum(), min=1e-8)


def _predict_transformer(
    model: StructuredTransformer,
    values: np.ndarray,
    availability: np.ndarray,
    query: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    ds = TensorDataset(
        torch.from_numpy(values),
        torch.from_numpy(availability),
        torch.from_numpy(query),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.inference_mode():
        for v, a, q in loader:
            logits = model(v.to(device), a.to(device), q.to(device))
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def _epoch_loader(ds: TensorDataset, batch_size: int, seed: int, epoch: int) -> DataLoader:
    """Deterministic per-epoch shuffling so interrupted/resumed runs match uninterrupted runs."""
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _append_history_csv(history: Sequence[Mapping[str, Any]], path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(history)).to_csv(path, index=False)


def _save_training_checkpoint(
    path: Optional[Path],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    best_epoch: int,
    best_score: float,
    best_state: Optional[Mapping[str, torch.Tensor]],
    no_improve_evals: int,
    stopped_early: bool,
    signature: Mapping[str, Any],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state": _cpu_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "history": list(history),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_state": dict(best_state) if best_state is not None else None,
        "no_improve_evals": int(no_improve_evals),
        "stopped_early": bool(stopped_early),
        "signature": dict(signature),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m:d}m {sec:02d}s"
    return f"{sec:d}s"


def train_transformer_epochs(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    schema: StateSchema,
    query_cols: Sequence[str],
    cfg: PipelineConfig,
    seed: int,
    epochs: int,
    device: torch.device,
    track_each_epoch: bool = False,
    checkpoint_path: Optional[Path] = None,
    history_csv_path: Optional[Path] = None,
    allow_resume: bool = False,
    early_stopping: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    set_global_seed(seed)
    qprep = QueryPreprocessor.fit(train_df, query_cols)
    q_train = qprep.transform(train_df)
    q_pred = qprep.transform(pred_df)
    v_train, a_train, context_ids, time_ids = build_state_arrays(train_df, schema, cfg)
    v_pred, a_pred, _, _ = build_state_arrays(pred_df, schema, cfg)

    model = StructuredTransformer(
        value_dim=v_train.shape[-1],
        query_dim=q_train.shape[-1],
        n_contexts=len(schema.contexts),
        n_times=len(schema.times),
        context_ids=context_ids,
        time_ids=time_ids,
        cfg=cfg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.dl_lr, weight_decay=cfg.dl_weight_decay)
    y_train = train_df[cfg.target_col].to_numpy(dtype=np.float32)
    w_train = compute_sample_weights(train_df, cfg).astype(np.float32)
    ds = TensorDataset(
        torch.from_numpy(v_train), torch.from_numpy(a_train), torch.from_numpy(q_train),
        torch.from_numpy(y_train), torch.from_numpy(w_train),
    )
    y_pred_true = pred_df[cfg.target_col].to_numpy(dtype=float) if cfg.target_col in pred_df.columns else None

    signature = {
        "kind": "transformer", "seed": int(seed), "epochs": int(epochs),
        "train_rows": int(len(train_df)), "pred_rows": int(len(pred_df)),
        "value_dim": int(v_train.shape[-1]), "query_dim": int(q_train.shape[-1]),
        "batch_size": int(cfg.dl_batch_size), "lr": float(cfg.dl_lr),
        "weight_decay": float(cfg.dl_weight_decay), "d_model": int(cfg.d_model),
        "n_heads": int(cfg.n_heads), "n_layers": int(cfg.n_layers),
        "ffn_dim": int(cfg.ffn_dim), "dropout": float(cfg.dropout),
        "token_dropout": float(cfg.token_dropout),
    }
    history: List[Dict[str, float]] = []
    best_score = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve_evals = 0
    start_epoch = 1
    stopped_early = False

    if allow_resume and checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        old_sig = ckpt.get("signature", {})
        if old_sig != signature:
            raise RuntimeError(
                f"Checkpoint signature mismatch at {checkpoint_path}. "
                "Use a new experiment id or remove the stale checkpoint."
            )
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = list(ckpt.get("history", []))
        best_epoch = int(ckpt.get("best_epoch", 0))
        best_score = float(ckpt.get("best_score", float("inf")))
        best_state = ckpt.get("best_state")
        no_improve_evals = int(ckpt.get("no_improve_evals", 0))
        stopped_early = bool(ckpt.get("stopped_early", False))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(
            f"[Transformer] resume checkpoint: epoch={start_epoch-1}, "
            f"best_epoch={best_epoch}, best_brier={best_score:.8f}"
        )

    n_batches = math.ceil(len(ds) / max(cfg.dl_batch_size, 1))
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Transformer] device={device} train_rows={len(train_df):,} val_rows={len(pred_df):,} "
        f"batch={cfg.dl_batch_size:,} batches/epoch={n_batches:,} params={n_params:,}"
    )

    if stopped_early or start_epoch > epochs:
        if best_state is not None:
            model.load_state_dict(best_state)
        p_final = _predict_transformer(model, v_pred, a_pred, q_pred, device, cfg.dl_batch_size)
        return p_final, history

    total_start = time.perf_counter()
    # If resuming, elapsed below intentionally measures the current invocation only.
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        batch_count = 0
        loader = _epoch_loader(ds, cfg.dl_batch_size, seed, epoch)
        for v, a, q, y, w in loader:
            v, a, q, y, w = v.to(device), a.to(device), q.to(device), y.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(v, a, q)
            loss = _weighted_bce(logits, y, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batch_count += 1

        train_loss = loss_sum / max(batch_count, 1)
        do_eval = track_each_epoch and (epoch % max(cfg.dl_eval_every, 1) == 0 or epoch == epochs)
        score = float("nan")
        is_best = False
        if do_eval:
            p = _predict_transformer(model, v_pred, a_pred, q_pred, device, cfg.dl_batch_size)
            score = safe_brier(y_pred_true, p) if y_pred_true is not None else float("nan")
            if np.isfinite(score) and score < best_score - cfg.dl_min_delta:
                best_score = float(score)
                best_epoch = int(epoch)
                best_state = _cpu_state_dict(model)
                no_improve_evals = 0
                is_best = True
            elif epoch >= cfg.dl_min_epochs:
                no_improve_evals += 1

        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - total_start
        if do_eval:
            history.append({
                "epoch": epoch, "train_loss": train_loss, "pred_brier": score,
                "epoch_seconds": epoch_seconds, "elapsed_seconds": elapsed_seconds,
                "is_best": bool(is_best), "no_improve_evals": int(no_improve_evals),
                "batch_size": int(cfg.dl_batch_size), "device": str(device),
            })
            _append_history_csv(history, history_csv_path)

        should_log = (
            epoch == start_epoch or epoch == epochs or do_eval or
            epoch % max(cfg.dl_log_every, 1) == 0
        )
        if should_log:
            val_text = f"val_brier={score:.8f}" if do_eval and np.isfinite(score) else "val_brier=skipped"
            best_text = f"best={best_score:.8f}@{best_epoch}" if np.isfinite(best_score) else "best=n/a"
            marker = " *BEST*" if is_best else ""
            print(
                f"[Transformer] epoch {epoch:03d}/{epochs:03d} | train_loss={train_loss:.6f} | "
                f"{val_text} | {best_text} | patience={no_improve_evals}/{cfg.dl_patience} | "
                f"epoch={_format_seconds(epoch_seconds)} | run={_format_seconds(elapsed_seconds)}{marker}",
                flush=True,
            )

        if do_eval:
            stop_now = bool(
                early_stopping and epoch >= cfg.dl_min_epochs and
                no_improve_evals >= max(cfg.dl_patience, 1)
            )
            _save_training_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, epoch=epoch, history=history,
                best_epoch=best_epoch, best_score=best_score, best_state=best_state,
                no_improve_evals=no_improve_evals, stopped_early=stop_now, signature=signature,
            )
            if stop_now:
                stopped_early = True
                print(
                    f"[Transformer] early stopping at epoch {epoch}; "
                    f"best epoch={best_epoch}, best_brier={best_score:.8f}",
                    flush=True,
                )
                break

    if track_each_epoch and best_state is not None:
        model.load_state_dict(best_state)
    p_final = _predict_transformer(model, v_pred, a_pred, q_pred, device, cfg.dl_batch_size)
    return p_final, history


def _mlp_features(values: np.ndarray, availability: np.ndarray, query: np.ndarray) -> np.ndarray:
    # Same state matrix sanity baseline. Availability is already the final value channel,
    # but concatenating it separately makes missingness explicit after flattening too.
    return np.concatenate([values.reshape(len(values), -1), availability, query], axis=1).astype(np.float32)


def _predict_mlp(model: FlattenMLP, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    out = []
    with torch.inference_mode():
        for (xb,) in loader:
            out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out)


def train_mlp_epochs(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    schema: StateSchema,
    query_cols: Sequence[str],
    cfg: PipelineConfig,
    seed: int,
    epochs: int,
    device: torch.device,
    track_each_epoch: bool = False,
    checkpoint_path: Optional[Path] = None,
    history_csv_path: Optional[Path] = None,
    allow_resume: bool = False,
    early_stopping: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    set_global_seed(seed)
    qprep = QueryPreprocessor.fit(train_df, query_cols)
    q_train = qprep.transform(train_df)
    q_pred = qprep.transform(pred_df)
    v_train, a_train, _, _ = build_state_arrays(train_df, schema, cfg)
    v_pred, a_pred, _, _ = build_state_arrays(pred_df, schema, cfg)
    x_train = _mlp_features(v_train, a_train, q_train)
    x_pred = _mlp_features(v_pred, a_pred, q_pred)

    model = FlattenMLP(x_train.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.dl_lr, weight_decay=cfg.dl_weight_decay)
    y_train = train_df[cfg.target_col].to_numpy(dtype=np.float32)
    w_train = compute_sample_weights(train_df, cfg).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(w_train))
    y_pred_true = pred_df[cfg.target_col].to_numpy(dtype=float) if cfg.target_col in pred_df.columns else None

    signature = {
        "kind": "mlp", "seed": int(seed), "epochs": int(epochs),
        "train_rows": int(len(train_df)), "pred_rows": int(len(pred_df)),
        "input_dim": int(x_train.shape[1]), "batch_size": int(cfg.dl_batch_size),
        "lr": float(cfg.dl_lr), "weight_decay": float(cfg.dl_weight_decay),
        "hidden_1": int(cfg.mlp_hidden_1), "hidden_2": int(cfg.mlp_hidden_2),
        "dropout": float(cfg.mlp_dropout),
    }
    history: List[Dict[str, float]] = []
    best_score = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve_evals = 0
    start_epoch = 1
    stopped_early = False

    if allow_resume and checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if ckpt.get("signature", {}) != signature:
            raise RuntimeError(f"Checkpoint signature mismatch at {checkpoint_path}.")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = list(ckpt.get("history", []))
        best_epoch = int(ckpt.get("best_epoch", 0))
        best_score = float(ckpt.get("best_score", float("inf")))
        best_state = ckpt.get("best_state")
        no_improve_evals = int(ckpt.get("no_improve_evals", 0))
        stopped_early = bool(ckpt.get("stopped_early", False))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[MLP] resume checkpoint: epoch={start_epoch-1}, best_epoch={best_epoch}, best_brier={best_score:.8f}")

    n_batches = math.ceil(len(ds) / max(cfg.dl_batch_size, 1))
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[MLP] device={device} train_rows={len(train_df):,} val_rows={len(pred_df):,} "
        f"batch={cfg.dl_batch_size:,} batches/epoch={n_batches:,} params={n_params:,}"
    )

    if stopped_early or start_epoch > epochs:
        if best_state is not None:
            model.load_state_dict(best_state)
        return _predict_mlp(model, x_pred, device, cfg.dl_batch_size), history

    total_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        batch_count = 0
        loader = _epoch_loader(ds, cfg.dl_batch_size, seed, epoch)
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = _weighted_bce(logits, y, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batch_count += 1

        train_loss = loss_sum / max(batch_count, 1)
        do_eval = track_each_epoch and (epoch % max(cfg.dl_eval_every, 1) == 0 or epoch == epochs)
        score = float("nan")
        is_best = False
        if do_eval:
            p = _predict_mlp(model, x_pred, device, cfg.dl_batch_size)
            score = safe_brier(y_pred_true, p) if y_pred_true is not None else float("nan")
            if np.isfinite(score) and score < best_score - cfg.dl_min_delta:
                best_score = float(score)
                best_epoch = int(epoch)
                best_state = _cpu_state_dict(model)
                no_improve_evals = 0
                is_best = True
            elif epoch >= cfg.dl_min_epochs:
                no_improve_evals += 1

        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - total_start
        if do_eval:
            history.append({
                "epoch": epoch, "train_loss": train_loss, "pred_brier": score,
                "epoch_seconds": epoch_seconds, "elapsed_seconds": elapsed_seconds,
                "is_best": bool(is_best), "no_improve_evals": int(no_improve_evals),
                "batch_size": int(cfg.dl_batch_size), "device": str(device),
            })
            _append_history_csv(history, history_csv_path)

        if epoch == start_epoch or epoch == epochs or do_eval or epoch % max(cfg.dl_log_every, 1) == 0:
            val_text = f"val_brier={score:.8f}" if do_eval and np.isfinite(score) else "val_brier=skipped"
            best_text = f"best={best_score:.8f}@{best_epoch}" if np.isfinite(best_score) else "best=n/a"
            marker = " *BEST*" if is_best else ""
            print(
                f"[MLP] epoch {epoch:03d}/{epochs:03d} | train_loss={train_loss:.6f} | {val_text} | "
                f"{best_text} | patience={no_improve_evals}/{cfg.dl_patience} | "
                f"epoch={_format_seconds(epoch_seconds)} | run={_format_seconds(elapsed_seconds)}{marker}",
                flush=True,
            )

        if do_eval:
            stop_now = bool(early_stopping and epoch >= cfg.dl_min_epochs and no_improve_evals >= max(cfg.dl_patience, 1))
            _save_training_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, epoch=epoch, history=history,
                best_epoch=best_epoch, best_score=best_score, best_state=best_state,
                no_improve_evals=no_improve_evals, stopped_early=stop_now, signature=signature,
            )
            if stop_now:
                stopped_early = True
                print(f"[MLP] early stopping at epoch {epoch}; best epoch={best_epoch}, best_brier={best_score:.8f}", flush=True)
                break

    if track_each_epoch and best_state is not None:
        model.load_state_dict(best_state)
    return _predict_mlp(model, x_pred, device, cfg.dl_batch_size), history


def select_epoch_on_2023(
    df: pd.DataFrame,
    schema: StateSchema,
    query_cols: Sequence[str],
    cfg: PipelineConfig,
    seed: int,
    device: torch.device,
    model_kind: str,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    season = df[cfg.season_col].to_numpy()
    train_df = df.loc[(season >= cfg.train_start_year) & (season < cfg.dev_val_year_2)]
    val_df = df.loc[season == cfg.dev_val_year_2]
    if train_df.empty or val_df.empty:
        raise ValueError("Cannot select DL epoch: 2019~2022 train or 2023 validation is empty.")

    checkpoint_path = None
    history_csv_path = None
    if out_dir is not None:
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / f"dev_{model_kind}_epoch_selection_seed{seed}.pt"
        history_csv_path = out_dir / f"dev_{model_kind}_epoch_curve.csv"

    common = dict(
        checkpoint_path=checkpoint_path,
        history_csv_path=history_csv_path,
        allow_resume=cfg.dl_resume,
        early_stopping=True,
    )
    if model_kind == "transformer":
        _, hist = train_transformer_epochs(
            train_df, val_df, schema, query_cols, cfg, seed, cfg.dl_max_epochs,
            device, track_each_epoch=True, **common
        )
    elif model_kind == "mlp":
        _, hist = train_mlp_epochs(
            train_df, val_df, schema, query_cols, cfg, seed, cfg.dl_max_epochs,
            device, track_each_epoch=True, **common
        )
    else:
        raise ValueError(model_kind)

    if not hist:
        raise RuntimeError(f"No validation history produced for {model_kind}.")
    best = min(hist, key=lambda x: x["pred_brier"])
    return {
        "model_kind": model_kind,
        "seed": seed,
        "best_epoch": int(best["epoch"]),
        "best_2023_brier": float(best["pred_brier"]),
        "stopped_after_epoch": int(hist[-1]["epoch"]),
        "early_stopping": {
            "min_epochs": int(cfg.dl_min_epochs),
            "patience": int(cfg.dl_patience),
            "min_delta": float(cfg.dl_min_delta),
            "eval_every": int(cfg.dl_eval_every),
        },
        "history": hist,
    }


def locked_dl_eval(
    df: pd.DataFrame,
    schema: StateSchema,
    query_cols: Sequence[str],
    cfg: PipelineConfig,
    epochs: int,
    device: torch.device,
    model_kind: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    season = df[cfg.season_col].to_numpy()
    train_df = df.loc[(season >= cfg.train_start_year) & (season < cfg.locked_val_year)]
    val_df = df.loc[season == cfg.locked_val_year]
    y = val_df[cfg.target_col].to_numpy(dtype=float)

    scores = {}
    primary_pred = None
    for seed in cfg.seeds:
        if model_kind == "transformer":
            p, _ = train_transformer_epochs(train_df, val_df, schema, query_cols, cfg, seed, epochs, device)
        elif model_kind == "mlp":
            p, _ = train_mlp_epochs(train_df, val_df, schema, query_cols, cfg, seed, epochs, device)
        else:
            raise ValueError(model_kind)
        scores[str(seed)] = safe_brier(y, p)
        if seed == cfg.seeds[0]:
            primary_pred = p

    vals = list(scores.values())
    result = {
        "epochs": int(epochs),
        "brier_by_seed": {k: float(v) for k, v in scores.items()},
        "mean_brier": float(np.mean(vals)),
        "std_brier": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "primary_seed_brier": float(vals[0]),
    }
    assert primary_pred is not None
    return result, primary_pred





# -----------------------------------------------------------------------------
# Prepared-cache IO
# -----------------------------------------------------------------------------

PREPARED_FILENAME = "prepared_features.pkl"
PREPARED_MANIFEST = "prepared_manifest.json"


def resolve_data_paths(project_dir: Path) -> Dict[str, Path]:
    data_dir = project_dir / "data"
    return {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample_submission": data_dir / "sample_submission.csv",
        "trackman_history": data_dir / "trackman_history.csv",
    }


def prepare_and_cache(project_dir: Path, out_dir: Path, cfg: PipelineConfig, force: bool = False) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / PREPARED_FILENAME
    manifest_path = out_dir / PREPARED_MANIFEST
    paths = resolve_data_paths(project_dir)
    raw_fp = raw_file_fingerprint(paths)

    if cache_path.exists() and manifest_path.exists() and not force:
        old = load_json(manifest_path)
        if old.get("raw_fingerprint", {}).get("sha256") == raw_fp["sha256"]:
            same_scope = old.get("config", {}).get("temporal_history_scope") == cfg.temporal_history_scope
            same_code = old.get("code_sha256") == current_code_sha256()
            if same_scope and same_code:
                print(f"[prepare] reuse cache: {cache_path}")
                return pd.read_pickle(cache_path)
        print("[prepare] cache fingerprint/config mismatch -> rebuilding")

    if not paths["train"].exists() or not paths["test"].exists():
        raise FileNotFoundError(
            f"Expected {paths['train']} and {paths['test']}. "
            "Use --project-dir to point at the notebook project root."
        )

    train_raw = pd.read_csv(paths["train"])
    test_raw = pd.read_csv(paths["test"])
    df = prepare_feature_table(train_raw, test_raw, cfg)
    contract = save_prepared_contract(df, cfg, out_dir, raw_fp)
    fp = contract["prepared_fingerprint"]

    df.to_pickle(cache_path)
    manifest = {
        "created_at": now_iso(),
        "raw_fingerprint": raw_fp,
        "prepared_fingerprint": fp,
        "code_sha256": current_code_sha256(),
        "config": asdict(cfg),
        "cache_path": str(cache_path.resolve()),
    }
    save_json(manifest, manifest_path)
    print(f"[prepare] saved: {cache_path}")
    return df


def load_prepared(project_dir: Path, out_dir: Path, cfg: Optional[PipelineConfig] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    manifest_path = out_dir / PREPARED_MANIFEST
    cache_path = out_dir / PREPARED_FILENAME
    if not manifest_path.exists() or not cache_path.exists():
        if cfg is None:
            raise FileNotFoundError("Prepared cache missing; run --stage prepare or --stage dev first.")
        df = prepare_and_cache(project_dir, out_dir, cfg)
        return df, load_json(manifest_path)

    manifest = load_json(manifest_path)
    paths = resolve_data_paths(project_dir)
    current_raw = raw_file_fingerprint(paths)
    if current_raw["sha256"] != manifest["raw_fingerprint"]["sha256"]:
        raise RuntimeError(
            "Raw data fingerprint changed after preparation. "
            "Use a new output directory or rerun --stage prepare --force-prepare."
        )
    df = pd.read_pickle(cache_path)
    return df, manifest


def validate_prepared(df: pd.DataFrame, cfg: PipelineConfig, require_test: bool = False) -> None:
    required = [ID, "season", TARGET, "game_type", "_dataset_split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Prepared table missing columns: {missing}")
    if df[ID].duplicated().any():
        raise RuntimeError("Prepared row_id must be unique.")
    if require_test and not df["_dataset_split"].eq("test").any():
        raise RuntimeError("No test rows in prepared table.")


# -----------------------------------------------------------------------------
# Stage orchestration
# -----------------------------------------------------------------------------

def make_dev_manifest(
    df: pd.DataFrame,
    prep_manifest: Mapping[str, Any],
    cfg: PipelineConfig,
    arms: Mapping[str, Sequence[str]],
    cb_dev: Mapping[str, Any],
    schema: StateSchema,
    dl_epoch: Mapping[str, Any],
    mlp_epoch: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest = {
        "created_at": now_iso(),
        "prepared_fingerprint": prep_manifest["prepared_fingerprint"],
        "raw_fingerprint": prep_manifest["raw_fingerprint"],
        "code_sha256": current_code_sha256(),
        "package_versions": package_versions(),
        "config": asdict(cfg),
        "base_features": list(FEATURES),
        "catboost_arms": {k: list(v) for k, v in arms.items()},
        "catboost_dev": cb_dev,
        "locked_catboost_candidates": list(cb_dev["locked_candidates"]),
        "state_schema": schema.to_dict(),
        "query_columns": list(QUERY_COLUMNS),
        "transformer_epoch_selection": dl_epoch,
        "mlp_epoch_selection": mlp_epoch,
        "blend_decision_rule": (
            "Evaluate fixed CB=0.8/DL=0.2 first on locked 2024. "
            "If and only if it improves CatBoost, evaluate CB weights {0.9,0.8,0.7} "
            "and freeze the best coarse weight; otherwise final weight is CB=1.0."
        ),
    }
    manifest["frozen_signature"] = sha256_json(manifest)
    return manifest


def run_dev(project_dir: Path, out_dir: Path, cfg: PipelineConfig, device: torch.device, force_prepare: bool) -> Dict[str, Any]:
    if (out_dir / "locked_manifest.json").exists():
        raise RuntimeError(
            "This output directory already contains locked_manifest.json. "
            "Do not tune after opening the 2024 locked result; use a new output directory."
        )

    df = prepare_and_cache(project_dir, out_dir, cfg, force=force_prepare)
    prep_manifest = load_json(out_dir / PREPARED_MANIFEST)
    validate_prepared(df, cfg)

    dev_df = df.loc[
        df["_dataset_split"].eq("train")
        & df["season"].between(cfg.train_start_year, cfg.dev_val_year_2)
    ].copy()
    if dev_df[TARGET].isna().any():
        raise RuntimeError("Development target contains NaN.")

    arms = build_catboost_arms(dev_df)
    print("[dev] CatBoost one-seed screening -> 3-seed confirmation")
    cb_dev = catboost_dev_screen(dev_df, arms, cfg)
    base_score = dict(cb_dev["confirmed_rank"]).get("T0_BASE")
    print("[dev] CatBoost confirmed rank (3-seed):")
    for rank_i, (arm_name, arm_score) in enumerate(cb_dev["confirmed_rank"], start=1):
        delta = arm_score - base_score if base_score is not None else float("nan")
        print(f"  {rank_i:>2}. {arm_name:<28} Brier={arm_score:.8f}  ΔvsBase={delta:+.8f}")
    print("[dev] locked CatBoost candidates:", cb_dev["locked_candidates"])

    schema = build_notebook_state_schema(dev_df)
    missing_query = [c for c in QUERY_COLUMNS if c not in dev_df.columns]
    if missing_query:
        raise RuntimeError(f"Query columns missing: {missing_query}")

    print("[dev] select Transformer epoch: train 2019~2022 -> valid 2023")
    dl_epoch = select_epoch_on_2023(
        dev_df, schema, QUERY_COLUMNS, cfg, cfg.seeds[0], device, "transformer", out_dir=out_dir
    )
    print("[dev] Transformer best epoch:", dl_epoch["best_epoch"], dl_epoch["best_2023_brier"])

    print("[dev] select MLP epoch: train 2019~2022 -> valid 2023")
    mlp_epoch = select_epoch_on_2023(
        dev_df, schema, QUERY_COLUMNS, cfg, cfg.seeds[0], device, "mlp", out_dir=out_dir
    )
    print("[dev] MLP best epoch:", mlp_epoch["best_epoch"], mlp_epoch["best_2023_brier"])

    manifest = make_dev_manifest(df, prep_manifest, cfg, arms, cb_dev, schema, dl_epoch, mlp_epoch)
    save_json(manifest, out_dir / "dev_manifest.json")

    pd.DataFrame(
        [
            {
                "arm": arm,
                "dev_brier_3seed": score,
                "locked_candidate": arm in cb_dev["locked_candidates"],
            }
            for arm, score in cb_dev["confirmed_rank"]
        ]
    ).to_csv(out_dir / "dev_catboost_rank.csv", index=False)
    pd.DataFrame(dl_epoch["history"]).to_csv(out_dir / "dev_transformer_epoch_curve.csv", index=False)
    pd.DataFrame(mlp_epoch["history"]).to_csv(out_dir / "dev_mlp_epoch_curve.csv", index=False)

    return manifest


def _verify_manifest_signature(manifest: Mapping[str, Any], key: str) -> None:
    expected = manifest[key]
    clone = dict(manifest)
    clone.pop(key, None)
    actual = sha256_json(clone)
    if actual != expected:
        raise RuntimeError(f"Manifest signature mismatch: {key}")


def run_locked(project_dir: Path, out_dir: Path, device: torch.device) -> Dict[str, Any]:
    dev_path = out_dir / "dev_manifest.json"
    locked_path = out_dir / "locked_manifest.json"
    if locked_path.exists():
        raise RuntimeError(
            "Locked validation already exists. Refusing to rerun 2024 in this experiment lineage."
        )
    if not dev_path.exists():
        raise FileNotFoundError("Run --stage dev first.")

    dev = load_json(dev_path)
    _verify_manifest_signature(dev, "frozen_signature")
    if dev.get("code_sha256") != current_code_sha256():
        raise RuntimeError(
            "Code changed after dev was frozen. Use a new output directory and rerun dev."
        )
    cfg_dict = dict(dev["config"])
    cfg_dict["seeds"] = tuple(cfg_dict["seeds"])
    cfg = PipelineConfig(**cfg_dict)

    df, prep_manifest = load_prepared(project_dir, out_dir)
    validate_prepared(df, cfg)
    if prep_manifest["prepared_fingerprint"]["sha256"] != dev["prepared_fingerprint"]["sha256"]:
        raise RuntimeError("Prepared feature table differs from the frozen dev manifest.")

    train_df = df.loc[
        df["_dataset_split"].eq("train")
        & df["season"].between(cfg.train_start_year, cfg.locked_val_year)
    ].copy()

    arms = {k: list(v) for k, v in dev["catboost_arms"].items()}
    candidates = list(dev["locked_catboost_candidates"])
    print("[locked] pre-registered CatBoost candidates:", candidates)
    cb_results, chosen_arm, p_cb = catboost_locked_eval(train_df, arms, candidates, cfg)

    schema = StateSchema.from_dict(dev["state_schema"])
    dl_epochs = int(dev["transformer_epoch_selection"]["best_epoch"])
    mlp_epochs = int(dev["mlp_epoch_selection"]["best_epoch"])

    dl_result, p_dl = locked_dl_eval(
        train_df, schema, dev["query_columns"], cfg, dl_epochs, device, "transformer"
    )
    mlp_result, p_mlp = locked_dl_eval(
        train_df, schema, dev["query_columns"], cfg, mlp_epochs, device, "mlp"
    )

    val_df = train_df.loc[train_df["season"].eq(cfg.locked_val_year)].copy()
    y = val_df[TARGET].to_numpy(dtype=float)

    diag_8020 = ensemble_diagnostics(val_df, cfg, y, p_cb, p_dl, 0.80)
    sensitivity = None
    final_cb_weight = 1.0
    if diag_8020["delta_blend_minus_cb"] < 0:
        sensitivity = coarse_blend_sensitivity(y, p_cb, p_dl)
        best_key = min(sensitivity, key=sensitivity.get)
        # Robust parse of "cb_90_dl_10".
        final_cb_weight = int(best_key.split("_")[1]) / 100.0

    final_blend_pred = final_cb_weight * p_cb + (1.0 - final_cb_weight) * p_dl
    final_blend_brier = safe_brier(y, final_blend_pred)

    pred_frame = pd.DataFrame(
        {
            ID: val_df[ID].to_numpy(),
            "y": y,
            "p_catboost": p_cb,
            "p_transformer": p_dl,
            "p_mlp": p_mlp,
            "p_blend_80_20": 0.8*p_cb + 0.2*p_dl,
            "p_frozen_final_blend": final_blend_pred,
        }
    )
    pred_frame.to_csv(out_dir / "locked_2024_predictions.csv", index=False)

    locked = {
        "created_at": now_iso(),
        "prepared_fingerprint": prep_manifest["prepared_fingerprint"],
        "dev_frozen_signature": dev["frozen_signature"],
        "code_sha256": current_code_sha256(),
        "package_versions": package_versions(),
        "config": asdict(cfg),
        "chosen_catboost_arm": chosen_arm,
        "chosen_catboost_features": arms[chosen_arm],
        "catboost_locked_results": cb_results,
        "transformer_locked_result": dl_result,
        "mlp_locked_result": mlp_result,
        "mlp_vs_transformer": {
            "mlp_brier": safe_brier(y, p_mlp),
            "transformer_brier": safe_brier(y, p_dl),
            "transformer_minus_mlp": safe_brier(y, p_dl) - safe_brier(y, p_mlp),
        },
        "ensemble_80_20": diag_8020,
        "coarse_blend_sensitivity": sensitivity,
        "final_cb_weight": float(final_cb_weight),
        "final_locked_brier": float(final_blend_brier),
        "transformer_best_epoch": dl_epochs,
        "mlp_best_epoch": mlp_epochs,
        "state_schema": schema.to_dict(),
        "query_columns": list(dev["query_columns"]),
        "locked_protocol_note": (
            "One-time 2024 decision. Do not return to dev tuning after inspecting this manifest."
        ),
    }
    locked["locked_signature"] = sha256_json(locked)
    save_json(locked, locked_path)
    print(
        f"[locked] CatBoost arm={chosen_arm} | frozen CB weight={final_cb_weight:.2f} "
        f"| Brier={final_blend_brier:.10f}"
    )
    return locked


def run_final(project_dir: Path, out_dir: Path, device: torch.device) -> pd.DataFrame:
    dev_path = out_dir / "dev_manifest.json"
    locked_path = out_dir / "locked_manifest.json"
    if not dev_path.exists() or not locked_path.exists():
        raise FileNotFoundError("Run --stage dev and --stage locked first.")

    locked = load_json(locked_path)
    _verify_manifest_signature(locked, "locked_signature")
    if locked.get("code_sha256") != current_code_sha256():
        raise RuntimeError(
            "Code changed after locked validation. Use the frozen code to run final."
        )
    cfg_dict = dict(locked["config"])
    cfg_dict["seeds"] = tuple(cfg_dict["seeds"])
    cfg = PipelineConfig(**cfg_dict)

    df, prep_manifest = load_prepared(project_dir, out_dir)
    validate_prepared(df, cfg, require_test=True)
    if prep_manifest["prepared_fingerprint"]["sha256"] != locked["prepared_fingerprint"]["sha256"]:
        raise RuntimeError("Prepared table differs from locked manifest.")

    train_df = df.loc[
        df["_dataset_split"].eq("train")
        & df["season"].between(cfg.train_start_year, cfg.locked_val_year)
    ].copy()
    test_df = df.loc[df["_dataset_split"].eq("test")].copy()
    if test_df.empty:
        raise RuntimeError("No test rows.")

    seed = cfg.seeds[0]
    print(f"[final] CatBoost {locked['chosen_catboost_arm']} 2019~2024 -> test")
    p_cb = fit_predict_catboost(
        train_df, test_df, locked["chosen_catboost_features"], cfg, seed
    )

    schema = StateSchema.from_dict(locked["state_schema"])
    print(f"[final] Transformer epoch={locked['transformer_best_epoch']}")
    p_dl, _ = train_transformer_epochs(
        train_df,
        test_df,
        schema,
        locked["query_columns"],
        cfg,
        seed,
        int(locked["transformer_best_epoch"]),
        device,
        track_each_epoch=False,
    )

    cb_weight = float(locked["final_cb_weight"])
    p_final = cb_weight * p_cb + (1.0 - cb_weight) * p_dl

    detailed = pd.DataFrame(
        {
            ID: test_df[ID].to_numpy(),
            "p_catboost": p_cb,
            "p_transformer": p_dl,
            "cb_weight": cb_weight,
            TARGET: p_final,
        }
    )
    detailed.to_csv(out_dir / "final_predictions_2025.csv", index=False)

    sample_path = resolve_data_paths(project_dir)["sample_submission"]
    if sample_path.exists():
        submission = pd.read_csv(sample_path)
        if ID not in submission.columns:
            raise RuntimeError(f"sample_submission missing {ID}")
        pred_map = pd.Series(p_final, index=test_df[ID].to_numpy())
        submission[TARGET] = submission[ID].map(pred_map)
        if submission[TARGET].isna().any():
            raise RuntimeError("Could not map all test predictions into sample_submission.")
        submission.to_csv(out_dir / "submission_model_v4.csv", index=False)
        print("[final] saved:", out_dir / "submission_model_v4.csv")

    return detailed


# -----------------------------------------------------------------------------
# Lightweight structural smoke test
# -----------------------------------------------------------------------------

def smoke_test_helpers() -> None:
    # This does not pretend to reproduce the real dataset; it validates naming,
    # state-schema assembly, model forward shape, and temporal arm construction.
    rng = np.random.default_rng(7)
    rows = 12
    df = pd.DataFrame({c: np.zeros(rows, dtype=np.float32) for c in FEATURES if c not in CAT_COLS})
    for c in CAT_COLS:
        df[c] = "__NA__"
    df[ID] = np.arange(rows)
    df[TARGET] = rng.integers(0, 2, size=rows)
    df["pitcher_id"] = np.arange(rows)
    df["_dataset_split"] = "train"
    df["game_type"] = "R"
    df["season"] = np.repeat([2022, 2023], rows//2)

    for lag in (1,2,3):
        prefix = f"temporal_prev{lag}"
        for group in TEMPORAL_CONTEXT_GROUPS:
            for flag in FLAG_COLS:
                df[f"{prefix}_{group}_{flag}_rate"] = rng.random(rows).astype("float32")
            df[flat_n_column(prefix, group, "flag_success")] = rng.integers(0,100,size=rows)
            df[f"{prefix}_{group}_n"] = rng.integers(0,100,size=rows)
        for flag in FLAG_COLS:
            df[f"{prefix}_league_{flag}_rate"] = np.float32(0.5)
            df[f"{prefix}_pitcher_{flag}_rate_rel"] = (
                df[f"{prefix}_pitcher_{flag}_rate"] - 0.5
            ).astype("float32")
        df[f"{prefix}_pitcher_available"] = (
            df[flat_n_column(prefix, "pitcher", "flag_success")] > 0
        ).astype("int8")

    for flag in FLAG_COLS:
        df[f"temporal_trend12_pitcher_{flag}_rel"] = (
            df[f"temporal_prev1_pitcher_{flag}_rate_rel"]
            - df[f"temporal_prev2_pitcher_{flag}_rate_rel"]
        )
        df[f"temporal_trend23_pitcher_{flag}_rel"] = (
            df[f"temporal_prev2_pitcher_{flag}_rate_rel"]
            - df[f"temporal_prev3_pitcher_{flag}_rate_rel"]
        )

    # Career state source columns are Base263 Flat columns and already exist in df.
    schema = build_notebook_state_schema(df)
    cfg = PipelineConfig(dl_max_epochs=1, d_model=16, n_heads=4, n_layers=1, ffn_dim=32)
    values, availability, context_ids, time_ids = build_state_arrays(df, schema, cfg)
    assert values.shape[:2] == (rows, 24)
    assert availability.shape == (rows, 24)
    arms = build_catboost_arms(df)
    assert set(["T0_BASE","T4_REL_LEVEL_PLUS_TREND","T6_LEVEL_ONLY_CONTROL"]).issubset(arms)
    print("[smoke] helper/state schema OK", values.shape)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["prepare", "dev", "locked", "final", "smoke"], required=True)
    p.add_argument("--project-dir", type=Path, default=Path("/Users/chunyoomin/lgaimers"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    p.add_argument("--force-prepare", action="store_true")

    p.add_argument("--temporal-history-scope", choices=["all", "R"], default="all")
    p.add_argument("--sample-weight-mode", choices=["none", "guide_old_f"], default="none")
    p.add_argument("--old-f-weight", type=float, default=0.25)

    p.add_argument("--dev-2023-weight", type=float, default=2.0)
    p.add_argument("--catboost-locked-candidates", type=int, default=3)
    p.add_argument("--seeds", default="41,42,43")

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--token-dropout", type=float, default=0.10)
    p.add_argument("--dl-batch-size", type=int, default=512)
    p.add_argument("--dl-lr", type=float, default=1e-3)
    p.add_argument("--dl-weight-decay", type=float, default=1e-4)
    p.add_argument("--dl-max-epochs", type=int, default=60)
    p.add_argument("--dl-min-epochs", type=int, default=5)
    p.add_argument("--dl-patience", type=int, default=5)
    p.add_argument("--dl-min-delta", type=float, default=1e-6)
    p.add_argument("--dl-eval-every", type=int, default=1)
    p.add_argument("--dl-log-every", type=int, default=1)
    p.add_argument("--no-dl-resume", action="store_true", help="Disable dev DL checkpoint resume")
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    if not seeds:
        raise ValueError("At least one seed is required.")
    return PipelineConfig(
        dev_2023_weight=args.dev_2023_weight,
        catboost_locked_candidates=args.catboost_locked_candidates,
        seeds=seeds,
        temporal_history_scope=args.temporal_history_scope,
        sample_weight_mode=args.sample_weight_mode,
        old_f_weight=args.old_f_weight,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        token_dropout=args.token_dropout,
        dl_batch_size=args.dl_batch_size,
        dl_lr=args.dl_lr,
        dl_weight_decay=args.dl_weight_decay,
        dl_max_epochs=args.dl_max_epochs,
        dl_min_epochs=args.dl_min_epochs,
        dl_patience=args.dl_patience,
        dl_min_delta=args.dl_min_delta,
        dl_eval_every=args.dl_eval_every,
        dl_log_every=args.dl_log_every,
        dl_resume=not args.no_dl_resume,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_dir / "output" / "model_v4"
    )
    device = resolve_device(args.device)
    print("project_dir:", project_dir)
    print("output_dir :", out_dir)
    print("device     :", device)
    if device.type == "mps":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        print("[device] MPS enabled; ps/top CPU% can still be high because DataLoader and host->MPS copies run on CPU.")

    if args.stage == "smoke":
        smoke_test_helpers()
        return

    if args.stage == "prepare":
        cfg = config_from_args(args)
        prepare_and_cache(project_dir, out_dir, cfg, force=args.force_prepare)
        return

    if args.stage == "dev":
        cfg = config_from_args(args)
        run_dev(project_dir, out_dir, cfg, device, force_prepare=args.force_prepare)
        return

    if args.stage == "locked":
        run_locked(project_dir, out_dir, device)
        return

    if args.stage == "final":
        run_final(project_dir, out_dir, device)
        return

    raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
