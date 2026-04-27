# src/utils/config.py

# KBO 정규시즌 전체 경기 수
TOTAL_GAMES = 144

# 선수 기록 필터 기준 (출전이 극히 적은 선수 제외)
MIN_HITTER_PA = 30   # 타자 최소 타석 수

# 학습에 사용할 시즌
TRAIN_SEASONS = [2023, 2024, 2025]

# 예측할 시즌
PREDICT_SEASON = 2026

# 원본 CSV 파일명
RAW_FILES = {
    "team_daily_rank": "team_daily_rank.csv",
    "team_final_rank": "team_final_rank.csv",

    "team_hitter_basic": "team_hitter_basic.csv",
    "team_pitcher_basic": "team_pitcher_basic.csv",
    "team_defense_basic": "team_defense_basic.csv",
    "team_runner_basic": "team_runner_basic.csv",

    "player_hitter_basic": "player_hitter_basic.csv",
    "player_hitter_detail": "player_hitter_detail.csv",
    "player_pitcher_basic": "player_pitcher_basic.csv",
    "player_pitcher_detail": "player_pitcher_detail.csv",
    "player_defense_basic": "player_defense_basic.csv",
    "player_runner_basic": "player_runner_basic.csv",
}


# 1차 모델에서 사용할 주요 피처
FEATURE_COLS = [
    # 현재 시즌 순위 기반 변수
    "rank",
    "games",
    "win_rate",
    "games_behind",
    "games_behind_5th",
    "remaining_games",
    "recent10_win_rate",
    "recent20_win_rate",
    "recent30_win_rate",
    "home_win_rate",
    "away_win_rate",
    "home_away_win_diff",
    "streak_type",
    "streak_count",
    "games_played_ratio",

    # 추세 변수
    "win_rate_delta_30d",
    "rank_delta_30d",
    "wins_to_5th",

    # 전년도 종합 전력 지표
    "prev_run_differential",
    "prev_pythagorean_win_rate",
    "prev_k_bb_ratio",
    "prev_iso",
    "prev_bb_rate",

    # 전년도 팀 타격 변수
    "prev_team_avg",
    "prev_team_runs",
    "prev_team_hits",
    "prev_team_hr",
    "prev_team_rbi",

    # 전년도 팀 투수 변수
    "prev_team_era",
    "prev_team_whip",
    "prev_team_runs_allowed",
    "prev_team_er",
    "prev_team_hr_allowed",
    "prev_team_bb_allowed",
    "prev_team_so_pitcher",
    "prev_team_ip",

    # 전년도 팀 수비 변수
    "prev_team_errors",
    "prev_team_fpct",
    "prev_team_dp",

    # 전년도 팀 주루 변수
    "prev_team_sb",
    "prev_team_cs",
    "prev_team_sb_rate",

    # 전년도 선수 타자 요약 변수
    "prev_top5_hitter_avg",
    "prev_top5_hitter_hr_sum",
    "prev_top5_hitter_rbi_sum",
    "prev_top5_hitter_ops_avg",
    "prev_top5_hitter_obp_avg",
    "prev_top5_hitter_slg_avg",
    "prev_top3_hitter_ops_avg",
    "prev_ops_concentration",

    # 전년도 선수 투수 요약 변수
    "prev_top5_pitcher_era_avg",
    "prev_top5_pitcher_whip_avg",
    "prev_top5_pitcher_ip_sum",
    "prev_top5_pitcher_so_sum",
    "prev_ace_era",
    "prev_ace_era_gap",
]