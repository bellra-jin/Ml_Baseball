# src/utils/config.py
#
# 프로젝트 전체에서 공유하는 상수 및 설정값을 정의한다.
# 시즌 범위, 파일명, 피처 목록 등을 한 곳에서 관리해
# 다른 모듈에서는 이 파일만 import해서 사용한다.

# ─────────────────────────────────────────────
# KBO 리그 기본 설정
# ─────────────────────────────────────────────

# KBO 정규시즌 총 경기 수 (팀당)
TOTAL_GAMES = 144

# 타자 최소 타석 수 — 극소 출전 선수를 통계에서 제외하기 위한 기준
MIN_HITTER_PA = 30

# ─────────────────────────────────────────────
# 시즌 범위 설정
# ─────────────────────────────────────────────

# 모델 학습에 사용할 시즌 목록
# 2017년부터 시작하는 이유: 그 이전은 팀 수·규정이 달라 분포가 다름
TRAIN_SEASONS = list(range(2017, 2026))

# raw CSV → processed CSV 변환을 실행할 시즌 범위
# 2016 포함: 2017 학습 데이터의 prev_ 변수(전년도 기록)를 만들기 위해 필요
# 2026 포함: 현재 진행 중인 시즌의 예측 데이터를 만들기 위해 필요
PREPROCESS_SEASONS = list(range(2016, 2027))

# 포스트시즌 진출 확률을 예측할 대상 시즌
PREDICT_SEASON = 2026

# ─────────────────────────────────────────────
# 원본 CSV 파일명 매핑
# ─────────────────────────────────────────────

# data/raw/{year}/ 아래에 위치하는 파일명
# 키 이름으로 코드 곳곳에서 참조하므로 파일명을 바꿀 때 이 곳만 수정하면 된다
RAW_FILES = {
    "team_daily_rank": "team_daily_rank.csv",      # 일자별 팀 순위 (경기 결과 포함)
    "team_final_rank": "team_final_rank.csv",      # 시즌 최종 순위 (라벨 생성용)

    "team_hitter_basic": "team_hitter_basic.csv",  # 팀 타자 기본 기록
    "team_pitcher_basic": "team_pitcher_basic.csv",# 팀 투수 기본 기록
    "team_defense_basic": "team_defense_basic.csv",# 팀 수비 기본 기록
    "team_runner_basic": "team_runner_basic.csv",  # 팀 주루 기본 기록

    "player_hitter_basic": "player_hitter_basic.csv",    # 선수별 타자 기본 기록
    "player_hitter_detail": "player_hitter_detail.csv",  # 선수별 타자 세부 기록 (OPS 등)
    "player_pitcher_basic": "player_pitcher_basic.csv",  # 선수별 투수 기본 기록
    "player_pitcher_detail": "player_pitcher_detail.csv",# 선수별 투수 세부 기록
    "player_defense_basic": "player_defense_basic.csv",  # 선수별 수비 기록
    "player_runner_basic": "player_runner_basic.csv",    # 선수별 주루 기록
}

# ─────────────────────────────────────────────
# 다년도 집계 피처 설정
# ─────────────────────────────────────────────

# avg3yr_ / dyn_ 계산에 사용하는 핵심 지표 키 목록 (9개)
# make_features.py에서 avg3yr_{key} 컬럼을 생성하고,
# build_train/predict_dataset.py에서 dyn_{key} 컬럼을 생성할 때 이 목록을 참조한다
MULTI_YEAR_KEYS = [
    "pythagorean_win_rate",    # 피타고라스 승률 — 득실 기반 기대 승률 (운 요소 제거)
    "run_differential",        # 득실차 — 공격력 + 수비력 종합 지표
    "team_era",                # 팀 ERA — 투수진 전반 평균 자책점
    "k_bb_ratio",              # 탈삼진/볼넷 비율 — 투수 제구력
    "top5_hitter_ops_avg",     # 주전 타자 상위 5인 OPS 평균 — 타선 핵심 전력
    "ace_era",                 # 에이스 ERA — 로테이션 1선발 품질
    "iso",                     # ISO (장타율 - 타율) — 순수 장타력
    "ops_concentration",       # 타선 집중도 — 특정 선수 의존도 (낮을수록 균형)
    "bb_rate",                 # 볼넷 비율 — 선구안
]

# ─────────────────────────────────────────────
# 모델 학습 피처 목록 (36개)
# ─────────────────────────────────────────────
#
# 크게 세 그룹으로 구성된다:
#
#   1. 현재 시즌 성적 (18개)
#      — 순위·승률·최근 N경기 승률 등 날짜별로 업데이트되는 실시간 지표
#
#   2. 전년도(prev_) 핵심 지표 (9개)
#      — 직전 시즌의 팀 전력을 나타내는 지표
#      — 시즌 초반 예측에서 중요한 역할을 한다
#
#   3. 3년 평균 역가중(dyn_) 지표 (9개)
#      — dyn_{k} = (1 - games_played_ratio) × avg3yr_{k}
#      — 시즌 초반(ratio≈0): 3년 평균 전력이 그대로 반영 → 과거 강팀 식별
#      — 시즌 후반(ratio≈1): 0에 수렴 → 현재 시즌 성적이 예측을 주도
#      — 과거 우승팀(KIA·SSG 등)이 부진해도 말기까지 과대평가되는 문제를 해소한다
#
FEATURE_COLS = [
    # ── 현재 시즌 순위·성적 ──────────────────────────
    "rank",                 # 현재 순위
    "games",                # 누적 경기 수
    "win_rate",             # 누적 승률
    "games_behind",         # 1위와의 게임차
    "games_behind_5th",     # 5위와의 게임차 (음수=5위권 내)
    "remaining_games",      # 남은 경기 수
    "recent10_win_rate",    # 최근 10경기 승률
    "recent20_win_rate",    # 최근 20경기 승률
    "recent30_win_rate",    # 최근 30경기 승률
    "home_win_rate",        # 홈 승률
    "away_win_rate",        # 원정 승률
    "home_away_win_diff",   # 홈-원정 승률 차 (양수=홈 강팀)
    "streak_type",          # 연속 흐름 유형 (연승/연패 등)
    "streak_count",         # 연속 흐름 길이
    "games_played_ratio",   # 시즌 진행도 (0=개막, 1=종료)
    "win_rate_delta_30d",   # 30일 전 대비 승률 변화 (양수=상승 중)
    "rank_delta_30d",       # 30일 전 대비 순위 변화 (양수=상승 중)
    "wins_to_5th",          # 5위 추월에 필요한 잔여 승수

    # ── 전년도(t-1) 핵심 지표 ────────────────────────
    "prev_pythagorean_win_rate",   # 종합 전력 (운 제거한 기대 승률)
    "prev_run_differential",       # 득실차 (공격+수비 종합)
    "prev_team_era",               # 팀 ERA (투수진 전체 품질)
    "prev_k_bb_ratio",             # 탈삼진/볼넷 비율 (제구력)
    "prev_top5_hitter_ops_avg",    # 주전 타자 OPS 평균 (타격력)
    "prev_ace_era",                # 에이스 ERA (1선발 품질)
    "prev_iso",                    # ISO (순수 장타력)
    "prev_ops_concentration",      # 타선 균형도 (낮을수록 균형)
    "prev_bb_rate",                # 볼넷 비율 (선구안)

    # ── 3년 평균 역가중(dyn_) 지표 ───────────────────
    "dyn_pythagorean_win_rate",    # (1 - ratio) × avg3yr 피타고라스 승률
    "dyn_run_differential",        # (1 - ratio) × avg3yr 득실차
    "dyn_team_era",                # (1 - ratio) × avg3yr 팀 ERA
    "dyn_k_bb_ratio",              # (1 - ratio) × avg3yr 탈삼진/볼넷 비율
    "dyn_top5_hitter_ops_avg",     # (1 - ratio) × avg3yr 주전 타자 OPS
    "dyn_ace_era",                 # (1 - ratio) × avg3yr 에이스 ERA
    "dyn_iso",                     # (1 - ratio) × avg3yr ISO
    "dyn_ops_concentration",       # (1 - ratio) × avg3yr 타선 균형도
    "dyn_bb_rate",                 # (1 - ratio) × avg3yr 볼넷 비율
]
