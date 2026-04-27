# src/preprocessing/build_preprocessed.py

import pandas as pd

from src.preprocessing.clean_team_rank import (
    clean_team_daily_rank,
    clean_team_final_rank,
)

from src.preprocessing.clean_team_stats import (
    clean_team_hitter,
    clean_team_pitcher,
    clean_team_defense,
    clean_team_runner,
    merge_team_stats,
)

from src.preprocessing.clean_player_stats import (
    clean_player_hitter_basic,
    clean_player_hitter_detail,
    clean_player_pitcher_basic,
    make_hitter_summary,
    make_pitcher_summary,
)

from src.preprocessing.make_features import make_prev_features, make_daily_features

from src.utils.paths import RAW_DIR, PROCESSED_DIR, make_dirs
from src.utils.config import RAW_FILES


# =========================================================
# 03. 팀 순위 일자별 변수 생성
# =========================================================

def build_daily_rank_features(year):
    """
    team_daily_rank.csv에서 팀 순위 일자별 변수를 생성한다.

    생성 변수:
    - recent10_wins
    - recent10_draws
    - recent10_losses
    - recent10_win_rate
    - home_wins
    - home_draws
    - home_losses
    - home_win_rate
    - away_wins
    - away_draws
    - away_losses
    - away_win_rate
    - streak_type
    - streak_count
    - games_played_ratio
    - month
    """
    raw_year_dir = RAW_DIR / str(year)

    daily_rank = clean_team_daily_rank(
        raw_year_dir / RAW_FILES["team_daily_rank"],
        year,
    )

    daily_rank = make_daily_features(daily_rank)

    return daily_rank


# =========================================================
# 04. 팀 최종 순위 라벨 생성
# =========================================================

def build_final_rank_label(year):
    """
    team_final_rank.csv에서 final_rank와 postseason을 생성한다.

    postseason:
    - final_rank <= 5 이면 1
    - final_rank >= 6 이면 0
    """
    raw_year_dir = RAW_DIR / str(year)

    final_rank = clean_team_final_rank(
        raw_year_dir / RAW_FILES["team_final_rank"],
        year,
    )

    return final_rank


# =========================================================
# 05. 팀 시즌 기록 변수 생성
# =========================================================

def build_team_stats(year):
    """
    팀 타자/투수/수비/주루 기록을 합쳐 팀 시즌 기록 변수를 생성한다.

    생성 변수 예:
    - team_avg
    - team_runs
    - team_hits
    - team_hr
    - team_rbi
    - team_era
    - team_whip
    - team_runs_allowed
    - team_er
    - team_hr_allowed
    - team_bb_allowed
    - team_so_pitcher
    - team_ip
    - team_errors
    - team_fpct
    - team_dp
    - team_sb
    - team_cs
    - team_sb_rate
    """
    raw_year_dir = RAW_DIR / str(year)

    team_hitter = clean_team_hitter(
        raw_year_dir / RAW_FILES["team_hitter_basic"],
        year,
    )

    team_pitcher = clean_team_pitcher(
        raw_year_dir / RAW_FILES["team_pitcher_basic"],
        year,
    )

    team_defense = clean_team_defense(
        raw_year_dir / RAW_FILES["team_defense_basic"],
        year,
    )

    team_runner = clean_team_runner(
        raw_year_dir / RAW_FILES["team_runner_basic"],
        year,
    )

    team_stats = merge_team_stats(
        team_hitter,
        team_pitcher,
        team_defense,
        team_runner,
    )

    return team_stats


# =========================================================
# 06. 선수 기록 요약 변수 생성
# =========================================================

def build_player_summary(year):
    """
    선수 기록을 팀 단위 top5 요약 변수로 변환한다.

    생성 변수:
    - top5_hitter_avg
    - top5_hitter_hr_sum
    - top5_hitter_rbi_sum
    - top5_hitter_ops_avg
    - top5_hitter_obp_avg
    - top5_hitter_slg_avg
    - top5_pitcher_era_avg
    - top5_pitcher_whip_avg
    - top5_pitcher_ip_sum
    - top5_pitcher_so_sum
    """
    raw_year_dir = RAW_DIR / str(year)

    hitter_basic = clean_player_hitter_basic(
        raw_year_dir / RAW_FILES["player_hitter_basic"],
        year,
    )

    hitter_detail = clean_player_hitter_detail(
        raw_year_dir / RAW_FILES["player_hitter_detail"],
        year,
    )

    hitter_summary = make_hitter_summary(
        hitter_basic,
        hitter_detail,
    )

    pitcher_basic = clean_player_pitcher_basic(
        raw_year_dir / RAW_FILES["player_pitcher_basic"],
        year,
    )

    pitcher_summary = make_pitcher_summary(pitcher_basic)

    return hitter_summary, pitcher_summary


# =========================================================
# 07. 시즌 전체 변수 생성
# =========================================================

def build_season_features(year):
    """
    팀 시즌 기록 변수와 선수 요약 변수를 합친다.

    결과:
    - season_features_2022.csv
    - season_features_2023.csv
    - season_features_2024.csv
    - season_features_2025.csv
    - season_features_2026.csv
    """
    team_stats = build_team_stats(year)
    hitter_summary, pitcher_summary = build_player_summary(year)

    season_features = (
        team_stats
        .merge(hitter_summary, on=["season", "team"], how="left")
        .merge(pitcher_summary, on=["season", "team"], how="left")
    )

    return team_stats, hitter_summary, pitcher_summary, season_features


# =========================================================
# 08. 전년도 prev_ 변수 생성
# =========================================================

def build_prev_features(year):
    """
    해당 연도 시즌 변수를 다음 시즌에 붙일 prev_ 변수로 변환한다.

    예:
    - 2022 season_features → prev_features_for_2023
    - team_avg → prev_team_avg
    - team_era → prev_team_era
    - top5_hitter_ops_avg → prev_top5_hitter_ops_avg
    """
    _, _, _, season_features = build_season_features(year)

    prev_features = make_prev_features(season_features)

    return prev_features


# =========================================================
# 09. 한 연도 전처리 CSV 저장
# =========================================================

def save_preprocessed_year(year):
    """
    특정 연도의 raw CSV를 읽고,
    변수 생성이 끝난 processed CSV를 저장한다.
    """
    print(f"\n===== {year} 전처리 시작 =====")

    raw_year_dir = RAW_DIR / str(year)
    processed_year_dir = PROCESSED_DIR / str(year)

    if not raw_year_dir.exists():
        print(f"[SKIP] {raw_year_dir} 폴더가 없습니다.")
        return

    processed_year_dir.mkdir(parents=True, exist_ok=True)

    # 1. 팀 순위 일자별 변수 저장
    daily_rank = build_daily_rank_features(year)

    daily_rank.to_csv(
        processed_year_dir / "team_daily_rank_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("team_daily_rank_clean 저장:", daily_rank.shape)

    # 2. 최종 순위 라벨 저장 (2026은 시즌 진행 중이므로 라벨 없음)
    if year < 2026:
        final_rank = build_final_rank_label(year)

        final_rank.to_csv(
            processed_year_dir / "team_final_rank_clean.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("team_final_rank_clean 저장:", final_rank.shape)

    # 3. 팀 기록 + 선수 요약 변수 저장
    team_stats, hitter_summary, pitcher_summary, season_features = build_season_features(year)

    team_stats.to_csv(
        processed_year_dir / "team_stats_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hitter_summary.to_csv(
        processed_year_dir / "hitter_summary_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pitcher_summary.to_csv(
        processed_year_dir / "pitcher_summary_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    season_features.to_csv(
        processed_year_dir / f"season_features_{year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("team_stats_clean 저장:", team_stats.shape)
    print("hitter_summary_clean 저장:", hitter_summary.shape)
    print("pitcher_summary_clean 저장:", pitcher_summary.shape)
    print(f"season_features_{year} 저장:", season_features.shape)

    # 4. 다음 시즌에 붙일 prev_ 변수 저장
    if year < 2026:
        prev_features = make_prev_features(season_features)

        prev_features.to_csv(
            processed_year_dir / f"prev_features_for_{year + 1}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(f"prev_features_for_{year + 1} 저장:", prev_features.shape)

    print(f"===== {year} 전처리 완료 =====")


# =========================================================
# 10. 전체 연도 전처리 실행
# =========================================================

def main():
    """
    2022~2026 데이터를 전처리하고
    data/processed/연도/ 폴더에 저장한다.
    """
    make_dirs()

    years = [2022, 2023, 2024, 2025, 2026]

    for year in years:
        save_preprocessed_year(year)


if __name__ == "__main__":
    main()