# src/preprocessing/build_preprocessed.py
#
# raw CSV 데이터를 읽어 전처리된 processed CSV로 저장하는 실행 파일이다.
# 팀 순위, 팀 기록, 선수 기록, 전년도 변수, 다년도 집계 변수를 생성하고
# 연도별 data/processed/{year}/ 폴더에 저장한다.

import pandas as pd

from src.preprocessing.clean_team_rank import (
    clean_team_daily_rank,
    clean_team_final_rank,
)

from src.preprocessing.clean_team_stats import (
    clean_team_hitter,
    clean_team_hitter_from_player,
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

from src.preprocessing.make_features import make_prev_features, make_daily_features, make_multi_year_features

from src.utils.paths import RAW_DIR, PROCESSED_DIR, make_dirs
from src.utils.config import PREPROCESS_SEASONS, RAW_FILES


# =========================================================
# 01. 팀 순위 일자별 변수 생성
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

    # 해당 연도의 raw 데이터 폴더 경로
    raw_year_dir = RAW_DIR / str(year)

    # 일자별 팀 순위 원본 CSV를 읽고 기본 정제 수행
    daily_rank = clean_team_daily_rank(
        raw_year_dir / RAW_FILES["team_daily_rank"],
        year,
    )

    # 최근10경기, 홈/원정, 연승/연패, 시즌 진행률 등 파생 변수 생성
    daily_rank = make_daily_features(daily_rank)

    # 전처리된 일자별 팀 순위 데이터 반환
    return daily_rank


# =========================================================
# 02. 팀 최종 순위 라벨 생성
# =========================================================

def build_final_rank_label(year):
    """
    team_final_rank.csv에서 final_rank와 postseason을 생성한다.

    postseason:
    - final_rank <= 5 이면 1
    - final_rank >= 6 이면 0
    """

    # 해당 연도의 raw 데이터 폴더 경로
    raw_year_dir = RAW_DIR / str(year)

    # 최종 순위 CSV를 읽고 postseason 라벨 생성
    final_rank = clean_team_final_rank(
        raw_year_dir / RAW_FILES["team_final_rank"],
        year,
    )

    # 최종 순위 및 포스트시즌 진출 여부 데이터 반환
    return final_rank


# =========================================================
# 03. 팀 시즌 기록 변수 생성
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

    # 해당 연도의 raw 데이터 폴더 경로
    raw_year_dir = RAW_DIR / str(year)

    # 팀 타자 기록 파일 경로
    team_hitter_path = raw_year_dir / RAW_FILES["team_hitter_basic"]

    # 팀 타자 기록 파일이 있으면 팀 기록 기준으로 정제
    if team_hitter_path.exists():
        team_hitter = clean_team_hitter(team_hitter_path, year)

    # 팀 타자 기록 파일이 없으면 선수 타자 기록을 팀 단위로 집계
    else:
        team_hitter = clean_team_hitter_from_player(
            raw_year_dir / RAW_FILES["player_hitter_basic"],
            year,
        )

    # 팀 투수 기록 정제
    team_pitcher = clean_team_pitcher(
        raw_year_dir / RAW_FILES["team_pitcher_basic"],
        year,
    )

    # 팀 수비 기록 정제
    team_defense = clean_team_defense(
        raw_year_dir / RAW_FILES["team_defense_basic"],
        year,
    )

    # 팀 주루 기록 정제
    team_runner = clean_team_runner(
        raw_year_dir / RAW_FILES["team_runner_basic"],
        year,
    )

    # 타자/투수/수비/주루 기록을 팀 기준으로 병합
    team_stats = merge_team_stats(
        team_hitter,
        team_pitcher,
        team_defense,
        team_runner,
    )

    # 팀 시즌 기록 데이터 반환
    return team_stats


# =========================================================
# 04. 선수 기록 요약 변수 생성
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

    # 해당 연도의 raw 데이터 폴더 경로
    raw_year_dir = RAW_DIR / str(year)

    # 선수 타자 기본 기록 정제
    hitter_basic = clean_player_hitter_basic(
        raw_year_dir / RAW_FILES["player_hitter_basic"],
        year,
    )

    # 선수 타자 세부 기록 정제
    hitter_detail = clean_player_hitter_detail(
        raw_year_dir / RAW_FILES["player_hitter_detail"],
        year,
    )

    # 타자 기본 기록과 세부 기록을 이용해 팀별 상위 타자 요약 변수 생성
    hitter_summary = make_hitter_summary(
        hitter_basic,
        hitter_detail,
    )

    # 선수 투수 기본 기록 정제
    pitcher_basic = clean_player_pitcher_basic(
        raw_year_dir / RAW_FILES["player_pitcher_basic"],
        year,
    )

    # 투수 기록을 이용해 팀별 상위 투수 요약 변수 생성
    pitcher_summary = make_pitcher_summary(pitcher_basic)

    # 타자 요약 데이터와 투수 요약 데이터 반환
    return hitter_summary, pitcher_summary


# =========================================================
# 05. 시즌 전체 변수 생성
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

    # 팀 단위 시즌 기록 생성
    team_stats = build_team_stats(year)

    # 선수 기록을 팀 단위 요약 변수로 생성
    hitter_summary, pitcher_summary = build_player_summary(year)

    # 팀 시즌 기록 + 타자 요약 + 투수 요약을 하나의 시즌 피처로 병합
    season_features = (
        team_stats
        .merge(hitter_summary, on=["season", "team"], how="left")
        .merge(pitcher_summary, on=["season", "team"], how="left")
    )

    # 중간 산출물과 최종 시즌 피처를 함께 반환
    return team_stats, hitter_summary, pitcher_summary, season_features


# =========================================================
# 06. 다년도 집계 변수 생성
# =========================================================

def build_multi_year_features(year):
    """
    해당 연도 예측에 쓸 다년도 집계 변수를 생성한다.
    t-1 ~ t-3 시즌 season_features를 읽어 avg3yr_ / trend_ 컬럼을 반환한다.
    """

    # t-1, t-2, t-3 시즌 데이터를 담을 리스트
    seasons = []

    # 직전 1~3개 시즌 데이터를 순서대로 확인
    for offset in range(1, 4):
        y = year - offset

        # 이전 시즌의 season_features 파일 경로
        path = PROCESSED_DIR / str(y) / f"season_features_{y}.csv"

        # 파일이 있으면 읽어서 리스트에 추가
        if path.exists():
            seasons.append(pd.read_csv(path))

        # 파일이 없으면 None으로 표시
        else:
            seasons.append(None)

    # t-1 데이터가 없으면 다년도 변수 생성 불가
    if seasons[0] is None:
        print(f"[SKIP] {year} 다년도 변수: t-1({year - 1}) 데이터 없음")
        return None

    # 이전 시즌 데이터를 기반으로 avg3yr_ / trend_ 계열 변수 생성
    return make_multi_year_features(year, seasons)


# =========================================================
# 07. 한 연도 전처리 CSV 저장
# =========================================================

def save_preprocessed_year(year):
    """
    특정 연도의 raw CSV를 읽고,
    변수 생성이 끝난 processed CSV를 저장한다.
    """

    # 전처리 시작 로그 출력
    print(f"\n===== {year} 전처리 시작 =====")

    # 해당 연도의 raw / processed 폴더 경로
    raw_year_dir = RAW_DIR / str(year)
    processed_year_dir = PROCESSED_DIR / str(year)

    # raw 데이터 폴더가 없으면 해당 연도 전처리 건너뜀
    if not raw_year_dir.exists():
        print(f"[SKIP] {raw_year_dir} 폴더가 없습니다.")
        return

    # processed 연도별 폴더 생성
    processed_year_dir.mkdir(parents=True, exist_ok=True)

    # 1. 팀 순위 일자별 변수 저장
    daily_rank = build_daily_rank_features(year)

    # 일자별 팀 순위 전처리 결과 저장
    daily_rank.to_csv(
        processed_year_dir / "team_daily_rank_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 저장 결과 확인용 로그 출력
    print("team_daily_rank_clean 저장:", daily_rank.shape)

    # 2. 최종 순위 라벨 저장 (2026은 시즌 진행 중이므로 라벨 없음)
    if year < 2026:
        final_rank = build_final_rank_label(year)

        # 최종 순위 및 postseason 라벨 저장
        final_rank.to_csv(
            processed_year_dir / "team_final_rank_clean.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 저장 결과 확인용 로그 출력
        print("team_final_rank_clean 저장:", final_rank.shape)

    # 3. 팀 기록 + 선수 요약 변수 저장
    team_stats, hitter_summary, pitcher_summary, season_features = build_season_features(year)

    # 팀 시즌 기록 저장
    team_stats.to_csv(
        processed_year_dir / "team_stats_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 팀별 상위 타자 요약 변수 저장
    hitter_summary.to_csv(
        processed_year_dir / "hitter_summary_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 팀별 상위 투수 요약 변수 저장
    pitcher_summary.to_csv(
        processed_year_dir / "pitcher_summary_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 팀 기록 + 선수 요약을 합친 시즌 피처 저장
    season_features.to_csv(
        processed_year_dir / f"season_features_{year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 저장 결과 확인용 로그 출력
    print("team_stats_clean 저장:", team_stats.shape)
    print("hitter_summary_clean 저장:", hitter_summary.shape)
    print("pitcher_summary_clean 저장:", pitcher_summary.shape)
    print(f"season_features_{year} 저장:", season_features.shape)

    # 4. 다음 시즌 폴더에 prev_ 변수 저장 (사용되는 연도 폴더에 보관)
    if year < 2026:
        # 현재 시즌 기록을 다음 시즌에서 사용할 prev_ 변수로 변환
        prev_features = make_prev_features(season_features)

        # 다음 시즌 processed 폴더 경로
        next_year_dir = PROCESSED_DIR / str(year + 1)

        # 다음 시즌 폴더가 없으면 생성
        next_year_dir.mkdir(parents=True, exist_ok=True)

        # prev_ 변수를 다음 연도 폴더에 저장
        prev_features.to_csv(
            next_year_dir / f"prev_features_from_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 저장 결과 확인용 로그 출력
        print(f"prev_features_from_{year} 저장 (→ {year + 1}/ 폴더):", prev_features.shape)

    # 5. 다년도 집계 변수 저장 (avg2yr_ / avg3yr_ / avg5yr_ / trend_)
    multi_year = build_multi_year_features(year)

    # 다년도 집계 변수가 생성된 경우에만 저장
    if multi_year is not None:
        multi_year.to_csv(
            processed_year_dir / f"multi_year_features_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # 저장 결과 확인용 로그 출력
        print(f"multi_year_features_{year} 저장:", multi_year.shape)

    # 전처리 완료 로그 출력
    print(f"===== {year} 전처리 완료 =====")


# =========================================================
# 08. 전체 연도 전처리 실행
# =========================================================

def main():
    """
    2016~2026 데이터를 전처리하고
    data/processed/연도/ 폴더에 저장한다.
    """

    # 프로젝트에서 사용하는 기본 폴더 생성
    make_dirs()

    # 설정된 전처리 대상 시즌을 순서대로 처리
    for year in PREPROCESS_SEASONS:
        save_preprocessed_year(year)


# 이 파일을 직접 실행할 때만 전체 전처리 실행
if __name__ == "__main__":
    main()