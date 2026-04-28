# -*- coding: utf-8 -*-
"""Patch dy_final KBO notebook so it loads the existing data_ clean CSV files."""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path


NOTEBOOK_DIR = Path(r"C:\Users\Admin\Documents\GitHub\Ml_Baseball\dy_final")
NOTEBOOK_PATH = next(NOTEBOOK_DIR.glob("*.ipynb"))


CELL4_SOURCE = r'''
# ── 경로 설정 ──────────────────────────────────────────────────
from pathlib import Path
import pandas as pd
import numpy as np

NOTEBOOK_DIR = Path(r"C:\Users\Admin\Documents\GitHub\Ml_Baseball\dy_final")
CLEAN_DIR    = NOTEBOOK_DIR / "data_"      # 이미 만들어진 클린/전처리 CSV
RAW_ROOT     = NOTEBOOK_DIR / "data"       # 원본이 있을 때만 보조 사용
PROCESSED_DIR = CLEAN_DIR

YEARS         = [2022, 2023, 2024, 2025, 2026]
COMPLETE_YRS  = [2022, 2023, 2024, 2025]
PLAYOFF_CUT   = 5

if 'read_csv_safe' not in globals():
    def read_csv_safe(path):
        for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        raise ValueError(f"읽기 실패: {path}")

def _read_clean(filename):
    path = CLEAN_DIR / filename
    if not path.exists():
        return None
    df = read_csv_safe(str(path))
    if '날짜' in df.columns:
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    return df

def _cols(df, cols):
    return df[[c for c in cols if c in df.columns]].copy()

def _team_from_prefixed(team_df, prefix):
    base_cols = [c for c in ['연도', '팀명'] if c in team_df.columns]
    pref_cols = [c for c in team_df.columns if c.startswith(prefix)]
    out = team_df[base_cols + pref_cols].copy()
    out = out.rename(columns={c: c.replace(prefix, '', 1) for c in pref_cols})
    if '경기' in team_df.columns and 'G' not in out.columns:
        out['G'] = team_df['경기']
    return out

def _monthly_summary_to_daily(rank_vol):
    rows = []
    month_cols = [c for c in rank_vol.columns if c.endswith('_평균순위') and '월' in c]
    for col in month_cols:
        try:
            month = int(col.split('월')[0])
        except Exception:
            continue
        sub = rank_vol[['연도', '팀명', col]].dropna().copy()
        sub = sub.rename(columns={col: '순위'})
        sub['월'] = month
        sub['날짜'] = pd.to_datetime(
            sub['연도'].astype(int).astype(str) + f'-{month:02d}-15',
            errors='coerce'
        )
        rows.append(sub[['연도', '날짜', '월', '팀명', '순위']])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

data = {}

if not CLEAN_DIR.exists():
    raise FileNotFoundError(f"클린 데이터 폴더가 없습니다: {CLEAN_DIR}")

print(f"📂 클린 CSV 로드: {CLEAN_DIR.resolve()}")

# ── 선수 통합 파일 로드 및 기본/세부 alias 생성 ───────────────
bat_all = _read_clean('타자_기본기록_통합.csv')
if bat_all is not None:
    data['타자_기본기록'] = _cols(bat_all, [
        '연도','순위','선수명','팀명','AVG','G','PA','AB','R','H','2B','3B','HR',
        'TB','RBI','SAC','SF','BB','IBB','HBP','SO','GDP','SLG','OBP','OPS',
        'MH','RISP','PH-BA','ISO','BB_pct','K_pct','BABIP'
    ])
    data['타자_세부기록'] = _cols(bat_all, [
        '연도','순위','선수명','팀명','AVG','XBH','GO','AO','GO/AO','GW RBI',
        'BB/K','P/PA','ISOP','XR','GPA'
    ])

pit_all = _read_clean('투수_기본기록_통합.csv')
if pit_all is not None:
    data['투수_기본기록'] = _cols(pit_all, [
        '연도','순위','선수명','팀명','ERA','G','W','L','SV','HLD','WPCT','IP',
        'H','HR','BB','HBP','SO','R','ER','WHIP','K9','BB9','FIP'
    ])
    data['투수_세부기록'] = _cols(pit_all, [
        '연도','순위','선수명','팀명','ERA','GS','Wgs','Wgr','GF','SVO',
        'TS','GDP','GO','AO','GO/AO'
    ])

for key, filename in [
    ('수비_기본기록', '수비_기본기록_통합.csv'),
    ('주루_기본기록', '주루_기본기록_통합.csv'),
    ('타자_마스터', '타자_마스터.csv'),
    ('투수_마스터', '투수_마스터.csv'),
    ('팀_종합', '팀_종합.csv'),
    ('팀_순위변동성', '팀_순위변동성.csv'),
    ('팀_시즌최종순위', '팀_시즌최종순위.csv'),
    ('리그_타격환경', '리그_타격환경.csv'),
    ('리그_투구환경', '리그_투구환경.csv'),
    ('피처_중요도', '피처_중요도.csv'),
    ('선수이동현황', '선수이동현황_통합.csv'),
]:
    df = _read_clean(filename)
    if df is not None:
        data[key] = df

# ── 팀 기본 기록 alias 생성: 후속 셀의 기존 JOIN 코드와 호환 ──
team_clean = data.get('팀_종합')
if team_clean is not None:
    data['팀_순위'] = data.get('팀_시즌최종순위', _cols(team_clean, [
        '연도','순위','팀명','경기','승','패','무','승률','게임차',
        '홈승','홈무','홈패','원정승','원정무','원정패','홈승률','원정승률','가을야구'
    ]))
    data['팀_타자_기본기록'] = _team_from_prefixed(team_clean, 'bat_')
    data['팀_투수_기본기록'] = _team_from_prefixed(team_clean, 'pit_')
    data['팀_수비_기본기록'] = _team_from_prefixed(team_clean, 'def_')
    data['팀_주루_기본기록'] = _team_from_prefixed(team_clean, 'run_')

# ── 일자별 순위가 없으면 월별 평균순위로 4월 분석용 proxy 생성 ─
daily_file = _read_clean('팀_일자별순위_통합.csv')
if daily_file is not None:
    data['팀_일자별순위'] = daily_file
elif '팀_순위변동성' in data:
    data['팀_일자별순위'] = _monthly_summary_to_daily(data['팀_순위변동성'])

print(f"\n{'='*55}")
print(f"✅ data 딕셔너리: {len(data)}개 키")
for k, v in data.items():
    yrs = sorted(v['연도'].dropna().astype(int).unique().tolist()) if '연도' in v.columns else '-'
    print(f"  ✔ {k}: {v.shape}  연도: {yrs}")

required = ['타자_기본기록','투수_기본기록','팀_순위','팀_일자별순위',
            '팀_타자_기본기록','팀_투수_기본기록','타자_마스터','팀_종합']
missing = [k for k in required if k not in data or data[k].empty]
if missing:
    raise AssertionError(f"필수 데이터 누락: {missing}")
else:
    print("\n✅ 필수 키 모두 존재 → CELL 5 실행 가능")
print('='*55)
'''


CELL5_SOURCE = r'''
assert 'data' in dir() and len(data)>0, "CELL 4를 먼저 실행하세요!"

def _display_head(title, df, year=2026, n=3):
    print(f"{title}: {df.shape}")
    if '연도' in df.columns:
        display(df[df['연도']==year].head(n))
    else:
        display(df.head(n))

def _add_team_aliases(df):
    df = df.copy()
    alias_map = {
        'bat_AVG':'AVG_타자', 'bat_OBP':'OBP_타자', 'bat_SLG':'SLG_타자',
        'bat_OPS':'OPS_타자', 'bat_HR':'HR_타자', 'bat_R':'R_타자',
        'bat_RBI':'RBI_타자', 'bat_H':'H_타자', 'bat_BB':'BB_타자',
        'bat_SO':'SO_타자', 'pit_ERA':'ERA_투수', 'pit_WHIP':'WHIP_투수',
        'pit_K9':'K/9_투수', 'pit_BB9':'BB/9_투수', 'pit_FIP':'FIP_투수',
        'pit_R':'R_투수', 'pit_IP':'IP_투수', 'pit_SO':'SO_투수',
        'pit_BB':'BB_투수', 'pit_HR':'HR_투수', 'def_FPCT':'FPCT_수비',
        'def_E':'E_수비', 'def_DP':'DP_수비', 'run_SB':'SB_주루',
        'run_SB%':'SB%_주루', 'run_SBA':'SBA_주루', 'run_CS':'CS_주루'
    }
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    if {'bat_R','경기'}.issubset(df.columns) and '득점/G_타자' not in df.columns:
        df['득점/G_타자'] = (df['bat_R'] / df['경기'].replace(0, np.nan)).round(3)
    if {'bat_HR','경기'}.issubset(df.columns) and 'HR/G_타자' not in df.columns:
        df['HR/G_타자'] = (df['bat_HR'] / df['경기'].replace(0, np.nan)).round(3)
    if '가을야구' in df.columns:
        df.loc[df['연도']==2026, '가을야구'] = np.nan
    return df

def _league_bat_table(df):
    out = df.copy()
    ren = {
        '리그_AVG':'AVG', '리그_OBP':'OBP', '리그_SLG':'SLG', '리그_OPS':'OPS',
        '리그_HR평균':'HR', '리그_RBI평균':'RBI', '리그_BABIP':'BABIP',
        '리그_ISO':'ISO'
    }
    return out.rename(columns=ren)

def _league_pit_table(df):
    out = df.copy()
    ren = {
        '리그_ERA':'ERA', '리그_WHIP':'WHIP', '리그_K9':'K/9',
        '리그_BB9':'BB/9', '리그_FIP':'FIP', '리그_HR9':'HR/9'
    }
    return out.rename(columns=ren)

def _monthly_from_rank_vol(rank_vol):
    month_cols = [c for c in rank_vol.columns if c.endswith('_평균순위') and '월' in c]
    frames = []
    for col in month_cols:
        try:
            month = int(col.split('월')[0])
        except Exception:
            continue
        sub = rank_vol[['연도','팀명',col]].dropna().copy()
        sub['월'] = month
        sub = sub.rename(columns={col:'월평균순위'})
        frames.append(sub[['연도','팀명','월','월평균순위']])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ── 이미 생성된 클린/마스터 데이터 우선 사용 ─────────────────
batter_master = data.get('타자_마스터', data['타자_기본기록']).copy()
pitcher_master = data.get('투수_마스터', data['투수_기본기록']).copy()
team_master = _add_team_aliases(data.get('팀_종합', data['팀_순위']).copy())

_display_head("타자_마스터", batter_master)
_display_head("\n투수_마스터", pitcher_master)
_display_head("\n팀_종합_마스터", team_master)

# ── 일자별/월별 순위 테이블 ───────────────────────────────────
daily = data['팀_일자별순위'].copy()
if '날짜' in daily.columns:
    daily['날짜'] = pd.to_datetime(daily['날짜'], errors='coerce')
if '월' not in daily.columns and '날짜' in daily.columns:
    daily['월'] = daily['날짜'].dt.month

if '팀_순위변동성' in data:
    rank_vol = data['팀_순위변동성'].copy()
else:
    rank_vol = (daily.groupby(['연도','팀명'])['순위']
                .agg(['mean','std','min','max']).reset_index())
    rank_vol.columns = ['연도','팀명','평균순위','순위변동성','최고순위','최저순위']
    rank_vol['순위변동성'] = rank_vol['순위변동성'].round(2)

monthly_rank = _monthly_from_rank_vol(rank_vol)
if monthly_rank.empty and {'연도','팀명','월','순위'}.issubset(daily.columns):
    monthly_rank = (daily.groupby(['연도','팀명','월'])['순위']
                    .mean().round(2).reset_index()
                    .rename(columns={'순위':'월평균순위'}))

if '리그_타격환경' in data:
    league_bat = _league_bat_table(data['리그_타격환경'])
else:
    league_bat = batter_master.groupby('연도')[['AVG','OPS','HR','RBI','BABIP']].mean().round(3).reset_index()

if '리그_투구환경' in data:
    league_pit = _league_pit_table(data['리그_투구환경'])
else:
    league_pit = pitcher_master.groupby('연도')[['ERA','WHIP','K9','BB9','FIP']].mean().round(3).reset_index()
    league_pit = league_pit.rename(columns={'K9':'K/9', 'BB9':'BB/9'})

print("\n=== 리그 타격환경 ===")
display(league_bat)
print("=== 리그 투구환경 ===")
display(league_pit)
print("\n✅ 인사이트 테이블 생성 완료")
'''


def set_cell_source(cell: dict, source: str) -> None:
    cell["source"] = textwrap.dedent(source).strip().splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    backup_path = NOTEBOOK_PATH.with_suffix(".backup_before_data_patch.ipynb")
    if not backup_path.exists():
        shutil.copy2(NOTEBOOK_PATH, backup_path)

    replaced4 = replaced5 = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "RAW_ROOT      = Path('./data/raw')" in src and "data = {}" in src:
            set_cell_source(cell, CELL4_SOURCE)
            replaced4 = True
        elif "assert 'data' in dir() and len(data)>0" in src and "타자_마스터" in src:
            set_cell_source(cell, CELL5_SOURCE)
            replaced5 = True

    if not (replaced4 and replaced5):
        raise RuntimeError(f"패치 대상 셀을 찾지 못했습니다. CELL4={replaced4}, CELL5={replaced5}")

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"patched: {NOTEBOOK_PATH}")
    print(f"backup : {backup_path}")


if __name__ == "__main__":
    main()
