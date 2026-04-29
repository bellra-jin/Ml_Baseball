# -*- coding: utf-8 -*-
# KBO 가을야구 예측 전처리 v2 실행본

# %% 셀 1
from pathlib import Path
import sys
import importlib
import pandas as pd

# 노트북을 어디서 열어도 같은 프로젝트 폴더를 기준으로 실행되게 고정합니다.
PROJECT_DIR = Path(r"C:\Users\Admin\Documents\GitHub\Ml_Baseball\dy_final")
REPO_DIR = PROJECT_DIR.parent
DATA_DIR = REPO_DIR / "data"
OUT_DIR = PROJECT_DIR / "kbo_outputs"

sys.path.insert(0, str(PROJECT_DIR))

print("프로젝트 폴더:", PROJECT_DIR)
print("데이터 폴더:", DATA_DIR)
print("결과 저장 폴더:", OUT_DIR)

# %% 셀 2
for path in [DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "2026"]:
    print("\n==", path, "==")
    if not path.exists():
        print("폴더 없음")
        continue
    for item in sorted(path.iterdir())[:20]:
        print(item.name)

# %% 셀 3
import kbo_postseason_pipeline as pipe
importlib.reload(pipe)

artifacts = pipe.run_pipeline(DATA_DIR, OUT_DIR)
artifacts["summary"]

# %% 셀 4
model_df = artifacts["model_dataset"]
print("데이터 크기:", model_df.shape)
model_df.head()

# %% 셀 5
cols = [
    "season", "date", "team", "rank", "games", "wins", "losses", "win_rate",
    "postseason", "prev_final_rank", "prev_win_rate", "prev_era", "prev_whip"
]
model_df[[c for c in cols if c in model_df.columns]].head(10)

# %% 셀 6
pred = artifacts["predictions"]
pred[[
    "team", "rank", "wins", "losses", "win_rate", "games_behind",
    "model_probability_pct", "standing_probability_pct", "postseason_probability_pct",
    "prediction_label"
]]

# %% 셀 7
print("시즌 단위 검증 결과")
display(artifacts["validation"])

print("4월 순위 인사이트")
display(artifacts["april"])

# %% 셀 8
import run_kbo_pipeline_outputs as refresh
importlib.reload(refresh)

refresh.export_dashboard_csvs(DATA_DIR, OUT_DIR)

for p in sorted(PROJECT_DIR.glob("*.csv")):
    print(p.name, p.stat().st_size)

# %% 셀 9
import generate_kbo_visualizations as viz
importlib.reload(viz)

viz.main()

for p in sorted(PROJECT_DIR.glob("0*.png")):
    print(p.name, p.stat().st_size)

# %% 셀 10
import generate_kbo_master_dashboard as dash
importlib.reload(dash)

dash.main()

html_path = PROJECT_DIR / "kbo_2022_2026_master_dashboard.html"
print("HTML 위치:", html_path)
print("파일 생성 여부:", html_path.exists())
print("브라우저 주소:", "file:///" + str(html_path).replace("\\", "/"))

