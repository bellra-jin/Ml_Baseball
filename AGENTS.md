# AGENTS.md

## Setup

- **Package manager**: `uv` only. Run everything via `uv run`.
- **Python**: 3.12 (`.python-version`, `requires-python = ">=3.12"`). No venv management needed — uv handles it.
- **uv virtual environment**: Use `uv venv` to create, `uv run` to execute. All `python` commands should be run via `uv run python ...` or `uv run python -m ...`. Do not manually activate or rely on system Python.
- **No tests, no linter, no typechecker** are configured in this repo. Do not attempt to run them.

## Pipeline (strict order)

1. **Crawl** → `src/web_crawling/` (fetches raw CSVs from koreabaseball.com into `data/raw/{year}/`)
2. **Preprocess** → `uv run python -m src.preprocessing.build_preprocessed` (2016–2026, `data/raw/` → `data/processed/`)
3. **Build train dataset** → `uv run python -m src.dataset.build_train_dataset` (assembles `data/modeling/train_dataset.csv`)
4. **Train + evaluate** → 외부 실험 스크립트에서 수행 (read `train_dataset.csv`, fit models, write charts)
5. **Build predict dataset** → `uv run python -m src.dataset.build_predict_dataset` (assembles `data/modeling/predict_dataset_2026.csv`)

Steps 2-3-4-5 are sequential. Step 2 must complete before step 3. Step 3 must complete before step 4 and 5.

## Key config constants

All in `src/utils/config.py`:
- `TRAIN_SEASONS` = 2017–2025
- `PREPROCESS_SEASONS` = 2016–2026 (2016 needed for 2017's `prev_` features; 2026 for prediction)
- `PREDICT_SEASON` = 2026
- `TOTAL_GAMES` = 144 (KBO)
- `FEATURE_COLS` = 36 features (18 current + 9 `prev_` + 9 `dyn_`)
- `MULTI_YEAR_KEYS` = 9 core indicators for `avg3yr_` / `dyn_` / `prev_`

## Architecture

- **Data flow**: `data/raw/{year}/` → `data/processed/{year}/` → `data/modeling/`
- **Model**: 3-model ensemble (XGBoost + LightGBM + RandomForest), 36 features, predicting postseason (top 5)
- **dyn_** variables (the key innovation): `dyn_{k} = (1 - games_played_ratio) * avg3yr_{k}` — decays 3-year average influence linearly as season progresses
- **Postseason label**: `final_rank ≤ 5 → 1`, else `0`. No final rank label for in-progress seasons (2026).
- **Main entrypoints**: `main.py` (stub, for future use). Do not reference experiment scripts in `notebooks/experiments/jh/`.
- **New files**: 모든 신규 파일은 `notebooks/experiments/lsh/` 내부에 작성한다.

## CSV & formatting quirks

- **Encoding**: KBO CSVs use `utf-8-sig` (BOM-prefixed UTF-8). Fallback: `cp949` (Korean Windows). See `src/utils/parser.py:read_csv_korean()`.
- **Column names**: Korean strings (e.g., "최근10경기", "연속"). These are parsed by `src/utils/parser.py`.
- **Numeric parsing**: Commas, percent signs, and empty/`-` values in numeric columns; handled by `to_numeric_safe()`.

## Modules

All project imports use the `src.` prefix (e.g., `from src.utils.config import FEATURE_COLS`). Run from the project root so the `src` package is discoverable, or use `python -m` syntax.
