# dy 실험 산출물 정리

이 폴더는 `data/raw`와 `data/processed`를 입력으로 재구성한 KBO 가을야구 예측 전처리/시각화 결과입니다.

## 입력 데이터

- 원본 데이터: `C:\Users\Admin\Documents\GitHub\Ml_Baseball\data\raw`
- 과정 데이터: `C:\Users\Admin\Documents\GitHub\Ml_Baseball\data\processed`
- 사용 연도: 2022~2026

## 주요 결과

- 2026 최신 기준일: `2026-04-26`
- 학습 행 수: `6,125`
- 예측 Top5: `LG, KT, SSG, 삼성, KIA`

## 핵심 파일

- `KBO_가을야구_예측_전처리_dy.ipynb`: 발표/제출용 실행 노트북
- `dy_experiment_pipeline.py`: raw/processed 기반 재현 파이프라인
- `kbo_outputs/model_dataset_2022_2026.csv`: 모델 학습/예측용 데이터셋
- `kbo_outputs/2026_postseason_predictions.csv`: 2026 팀별 예측 확률
- `2026_가을야구_예측결과.csv`: 대시보드용 예측 결과
- `team_master_2022_2026.csv`: 2022~2026 팀 마스터
- `01_*.png` ~ `09_*.png`: 발표용 시각화 이미지
- `kbo_2022_2026_master_dashboard.html`: HTML 대시보드
