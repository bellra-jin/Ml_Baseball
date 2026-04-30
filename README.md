# ⚾🧢 VCT3 - Ml_Baseball

## KBO 포스트시즌 진출 확률 예측 솔루션

> KBO 정규시즌 데이터를 기반으로 팀별 포스트시즌 진출 가능성을 예측하고, Streamlit 대시보드로 확률 추이와 주요 피처를 시각화하는 프로젝트입니다.

---

# 1. 프로젝트 개요

## 📌 프로젝트명

**KBO 2026 포스트시즌 진출 예측 대시보드**

## 📅 진행 기간

**2026.04.24 ~ 2026.04.29**

## ✅ 프로젝트 소개

KBO는 10개 구단 중 5개 팀이 포스트시즌에 진출하기 때문에, 시즌 초반 순위만으로 최종 5강을 판단하기 어렵습니다.

본 프로젝트는 2016~2026년 KBO 팀/선수 기록을 수집하고, 현재 성적과 과거 전력 지표를 결합해 2026 시즌 각 팀의 포스트시즌 진출 확률을 예측합니다.

핵심은 단순 승률 예측이 아니라 **현재 시즌 흐름**, **전년도 전력**, **최근 3년 평균 전력의 동적 감쇠값(`dyn_`)**을 함께 사용해 시즌 초반에도 설명 가능한 확률을 제공하는 것입니다.

---

# 2. 프로젝트 필요성

- **초반 순위의 불안정성:** 4월 순위만으로 최종 포스트시즌 팀을 예측하면 변동성이 큽니다.
- **KBO 구조 특성:** 10개 팀 중 5개 팀이 진출하므로 단순 정확도 0.5는 사실상 무작위 또는 기준선 수준입니다.
- **과거 전력 반영 필요:** 개막 직후에는 현재 시즌 표본이 작아 전년도와 최근 3년 지표를 함께 봐야 합니다.
- **해석 가능한 예측 필요:** 단순히 "진출/미진출"만 보여주는 것이 아니라 어떤 지표가 확률에 영향을 주는지 설명해야 합니다.

---

# 3. 프로젝트 목표

- **포스트시즌 진출 가능성 예측:** 2026 시즌 각 팀의 5강 진입 확률을 산출합니다.
- **시즌 진행도 반영:** 시즌 초반에는 과거 전력의 영향을 크게 보고, 후반으로 갈수록 현재 시즌 성적을 더 강하게 반영합니다.
- **예측 근거 시각화:** 팀별 확률, 순위 변화, 피처 중요도, 검증 리포트를 대시보드에서 확인할 수 있도록 구성합니다.
- **재현 가능한 데이터 파이프라인 구축:** raw CSV 수집부터 전처리, 학습 데이터 생성, 예측 데이터 생성까지 일관된 구조로 관리합니다.

---

# 4. 팀 소개

## 👥 팀명

**VCT3**

> KBO 데이터 수집, 전처리, 모델링, 검증, Streamlit 대시보드 구현을 역할별로 나누어 진행했습니다.

| <div style="font-size:100px;">😎</div> | <div style="font-size:100px;">🫣</div> | <div style="font-size:100px;">🫠</div> | <div style="font-size:100px;">🤯</div> |
| :---: | :---: | :---: | :---: |
| 이준모 | 박진희 | 이승형 | 김도연 |
| **팀장** | 팀원 | 팀원 | 팀원 |
| <a href="https://github.com/Jaydenlee07"><img src="https://img.shields.io/badge/GitHub-Jaydenlee07-181717?logo=github"></a> | <a href="https://github.com/bellra--jin"><img src="https://img.shields.io/badge/GitHub-bellra--jin-181717?logo=github"></a> | <a href="https://github.com/PiazzaSanPietro"><img src="https://img.shields.io/badge/GitHub-PiazzaSanPietro-181717?logo=github"></a> | <a href="https://github.com/doingkite"><img src="https://img.shields.io/badge/GitHub-doingkite-181717?logo=github"></a> |
| 키워드 크롤링·워드클라우드 / 데이터 분석·전처리 / 발표 리포트 정리 | 변수 선정·데이터 전처리 / 모델 학습·검증 / Streamlit 대시보드 구현 | 모델링 데이터셋 구축 / 모델 학습·검증 / KBO 데이터 크롤링·전처리 | 데이터 분석·전처리 / 시각화·대시보드 구성 / 확률 기반 모델 학습 |


---

# 5. 프로젝트 설계

## 🗂 데이터 파이프라인

```bash
data/raw/{year}/
    -> data/processed/{year}/
        -> data/modeling/train_dataset.csv
        -> data/modeling/predict_dataset_2026.csv
            -> Streamlit dashboard
```

## 🕷 데이터 수집 및 크롤링

KBO 공식 사이트(`koreabaseball.com`)에서 2016~2026년 팀/선수 기록을 수집해 `data/raw/{year}/`에 CSV로 저장했습니다.

| 수집 구분 | 파일 예시 | 활용 목적 |
| :--- | :--- | :--- |
| 팀 일자별 순위 | `team_daily_rank.csv` | 날짜별 순위, 승률, 게임차, 최근 흐름 피처 생성 |
| 팀 최종 순위 | `team_final_rank.csv` | 포스트시즌 진출 여부 라벨 생성 |
| 팀 타격 기록 | `team_hitter_basic.csv` | 팀 공격력, OPS, 장타력 관련 피처 생성 |
| 팀 투수 기록 | `team_pitcher_basic.csv` | ERA, WHIP, K/BB 등 투수력 피처 생성 |
| 팀 수비 기록 | `team_defense_basic.csv` | 수비율, 실책 등 수비 안정성 반영 |
| 팀 주루 기록 | `team_runner_basic.csv` | 도루, 주루 관련 보조 지표 반영 |
| 선수 타자 기록 | `player_hitter_basic.csv`, `player_hitter_detail.csv` | 상위 타자 OPS, 타선 집중도 등 전력 피처 생성 |
| 선수 투수 기록 | `player_pitcher_basic.csv`, `player_pitcher_detail.csv` | 에이스 ERA, 상위 투수 지표 등 전력 피처 생성 |

크롤링 대상 사이트는 ASP.NET WebForms 기반이라 페이지 이동과 연도/팀 선택이 `__doPostBack` 방식으로 동작합니다. 따라서 단순 URL 변경이 아니라 GET으로 받은 form hidden field(`__VIEWSTATE`, `__EVENTVALIDATION`, `__EVENTTARGET` 등)를 유지한 뒤 POST 요청을 보내는 방식으로 페이지네이션을 처리했습니다.

수집 과정에서는 아래 처리를 함께 적용했습니다.

- `requests.Session()`으로 쿠키와 상태값을 유지합니다.
- `Basic1`과 `Basic2`처럼 나뉜 기록 페이지는 선수ID/팀명 기준으로 병합합니다.
- 선수 basic/detail 파일은 공통 선수 집합으로 맞춰 후속 피처 생성 시 불일치를 줄입니다.
- 핵심 지표가 비어 있는 선수 행은 제거해 학습용 통계 품질을 높입니다.
- 모든 raw CSV는 한글 호환을 위해 `utf-8-sig`로 저장합니다.
- 진행 중인 2026 시즌은 `update_2026_raw.py`로 기존 2016~2025 파일을 건드리지 않고 `data/raw/2026/`만 갱신할 수 있게 분리했습니다.

---

# 6. 변수 선정 및 피처 설계

## ✅ 후보 변수 생성 범위

처음부터 모델에 넣을 변수 20개만 만든 것이 아니라, 크롤링한 원본 컬럼과 파생변수를 모두 포함해 훨씬 넓은 후보군을 만들었습니다. 최종 학습 데이터(`train_dataset.csv`)에는 날짜, 팀명, 원본 순위 기록, 최근 경기 파싱값, 홈/원정 성적, 전년도 팀/선수 요약, 3년 평균, 추세, `dyn_`, 라벨까지 포함해 총 122개 컬럼이 존재합니다.

이 중 `team`, `date`, 문자열 원본 컬럼, `final_rank`, `postseason`처럼 모델 입력으로 쓰면 안 되거나 직접 수치화 대상이 아닌 컬럼은 제외했습니다. 그 다음 실제 예측 시점에 사용할 수 있고, 시즌별로 동일하게 만들 수 있으며, 해석 가능한 변수만 추려 모델 입력 후보 36개를 구성했습니다.

## 📐 후보군 정제 기준

| 기준 | 적용 내용 |
| :--- | :--- |
| 데이터 누수 방지 | 시즌 종료 후에만 알 수 있는 `final_rank`, `postseason`은 라벨 생성에만 사용 |
| 예측 시점 일관성 | 과거 시즌의 특정 날짜에도 만들 수 있는 변수만 모델 입력 후보로 유지 |
| 도메인 설명력 | 순위, 승률, 게임차, 전년도 투타 전력, 득실 기반 기대승률처럼 포스트시즌과 연결되는 지표 우선 |
| 중복·불안정 변수 축소 | 의미가 겹치거나 결측/변동성이 큰 세부 컬럼은 후보에서 제외 |
| 일반화 성능 확인 | 중요도뿐 아니라 LOSO-CV의 Test AUC, F1, Brier Score, 과적합 갭을 함께 확인 |

## 📌 최종 피처 구조

정제된 36개 모델 후보는 크게 세 그룹으로 구성했습니다.

| 구분 | 설명 |
| :--- | :--- |
| 현재 시즌 피처 | 현재 순위, 승률, 5위와의 게임차, 홈/원정 승률, 최근 흐름 등 |
| `prev_` 피처 | 전년도 피타고라스 승률, 득실차, 팀 ERA, K/BB, 상위 타자 OPS 등 |
| `dyn_` 피처 | 최근 3년 평균 전력을 시즌 진행도에 따라 점점 줄여 반영하는 동적 피처 |

`dyn_` 피처는 다음 방식으로 계산했습니다.

```text
dyn_X = (1 - games_played_ratio) * avg3yr_X
```

시즌 초반에는 `avg3yr_` 기반 전력이 크게 반영되고, 시즌이 진행될수록 `dyn_` 값이 0에 가까워져 현재 시즌 성적이 예측을 주도합니다.

최종적으로는 36개 후보 중 XGBoost, LightGBM, RandomForest의 변수 중요도를 평균해 상위 20개를 선별했습니다. 즉, 전체 원천/파생 컬럼을 바로 줄인 것이 아니라 **122개 컬럼 → 모델 입력 후보 36개 → 최종 Top 20 피처** 순서로 단계적으로 압축했습니다.


---

# 7. 머신러닝 모델 선정 이유

## ✅ 최종 채택 모델: Strategy C 앙상블

최종 모델은 `Logistic Regression + RandomForest + lightXGB + lightLGBM`을 동일 가중치로 평균하는 소프트 보팅 앙상블입니다.

| 항목 | 내용 |
| :--- | :--- |
| 학습 기간 | 2017~2025 시즌 |
| 예측 대상 | 2026 시즌 |
| 검증 방식 | LOSO-CV, 각 시즌을 한 번씩 테스트셋으로 분리 |
| 사용 피처 | 중요도 기반 Top 20 피처 |
| 평균 Test AUC | 0.848 |
| 평균 Accuracy | 0.762 |
| 평균 F1 | 0.762 |
| 평균 Brier Score | 0.163 |

채택 이유는 다음과 같습니다.

- **일반화 검증 범위가 가장 넓음:** 2017~2025년 9개 시즌을 사용해 시즌 단위 검증을 수행했습니다.
- **과적합을 줄인 구조:** XGBoost와 LightGBM은 `max_depth=2`, `n_estimators=40`으로 제한하고, RandomForest도 얕은 트리로 구성했습니다.
- **초반 예측 목적과 잘 맞음:** `prev_`와 `dyn_` 피처를 함께 사용해 시즌 초반 작은 표본을 보완합니다.
- **대시보드와 직접 연결 가능:** 최종 산출물이 Streamlit 앱의 예측, 피처 중요도, 검증 리포트와 바로 연결됩니다.


## ❌ LSH 모델을 채택하지 않은 이유

LSH 실험은 TPOT AutoML 기반으로 파이프라인을 탐색했고, 정규화와 로지스틱 회귀를 통해 해석 가능한 모델을 만들었다는 장점이 있습니다. 정규화는 특정 피처나 가중치가 과하게 튀지 않도록 값을 안정화해 처음 보는 데이터에서도 더 나은 일반화 성능을 기대하게 해주는 장치입니다.

다만 최종 모델로 쓰기에는 초반 예측력이 부족했습니다.

| 체크포인트 | 경기 수 | Accuracy | AUC | F1 |
| :--- | ---: | ---: | ---: | ---: |
| M1 | 36경기 | 0.500 | 0.640 | 0.545 |
| M2 | 72경기 | 0.500 | 0.640 | 0.545 |
| M3 | 108경기 | 0.600 | 0.680 | 0.600 |
| M4 | 144경기 | 0.900 | 1.000 | 0.909 |

KBO는 10개 팀 중 5개 팀이 포스트시즌에 진출하므로, 정확도 0.5는 사실상 무작위 수준입니다. 특히 M1, M2에서 0.5가 나온 것은 시즌 초반과 중반에 모델이 의미 있는 패턴을 충분히 잡지 못했다는 신호입니다. 이 프로젝트는 초반에도 예측 가능해야 의미가 있으므로 최종 채택에서는 제외했습니다.

## ❌ dy_final 모델을 채택하지 않은 이유

dy_final 실험은 2023~2025년 일자별 팀 스냅샷 4,629행을 기반으로 현재 순위 상태와 전년도 지표를 활용했습니다. 2026년 4월 24일 기준 예측 Top5는 `LG, KT, 삼성, SSG, KIA`로 산출되었고, 일자별 데이터 기반 분석이라는 장점이 있었습니다.

하지만 최종 채택 모델로 쓰기에는 아래 한계가 있었습니다.

- **검증 시즌이 짧음:** 학습/검증 중심 데이터가 2023~2025년에 집중되어, 2017~2025 전체 시즌을 활용한 최종 모델보다 일반화 근거가 약했습니다.
- **초반 예측에서 기준선 대비 이득이 작음:** 4월 최신 시점 검증의 Top5 적중 평균이 3팀이었고, 단순 현재 Top5 기준선도 동일하게 3팀이었습니다.
- **현재 순위 보정 의존도가 있음:** 최종 확률이 모델 확률과 스탠딩 확률을 섞는 구조라, 순수한 피처 기반 모델 성능을 평가하기 어렵습니다.
- **`dyn_` 핵심 구조와의 연결이 약함:** 우리 프로젝트의 핵심 혁신은 3년 평균 전력을 시즌 진행도에 따라 감쇠시키는 `dyn_` 피처인데, dy_final은 이 구조가 최종 파이프라인과 완전히 맞물리지 않았습니다.

따라서 dy_final은 분석 참고 자료로 활용하고, 최종 예측 모델은 더 긴 시즌 범위와 `dyn_` 기반 피처를 반영한 Strategy C 앙상블로 결정했습니다.

---

# 8. 검증 리포트

## 📊 핵심 포인트

- 검증은 시즌 단위 누수를 막기 위해 **Leave-One-Season-Out CV**로 수행했습니다.
- 평균 Test AUC는 **0.848**로, 시즌별 변동은 있지만 전반적으로 포스트시즌 팀과 비포스트시즌 팀을 구분하는 힘이 확인되었습니다.
- 평균 Accuracy와 F1은 모두 **0.762** 수준으로 나타났습니다.
- 2021, 2025 시즌은 예측 난도가 높아 Train/Test AUC 갭이 크게 나타났고, 이 부분은 향후 부상, 외국인 선수 교체, 트레이드 등 외부 변수를 추가해 보완할 수 있습니다.

## 🔮 2026 최신 예측 결과

기준일: **2026-04-29**

| 순위 | 팀 | 정규화 확률 |
| ---: | :--- | ---: |
| 1 | LG | 96.9% |
| 2 | KT | 83.6% |
| 3 | 삼성 | 76.0% |
| 4 | SSG | 72.3% |
| 5 | KIA | 45.9% |
| 6 | 한화 | 34.8% |
| 7 | NC | 32.6% |
| 8 | 두산 | 25.4% |
| 9 | 롯데 | 18.9% |
| 10 | 키움 | 13.7% |

---

# 9. 주요 기능

- **홈 대시보드:** 2026 기준 팀별 포스트시즌 진출 확률과 Top5 예측팀 표시
<img src="src/assets/images/team/app.png" width="100%" height="auto" alt="app">
- **추이 분석:** 날짜별 확률 변화와 순위 변화를 시각화
<img src="src/assets/images/team/추이분석.png" width="100%" height="auto" alt="추이분석">
- **피처 분석:** 모델이 중요하게 본 지표와 팀별 특성 비교
<img src="src/assets/images/team/피처분석.png" width="100%" height="auto" alt="피처분석">
- **모델 소개:** 앙상블 구조, 피처셋, `dyn_` 변수 설명
<img src="src/assets/images/team/모델소개.png" width="100%" height="auto" alt="모델소개">
- **검증 리포트:** LOSO-CV 성능, 과적합 갭, 캘리브레이션, 체크포인트 적중률 확인
<img src="src/assets/images/team/검증리포트.png" width="100%" height="auto" alt="검증리포트">
- **분석 보고서:** 최종 예측 결과와 해석을 발표용 형태로 정리
<img src="src/assets/images/team/분석보고서.png" width="100%" height="auto" alt="분석보고서">

---

# 10. 기술 스택

## ETL

![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Requests](https://img.shields.io/badge/Requests-20232A?style=for-the-badge)

## Machine Learning

![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-111827?style=for-the-badge)

## Dashboard / Visualization

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

## Package / Collaboration

![uv](https://img.shields.io/badge/uv-package%20manager-5C4EE5?style=for-the-badge)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

# 11. 트러블 슈팅

## 1) 시즌 초반 데이터 부족 문제

**상황**

2026 시즌은 4월 기준 팀당 20경기대만 진행되어 현재 시즌 데이터만으로는 예측 표본이 부족했습니다.

**해결**

전년도(`prev_`) 지표와 최근 3년 평균을 감쇠시키는 `dyn_` 지표를 추가해 초반 예측 안정성을 보완했습니다.

## 2) 과적합 문제

**상황**

초기 앙상블 모델은 Train AUC가 높지만 Test AUC와 차이가 벌어지는 문제가 있었습니다.

**해결**

피처를 중요도 기반 Top 20으로 줄이고, XGBoost/LightGBM의 깊이와 라운드를 강하게 제한했습니다. RandomForest도 얕은 트리와 큰 leaf를 사용해 과적합을 낮췄습니다.

## 3) 실험 모델 비교 기준 정리

**상황**

TPOT 기반 LSH 모델, dy_final 분석 모델, Strategy C 앙상블 등 여러 후보가 존재했습니다.

**해결**

단순 종료 시점 성능보다 **초반 예측력**, **LOSO-CV 일반화 성능**, **프로젝트 핵심 피처(`dyn_`) 반영 여부**, **앱 파이프라인 연결성**을 기준으로 최종 모델을 선정했습니다.

---

# 12. 비고

- KBO 원본 CSV는 `utf-8-sig` 인코딩을 기본으로 사용하며, 필요 시 `cp949`로 fallback 처리합니다.
- 2026 시즌은 진행 중이므로 예측 결과는 확정 순위가 아니라 현재까지의 데이터 기반 가능성입니다.
- 향후 부상자, 외국인 선수 교체, 트레이드, 잔여 일정 난이도 데이터를 추가하면 예측 설명력을 더 높일 수 있습니다.
