# TriGuard

**TriGuard**는 AI 기반 위험 분석 및 검증 기능을 제공하는 프로젝트입니다.

Rule-based 위험 산출 결과를 기반으로 데이터를 분석하고, ML 기반 위험등급 분류 정합성 검증을 통해 분석 결과의 일관성과 신뢰성을 확인할 수 있도록 설계되었습니다. 사용자는 Streamlit 기반 대시보드를 통해 위험 현황, 주요 지표, 분석 결과를 직관적으로 확인할 수 있습니다.

---

## 프로젝트 개요

TriGuard는 단순한 데이터 시각화 도구가 아니라, 위험 요소를 체계적으로 산출하고 분석 결과의 정합성을 검증하는 것을 목표로 하는 AI 기반 분석 시스템입니다.

위험 산출 결과를 기반으로 지역별, 항목별 위험 수준을 확인하고, ML 보조 검증을 통해 Rule-based 분석 결과가 일관성 있게 분류되는지 확인할 수 있습니다.

---
### 데이터 정리

https://jungle-vision-612.notion.site/36a9778a382480a18991cf35e3a594e1?source=copy_link

---

## 주요 기능

### 1. 위험 지표 분석

Rule-based 방식으로 산출된 위험 점수를 기반으로 주요 위험 지표를 분석합니다.

### 2. 위험등급 분류

산출된 위험 점수를 기준으로 위험등급을 분류하고, 결과를 대시보드에서 확인할 수 있습니다.

### 3. ML 보조 검증

ML 기반 위험등급 분류 정합성 검증을 통해 Rule-based 결과와 ML 분류 결과 간의 일관성을 확인합니다.

### 4. Streamlit 대시보드

분석 결과를 사용자가 쉽게 확인할 수 있도록 Streamlit 기반 대시보드로 시각화합니다.

### 5. 향후 위험 변화 참고 분석

과거 및 현재 데이터를 기반으로 향후 위험 변화 가능성을 참고 분석합니다.

---

## 시스템 흐름

```text
데이터 수집 및 전처리
        ↓
Rule-based 위험 점수 산출
        ↓
위험등급 분류
        ↓
ML 기반 위험등급 분류 정합성 검증
        ↓
분석 결과 시각화
        ↓
Streamlit 대시보드에서 결과 확인
```

---

## 기술 스택

### Frontend / Dashboard

- Streamlit
- Plotly

### Data Analysis

- Python
- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- ML 기반 분류 모델

### Visualization

- Plotly
- 지도 기반 시각화
- 차트 기반 시각화

---

## 폴더 구조

```text
triguard/
├── data/                     # 분석에 사용되는 데이터 파일
├── pages/                    # Streamlit 페이지 구성
├── src/                      # 주요 분석 및 처리 코드
├── app.py                    # Streamlit 앱 실행 파일
├── requirements.txt          # 프로젝트 의존성 패키지 목록
├── README.md                 # 프로젝트 설명 문서
└── .gitignore                # Git 제외 파일 설정
```

---

## 실행 방법

본 프로젝트는 Python 환경에서 Streamlit을 통해 실행합니다.

### 1. 프로젝트 폴더로 이동

```bash
cd triguard
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. Streamlit 앱 실행

```bash
streamlit run app.py
```

실행 후 브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:8501
```

---

## 기대효과

TriGuard는 위험 산출 결과를 시각적으로 확인하고, ML 기반 정합성 검증을 통해 분석 결과의 신뢰성을 높일 수 있습니다.

이를 통해 단순한 위험 점수 산출에 그치지 않고, 데이터 기반의 위험 관리와 의사결정을 지원할 수 있습니다.

---

## 프로젝트 목표

TriGuard는 Rule-based 분석과 ML 보조 검증을 결합하여 위험 분석 결과의 일관성과 활용성을 높이는 것을 목표로 합니다.

데이터 기반 대시보드를 통해 위험 현황을 직관적으로 확인하고, 향후 위험 변화 가능성을 참고할 수 있는 분석 환경을 제공하고자 합니다.
