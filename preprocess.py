"""
preprocess.py - 데이터 전처리 모듈
- chardet 기반 인코딩 자동 감지
- 지역명 표준화 (공백/괄호 제거 → 사전 매핑 → startswith → fuzzy)
- 결측치 및 "-" 처리
- safe_divide 유틸
"""

import re
import chardet
import pandas as pd
from difflib import get_close_matches

# ──────────────────────────────────────────────
# 표준 지역명 & 매핑 사전
# ──────────────────────────────────────────────
STANDARD_REGIONS = [
    "서울", "경기", "부산", "대구", "인천", "광주",
    "대전", "울산", "세종", "강원", "충북", "충남",
    "전북", "전남", "경북", "경남", "제주",
]

REGION_REPLACE_DICT = {
    "서울특별시": "서울", "서울": "서울",
    "부산광역시": "부산", "부산": "부산", "부산울산": "부산", "부산·울산": "부산",
    "대구광역시": "대구", "대구": "대구",
    "인천광역시": "인천", "인천": "인천",
    "광주광역시": "광주", "광주": "광주",
    "대전광역시": "대전", "대전": "대전",
    "울산광역시": "울산", "울산": "울산",
    "세종특별자치시": "세종", "세종": "세종",
    "경기도": "경기", "경기": "경기",
    "강원특별자치도": "강원", "강원도": "강원", "강원": "강원",
    "충청북도": "충북", "충북": "충북",
    "충청남도": "충남", "충남": "충남",
    "전라북도": "전북", "전북특별자치도": "전북", "전북": "전북",
    "전라남도": "전남", "전남": "전남",
    "경상북도": "경북", "경북": "경북",
    "경상남도": "경남", "경남": "경남",
    "제주특별자치도": "제주", "제주": "제주",
}

# 병무청 지방청 → 표준 지역 매핑 (지방청명 포함)
MILITARY_REGION_MAP = {
    "서 울": "서울", "서울": "서울",
    "부산울산": "부산", "부산·울산": "부산", "부산": "부산",
    "대구": "대구", "인천": "인천", "광주": "광주",
    "대전": "대전", "울산": "울산",
    "경 기": "경기", "경기": "경기",
    "강 원": "강원", "강원": "강원",
    "충 북": "충북", "충북": "충북",
    "충 남": "충남", "충남": "충남",
    "전 북": "전북", "전북": "전북",
    "전 남": "전남", "전남": "전남",
    "경 북": "경북", "경북": "경북",
    "경 남": "경남", "경남": "경남",
    "제 주": "제주", "제주": "제주",
    "경인": "경기",
}


def clean_region_text(name: str) -> str:
    """공백 제거, 괄호 제거"""
    name = str(name).strip()
    name = re.sub(r"\(.*?\)", "", name)   # 경기(수원) → 경기
    name = name.replace(" ", "").replace("\u00a0", "")  # 일반 공백 + NBSP
    return name


def normalize_region(name: str) -> str:
    """지역명 → 표준 지역명 변환 (4단계 fallback)"""
    raw = str(name).strip()

    # 0) 병무청 지방청 직접 매핑 (공백 있는 원본 우선)
    if raw in MILITARY_REGION_MAP:
        return MILITARY_REGION_MAP[raw]

    cleaned = clean_region_text(raw)

    # 1) 사전 직접 매핑
    if cleaned in REGION_REPLACE_DICT:
        return REGION_REPLACE_DICT[cleaned]

    # 2) 표준명 직접 일치
    if cleaned in STANDARD_REGIONS:
        return cleaned

    # 3) startswith 매핑
    for region in STANDARD_REGIONS:
        if cleaned.startswith(region):
            return region

    # 4) Fuzzy matching (임계값 0.75)
    matches = get_close_matches(cleaned, STANDARD_REGIONS, n=1, cutoff=0.75)
    return matches[0] if matches else raw


# ──────────────────────────────────────────────
# CSV 로더
# ──────────────────────────────────────────────
def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """chardet으로 인코딩 감지 후 CSV 로드"""
    with open(path, "rb") as f:
        raw = f.read(20000)
    detected = chardet.detect(raw)
    enc = detected.get("encoding") or "cp949"
    # chardet이 ascii로 잘못 감지하는 경우 cp949 fallback
    if enc.lower() in ("ascii", "windows-1252"):
        enc = "cp949"
    try:
        return pd.read_csv(path, encoding=enc, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", errors="replace", **kwargs)


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def safe_divide(numerator, denominator, default=0.0):
    """0 나누기 방지"""
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except Exception:
        return default


def clean_numeric(series: pd.Series) -> pd.Series:
    """'-', 빈 문자열, NaN → 0 으로 변환"""
    return (
        series.replace(["-", "", " ", "　"], pd.NA)
              .fillna(0)
              .apply(lambda x: pd.to_numeric(x, errors="coerce"))
              .fillna(0)
    )


# ──────────────────────────────────────────────
# 병무청 전처리
# ──────────────────────────────────────────────
def preprocess_military_exam(df: pd.DataFrame) -> pd.DataFrame:
    """
    병역판정검사 현황 전처리
    컬럼: 연도, 지방청, 처분인원, 현역, 보충역, 전시근로역, 병역면제, 재신체검사
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    col_map = {
        "연도": "연도", "지방청": "지방청",
        "처분인원": "처분인원", "현역": "현역",
        "보충역": "보충역", "전시근로역": "전시근로역",
        "병역면제": "병역면제", "재신체검사": "재신체검사",
    }
    # 공백 포함 컬럼명 정규화
    rename = {}
    for c in df.columns:
        stripped = c.strip()
        if stripped in col_map:
            rename[c] = col_map[stripped]
    df = df.rename(columns=rename)

    required = ["연도", "지방청", "처분인원", "병역면제", "재신체검사"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼이 누락되었습니다: {missing}")

    # '전체' 행 제외, 숫자형 변환
    df = df[df["지방청"] != "전체"].copy()
    for col in ["처분인원", "현역", "보충역", "병역면제", "재신체검사"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    df["지역"] = df["지방청"].apply(normalize_region)
    return df


def preprocess_military_enlist(df: pd.DataFrame) -> pd.DataFrame:
    """
    현역병 입영현황 전처리
    컬럼: 구분, 입영실통지, 입영일자연기, 인도, 귀가, 입영, 행방불명, 기피
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = ["구분", "입영실통지", "입영"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼이 누락되었습니다: {missing}")

    for col in ["입영실통지", "입영일자연기", "인도", "귀가", "입영", "행방불명", "기피"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    df["지역"] = df["구분"].apply(normalize_region)
    return df


def preprocess_military_exempt(df: pd.DataFrame) -> pd.DataFrame:
    """
    병역면제자 관리현황 전처리
    컬럼: 구 분, 계, 연령별 ...
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # '구 분' → '구분'
    if "구 분" in df.columns:
        df = df.rename(columns={"구 분": "구분"})

    if "구분" not in df.columns or "계" not in df.columns:
        raise KeyError("필수 컬럼이 누락되었습니다: [구분, 계]")

    df["계"] = clean_numeric(df["계"])
    df["지역"] = df["구분"].apply(normalize_region)
    return df


# ──────────────────────────────────────────────
# 질병관리청 전처리
# ──────────────────────────────────────────────
def preprocess_disease_regional(path: str) -> pd.DataFrame:
    """
    지역별 감염병 발생현황 (인구 10만명당 발생률 형태)
    첫 두 컬럼: 지역(시도), 지역(시군구) + 나머지는 질병별 발생률
    """
    df = load_csv(path, header=0)
    df = df.copy()

    # 첫 행이 헤더처럼 쓰이는 경우 처리
    cols = list(df.columns)
    # 첫 두 컬럼을 시도/시군구로 rename
    df = df.rename(columns={cols[0]: "시도", cols[1]: "시군구"})

    # 시도 레벨만 추출 (시군구가 시도와 같은 행 = 시도 합계)
    df_sido = df[df["시도"] == df["시군구"]].copy()
    df_sido["지역"] = df_sido["시도"].apply(normalize_region)

    # 수치형 컬럼만 추출
    num_cols = [c for c in df_sido.columns if c not in ["시도", "시군구", "지역"]]
    for c in num_cols:
        df_sido[c] = pd.to_numeric(df_sido[c], errors="coerce").fillna(0)

    # 전체 발생률 합산 (총발생지수)
    df_sido["총발생률"] = df_sido[num_cols].sum(axis=1)
    return df_sido[["지역", "총발생률"]].reset_index(drop=True)


def preprocess_influenza(path: str) -> pd.DataFrame:
    """
    인플루엔자 표본감시: 절기별 주간 ILI 지수 → 절기 평균 산출
    첫 컬럼: 절기, 나머지: 주차별 ILI 값
    """
    df = load_csv(path, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "절기"})

    num_cols = [c for c in df.columns if c != "절기"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ILI평균"] = df[num_cols].mean(axis=1)
    # 최근 절기일수록 위험 높음 → 최신 절기 ILI 반환
    latest = df.sort_values("절기", ascending=False).iloc[0]["ILI평균"]
    overall_mean = df["ILI평균"].mean()
    # 전국 단일 값 → 모든 지역에 동일 적용
    return float(latest if pd.notna(latest) else overall_mean)


def preprocess_ari(path: str) -> pd.DataFrame:
    """
    급성호흡기감염증 표본감시
    컬럼 순서: 연도, 월, 전체, 0~6세, 7~12세, 13~18세, 19~49세, 50~64세, 65~74세, 75세이상
    (헤더 없이 첫 행이 첫 데이터인 파일)
    """
    df = load_csv(path, header=0)
    cols = list(df.columns)
    rename_map = {}
    expected = ["연도", "월", "전체", "0_6세", "7_12세", "13_18세", "19_49세", "50_64세", "65_74세", "75세이상"]
    for i, new_name in enumerate(expected):
        if i < len(cols):
            rename_map[cols[i]] = new_name
    df = df.rename(columns=rename_map)

    df["연도"] = pd.to_numeric(df.get("연도", 0), errors="coerce").fillna(0).astype(int)
    df["전체"] = pd.to_numeric(df.get("전체", 0), errors="coerce").fillna(0)

    # 최근 2년 평균 추세 → 전국 단일 지수
    recent = df[df["연도"] >= df["연도"].max() - 1]
    trend = recent["전체"].mean() if len(recent) > 0 else df["전체"].mean()
    overall = df["전체"].mean()
    return float(trend if pd.notna(trend) else overall)


# ──────────────────────────────────────────────
# 방위사업청 전처리
# ──────────────────────────────────────────────
def preprocess_domestic_contract(df: pd.DataFrame) -> pd.DataFrame:
    """국내조달 계약정보 전처리"""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    required = ["계약명", "계약체결방법명", "계약금액"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼이 누락되었습니다: {missing}")
    df["계약금액"] = pd.to_numeric(df["계약금액"], errors="coerce").fillna(0)
    df["총계약금액"] = pd.to_numeric(df.get("총계약금액", 0), errors="coerce").fillna(0)
    return df


def preprocess_foreign_contract(df: pd.DataFrame) -> pd.DataFrame:
    """국외조달 계약정보 전처리"""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    required = ["계약명", "대표업체명"]

    miss
    ing = [r for r in required if r not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼이 누락되었습니다: {missing}")
    return df


def preprocess_strategic_items(df: pd.DataFrame) -> pd.DataFrame:
    """전략물자 키워드 전처리"""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

