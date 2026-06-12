# modules/preprocess.py
"""
TriGuard AI - 전처리 모듈
인코딩 자동 감지, 지역명 표준화, 결측치 처리
"""

import re
import chardet
import pandas as pd
from difflib import get_close_matches

# ─────────────────────────────────────────────
# 지역명 상수
# ─────────────────────────────────────────────

# 병무청 지방청 권역 기준 (실제 CSV에서 쓰는 단위)
JIBANG_REGIONS = [
    "서울", "부산울산", "대구경북", "경인", "광주전남",
    "대전충남", "강원", "충북", "전북", "경남",
    "제주", "인천", "경기북부", "강원영동",
]

# 17개 시도 (행안부 인구, 질병관리청 지역별 CSV용)
STANDARD_REGIONS = [
    "서울", "경기", "부산", "대구", "인천", "광주",
    "대전", "울산", "세종", "강원", "충북", "충남",
    "전북", "전남", "경북", "경남", "제주",
]

# 시도 → 지방청 매핑 (1:N 허용 - 경기·강원은 두 지방청으로 분할)
SIDO_TO_JIBANG_MAP: dict[str, list[str]] = {
    "서울":  ["서울"],
    "부산":  ["부산울산"],
    "울산":  ["부산울산"],
    "대구":  ["대구경북"],
    "경북":  ["대구경북"],
    "인천":  ["인천"],
    "경기":  ["경인", "경기북부"],   # 경기도 → 경인(남부)/경기북부 동일값
    "광주":  ["광주전남"],
    "전남":  ["광주전남"],
    "대전":  ["대전충남"],
    "충남":  ["대전충남"],
    "세종":  ["대전충남"],
    "강원":  ["강원", "강원영동"],    # 강원도 → 강원/강원영동 동일값
    "충북":  ["충북"],
    "전북":  ["전북"],
    "경남":  ["경남"],
    "제주":  ["제주"],
}

# 시도 표준화 사전
REGION_REPLACE_DICT = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "강원특별자치도": "강원",
    "강원도": "강원",
    "충청북도": "충북",
    "충청남도": "충남",
    "전라북도": "전북",
    "전북특별자치도": "전북",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
}

# 병무청 지방청 표기 → 표준 권역명
JIBANG_REPLACE_DICT = {
    # 병역면제 CSV 표기
    "서울": "서울",
    "서 울": "서울",
    "부산·울산": "부산울산",
    "부산.울산": "부산울산",
    "부산울산": "부산울산",
    "대구·경북": "대구경북",
    "대구.경북": "대구경북",
    "대구경북": "대구경북",
    "경인": "경인",
    "경 인": "경인",
    "광주·전남": "광주전남",
    "광주.전남": "광주전남",
    "광주전남": "광주전남",
    "대전·충남": "대전충남",
    "대전.충남": "대전충남",
    "대전충남": "대전충남",
    "강원": "강원",
    "강 원": "강원",
    "충북": "충북",
    "충 북": "충북",
    "전북": "전북",
    "전 북": "전북",
    "경남": "경남",
    "경 남": "경남",
    "제주": "제주",
    "제 주": "제주",
    "인천": "인천",
    "인 천": "인천",
    "경기북부": "경기북부",
    "강원영동": "강원영동",
    # 입영현황 표기
    "서 울": "서울",
    "부산울산": "부산울산",
    "대구경북": "대구경북",
    "광주전남": "광주전남",
    "대전충남": "대전충남",
}


# ─────────────────────────────────────────────
# 유틸: 안전 나누기
# ─────────────────────────────────────────────

def safe_divide(numerator, denominator, default=0.0):
    """0 나누기 방지. denominator가 0이거나 NaN이면 default 반환."""
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except Exception:
        return default


# ─────────────────────────────────────────────
# 지역명 정규화
# ─────────────────────────────────────────────

def clean_region_text(name: str) -> str:
    """공백, 괄호, 특수문자 제거."""
    name = str(name).strip()
    name = re.sub(r"\(.*?\)", "", name)   # 경기(수원) → 경기
    name = re.sub(r"（.*?）", "", name)   # 전각 괄호
    name = name.replace(" ", "").replace("\u3000", "")
    return name


def normalize_region_sido(name: str) -> str:
    """17개 시도 기준 정규화 (행안부·질병관리청용)."""
    name = clean_region_text(name)
    if name in REGION_REPLACE_DICT:
        return REGION_REPLACE_DICT[name]
    if name in STANDARD_REGIONS:
        return name
    for region in STANDARD_REGIONS:
        if name.startswith(region):
            return region
    matches = get_close_matches(name, STANDARD_REGIONS, n=1, cutoff=0.75)
    return matches[0] if matches else name


def normalize_region_jibang(name: str) -> str:
    """병무청 지방청 권역 기준 정규화."""
    raw = str(name).strip()
    # 중간 점·가운데점 처리
    cleaned = raw.replace("·", "").replace(".", "").replace(" ", "")
    # 직접 매핑 우선
    if raw in JIBANG_REPLACE_DICT:
        return JIBANG_REPLACE_DICT[raw]
    if cleaned in JIBANG_REPLACE_DICT:
        return JIBANG_REPLACE_DICT[cleaned]
    # startswith fallback
    for std in JIBANG_REGIONS:
        if cleaned.startswith(std) or std.startswith(cleaned):
            return std
    # fuzzy
    matches = get_close_matches(cleaned, JIBANG_REGIONS, n=1, cutoff=0.6)
    return matches[0] if matches else cleaned


# ─────────────────────────────────────────────
# CSV 로더
# ─────────────────────────────────────────────

def detect_encoding(filepath: str) -> str:
    """chardet으로 인코딩 감지. 실패 시 cp949 fallback."""
    with open(filepath, "rb") as f:
        raw = f.read(20000)
    result = chardet.detect(raw)
    enc = result.get("encoding") or "cp949"
    # chardet이 ascii 반환하는 경우 cp949로 강제
    if enc.lower() in ("ascii", "utf-8"):
        # utf-8-sig 시도
        try:
            pd.read_csv(filepath, encoding="utf-8-sig", nrows=2)
            return "utf-8-sig"
        except Exception:
            pass
        try:
            pd.read_csv(filepath, encoding="cp949", nrows=2)
            return "cp949"
        except Exception:
            pass
    return enc


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """인코딩 자동 감지 CSV 로더."""
    enc = detect_encoding(filepath)
    for encoding in [enc, "cp949", "utf-8-sig", "euc-kr", "latin-1"]:
        try:
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            return df
        except Exception:
            continue
    raise ValueError(f"CSV 로드 실패: {filepath}")


def load_csv_from_upload(uploaded_file, **kwargs) -> pd.DataFrame:
    """Streamlit UploadedFile 객체에서 CSV 로드."""
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    result = chardet.detect(raw[:20000])
    enc = result.get("encoding") or "cp949"
    if enc.lower() in ("ascii",):
        enc = "cp949"
    for encoding in [enc, "cp949", "utf-8-sig", "euc-kr", "latin-1"]:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding, **kwargs)
            return df
        except Exception:
            continue
    raise ValueError(f"CSV 파일을 읽을 수 없습니다: {uploaded_file.name}")


# ─────────────────────────────────────────────
# 결측치 처리
# ─────────────────────────────────────────────

def clean_numeric(series: pd.Series) -> pd.Series:
    """
    '-', '~', 빈 문자열 등을 NaN으로 변환 후 숫자형으로 캐스팅.
    쉼표 포함 숫자 (예: 1,234) 처리.
    """
    s = series.astype(str).str.strip()
    s = s.replace({"-": None, "~": None, "": None, "nan": None, "N/A": None, "-": None})
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def clean_dataframe_numerics(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """숫자형이어야 하는 컬럼을 일괄 정제."""
    exclude_cols = exclude_cols or []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype == object:
            try:
                cleaned = clean_numeric(df[col])
                if cleaned.notna().sum() > 0:
                    df[col] = cleaned
            except Exception:
                pass
    return df


# ─────────────────────────────────────────────
# 병무청 데이터 파싱
# ─────────────────────────────────────────────

# 컬럼 매핑 딕셔너리 (실제 CSV 컬럼명 → 내부 표준명)
BYUNGMU_EXAM_COL_MAP = {
    "연도": "연도",
    "지방청": "지방청",
    " 처분인원 ": "처분인원",
    "처분인원": "처분인원",
    " 현역 ": "현역",
    "현역": "현역",
    " 보충역 ": "보충역",
    "보충역": "보충역",
    " 전시근로역 ": "전시근로역",
    "전시근로역": "전시근로역",
    " 병역면제 ": "병역면제",
    "병역면제": "병역면제",
    " 재신체검사 ": "재신체검사",
    "재신체검사": "재신체검사",
}

BYUNGMU_ENLIST_COL_MAP = {
    "구분": "지방청",
    "입영실통지": "입영실통지",
    "입영일자연기": "입영일자연기",
    "인도": "인도",
    "귀가": "귀가",
    "입영": "입영",
    "행방불명": "행방불명",
    "기피": "기피",
}

BYUNGMU_EXEMPT_COL_MAP = {
    "구 분": "지방청",
    "계": "계",
}


def parse_byungmu_exam(df: pd.DataFrame) -> pd.DataFrame:
    """병역판정검사 현황 파싱."""
    df = df.rename(columns={k: v for k, v in BYUNGMU_EXAM_COL_MAP.items() if k in df.columns})
    df.columns = df.columns.str.strip()
    numeric_cols = ["처분인원", "현역", "보충역", "전시근로역", "병역면제", "재신체검사"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
    if "지방청" in df.columns:
        df = df[df["지방청"] != "전체"].copy()
        df["지방청"] = df["지방청"].apply(normalize_region_jibang)
    if "연도" in df.columns:
        df["연도"] = pd.to_numeric(df["연도"], errors="coerce")
        df = df.dropna(subset=["연도"])
        df["연도"] = df["연도"].astype(int)
    return df


def parse_byungmu_enlist(df: pd.DataFrame) -> pd.DataFrame:
    """현역병 입영현황 파싱."""
    df = df.rename(columns={k: v for k, v in BYUNGMU_ENLIST_COL_MAP.items() if k in df.columns})
    df.columns = df.columns.str.strip()
    numeric_cols = ["입영실통지", "입영일자연기", "인도", "귀가", "입영", "행방불명", "기피"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
    if "지방청" in df.columns:
        df["지방청"] = df["지방청"].apply(normalize_region_jibang)
    return df


def parse_byungmu_exempt(df: pd.DataFrame) -> pd.DataFrame:
    """병역면제자 관리현황 파싱."""
    df = df.rename(columns={k: v for k, v in BYUNGMU_EXEMPT_COL_MAP.items() if k in df.columns})
    df.columns = df.columns.str.strip()
    if "계" in df.columns:
        df["계"] = clean_numeric(df["계"])
    if "지방청" in df.columns:
        df["지방청"] = df["지방청"].apply(normalize_region_jibang)
    return df


# ─────────────────────────────────────────────
# 질병관리청 데이터 파싱
# ─────────────────────────────────────────────

def parse_influenza(df: pd.DataFrame) -> pd.Series:
    """
    인플루엔자 표본감시 통계 (wide format).
    컬럼: 절기 | 1주~54주 의사환자분율
    반환: 절기별 최대 의사환자분율 Series (index=절기)
    """
    # 첫 컬럼이 절기
    col_절기 = df.columns[0]
    data_cols = df.columns[1:]
    numeric_df = df[data_cols].apply(pd.to_numeric, errors="coerce")
    result = pd.DataFrame({
        "절기": df[col_절기].astype(str).str.strip(),
        "최대분율": numeric_df.max(axis=1),
        "평균분율": numeric_df.mean(axis=1),
    })
    # '절기' 헤더 행 자체를 제거하되, '2020-2021절기' 같은 실제 데이터는 유지
    result = result[~result["절기"].str.fullmatch("절기|Unnamed.*", na=False)]
    result = result[result["절기"].str.strip() != ""]
    return result


def parse_ari(df: pd.DataFrame) -> pd.Series:
    """
    급성호흡기감염증 표본감시 통계.
    컬럼: 연도 | 월 | 총합계 | ...
    반환: 연도별 총합계 평균 Series
    """
    df = df.copy()
    df.columns = ["연도", "월", "총합계"] + list(df.columns[3:])
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce")
    df["총합계"] = pd.to_numeric(df["총합계"], errors="coerce")
    df = df.dropna(subset=["연도", "총합계"])
    result = df.groupby("연도")["총합계"].mean()
    return result


def aggregate_disease_by_jibang(regional_df: pd.DataFrame) -> pd.DataFrame:
    """
    시도별 감염병 발생률(parse_infectious_disease_regional 결과)을 지방청 단위로 집계.
    - 복수 시도 → 1 지방청 (부산+울산 → 부산울산): 평균
    - 1 시도 → 복수 지방청 (경기 → 경인/경기북부): 동일값 복사
    반환: DataFrame with columns [지방청, 총발생률]
    """
    rows = []
    for _, row in regional_df.iterrows():
        sido = str(row["시도"]).strip()
        rate = row["총발생률"]
        for jibang in SIDO_TO_JIBANG_MAP.get(sido, []):
            rows.append({"지방청": jibang, "총발생률": rate})

    if not rows:
        return pd.DataFrame(columns=["지방청", "총발생률"])

    df = pd.DataFrame(rows)
    # 복수 시도가 같은 지방청으로 합쳐지는 경우(부산울산 등) 평균
    return df.groupby("지방청", as_index=False)["총발생률"].mean()


def parse_infectious_disease_regional(df: pd.DataFrame) -> pd.DataFrame:
    """
    지역별 감염병 발생현황.
    실제 구조: 첫 컬럼=광역시도, 두번째 컬럼=시군구, 나머지=질병별 발생률
    첫 행이 헤더처럼 활용됨 → 광역 단위만 추출 후 합산.
    """
    df = df.copy()
    # 컬럼명 부여
    col_sido = df.columns[0]    # '전국' 등
    col_sigungu = df.columns[1]  # '전국.1' 등

    # 광역 단위 식별: 두 번째 컬럼이 첫 번째 컬럼과 동일한 행 = 광역 합계
    mask = df[col_sido].astype(str) == df[col_sigungu].astype(str)
    sido_df = df[mask].copy()
    sido_df = sido_df.rename(columns={col_sido: "시도", col_sigungu: "시도_확인"})
    sido_df["시도"] = sido_df["시도"].apply(normalize_region_sido)

    # 수치형 컬럼만 합산 → 총 감염병 발생률(incidence rate) 근사값
    numeric_cols = sido_df.columns[2:]
    sido_df[numeric_cols] = sido_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    sido_df["총발생률"] = sido_df[numeric_cols].sum(axis=1)

    return sido_df[["시도", "총발생률"]].reset_index(drop=True)


def parse_infectious_disease_national(df: pd.DataFrame) -> pd.DataFrame:
    """
    질병별 감염병 발생현황 (연도별).
    1급/2급/3급 구분 컬럼 포함.
    발생건수 합계로 등급별 가중합 계산용.
    """
    df = df.copy()
    col_grade = df.columns[0]   # '제1급' 등
    col_disease = df.columns[1]  # 질병명
    data_cols = df.columns[2:]
    numeric_df = df[data_cols].apply(pd.to_numeric, errors="coerce")

    # 1급 가중치 3, 2급 가중치 2, 3급 가중치 1
    grade_weights = {"제1급": 3, "제2급": 2, "제3급": 1}
    df["grade"] = df[col_grade].astype(str).str.strip()
    df["weight"] = df["grade"].map(grade_weights).fillna(1)
    df["발생합계"] = numeric_df.sum(axis=1)
    df["가중발생"] = df["발생합계"] * df["weight"]
    total_weighted = df["가중발생"].sum()
    return total_weighted


# ─────────────────────────────────────────────
# 방위사업청 데이터 파싱
# ─────────────────────────────────────────────

def parse_dapa_domestic(df: pd.DataFrame) -> dict:
    """
    방위사업청 국내조달 계약정보 파싱.
    반환: {총건수, 총금액, 수의계약건수, 수의계약금액, 업체목록}
    """
    df = df.copy()
    needed = ["계약체결방법명", "계약금액", "총계약금액", "대표업체명"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing}")

    df["계약금액"] = clean_numeric(df["계약금액"])
    total_count = len(df)
    total_amount = df["계약금액"].sum()

    suui = df[df["계약체결방법명"].astype(str).str.contains("수의", na=False)]
    suui_count = len(suui)
    suui_amount = suui["계약금액"].sum()

    company_counts = df["대표업체명"].value_counts()

    return {
        "총건수": total_count,
        "총금액": total_amount,
        "수의계약건수": suui_count,
        "수의계약금액": suui_amount,
        "업체별건수": company_counts,
    }


def parse_dapa_foreign(df: pd.DataFrame) -> dict:
    """방위사업청 국외조달 계약정보 파싱."""
    df = df.copy()
    total_count = len(df)
    return {"국외총건수": total_count}


def parse_strategic_goods(df: pd.DataFrame) -> dict:
    """전략물자 품목 키워드 CSV 파싱 → 품목 수."""
    return {"전략물자품목수": len(df)}
