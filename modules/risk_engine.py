# modules/risk_engine.py
"""
TriGuard AI - Risk Score 계산 엔진
인력 / 감염병 DC / 물자 / 통합 Risk Score
"""

import numpy as np
import pandas as pd
from modules.preprocess import safe_divide


# ─────────────────────────────────────────────
# 가중치 상수
# ─────────────────────────────────────────────

WEIGHTS_MANPOWER = {
    "입영인원_감소율": 0.30,
    "병역판정검사_감소율": 0.25,
    "병역면제율": 0.20,
    "재신체검사율": 0.10,
    "입영_차질률": 0.15,
}

WEIGHTS_DISEASE = {
    "지역별_발생률": 0.35,
    "질병_등급_가중합": 0.30,
    "인플루엔자_유행_강도": 0.20,
    "급성호흡기_트렌드": 0.15,
}

WEIGHTS_MATERIAL = {
    "국내조달_계약_감소율": 0.25,
    "국외조달_의존도": 0.25,
    "공급업체_집중도": 0.20,
    "수의계약_의존도": 0.15,
    "전략물자_관련_계약_비율": 0.15,
}

WEIGHTS_INTEGRATED = {
    "인력": 0.40,
    "감염병": 0.40,
    "물자": 0.20,
}

# 위험 등급 임계값
GRADE_DANGER = 60.0
GRADE_CAUTION = 35.0


# ─────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────

def validate_weights(weights: dict, name: str) -> list:
    """가중치 합이 1.0인지 검증."""
    total = sum(weights.values())
    warnings = []
    if abs(total - 1.0) > 0.01:
        warnings.append(
            f"⚠️ [{name}] 가중치 합계가 {total:.3f}입니다. "
            f"기대값은 1.000입니다."
        )
    return warnings


def clip_score(score: float) -> float:
    """Risk Score를 0~100 범위로 제한."""
    try:
        return float(np.clip(score, 0.0, 100.0))
    except Exception:
        return 0.0


def grade(score: float) -> str:
    """점수 → 위험 등급."""
    if score >= GRADE_DANGER:
        return "위험"
    elif score >= GRADE_CAUTION:
        return "주의"
    else:
        return "정상"


# ─────────────────────────────────────────────
# 인력 Risk Score
# ─────────────────────────────────────────────

def calc_manpower_risk(
    exam_df: pd.DataFrame,
    enlist_df: pd.DataFrame,
    exempt_df: pd.DataFrame,
    population_df: pd.DataFrame = None,
) -> tuple:
    """
    지방청별 인력 Risk Score 계산.

    exam_df: 병역판정검사
    enlist_df: 현역병 입영현황
    exempt_df: 병역면제자 관리현황
    population_df: 행정안전부 인구 데이터
    """
    warnings = validate_weights(WEIGHTS_MANPOWER, "인력 Risk")
    results = []

    if exam_df is None or exam_df.empty:
        return pd.DataFrame(), ["병역판정검사 데이터가 없습니다."]

    if "연도" not in exam_df.columns or "지방청" not in exam_df.columns:
        return pd.DataFrame(), ["병역판정검사 데이터에 연도 또는 지방청 컬럼이 없습니다."]

    years = sorted(exam_df["연도"].dropna().unique())
    if not years:
        return pd.DataFrame(), ["병역판정검사 데이터에서 연도 정보를 찾을 수 없습니다."]

    latest_year = years[-1]
    target_base_year = latest_year - 6
    base_candidates = [y for y in years if y <= target_base_year]
    prev_year = base_candidates[-1] if base_candidates else years[0]
    year_gap = max(latest_year - prev_year, 1)

    latest_exam = exam_df[exam_df["연도"] == latest_year]
    prev_exam = exam_df[exam_df["연도"] == prev_year]

    regions = latest_exam["지방청"].dropna().unique()

    for region in regions:
        row_now = latest_exam[latest_exam["지방청"] == region]
        row_prev = prev_exam[prev_exam["지방청"] == region]
        row_enl = enlist_df[enlist_df["지방청"] == region] if enlist_df is not None and not enlist_df.empty and "지방청" in enlist_df.columns else pd.DataFrame()
        row_ex = exempt_df[exempt_df["지방청"] == region] if exempt_df is not None and not exempt_df.empty and "지방청" in exempt_df.columns else pd.DataFrame()

        if row_now.empty:
            continue

        exam_now = float(row_now["현역"].values[0]) if "현역" in row_now.columns else 0.0
        exam_prev = float(row_prev["현역"].values[0]) if not row_prev.empty and "현역" in row_prev.columns else exam_now

        raw_입영감소 = safe_divide(exam_prev - exam_now, exam_prev + 1e-9) * 100
        입영_감소율 = clip_score(raw_입영감소 * (6 / year_gap))

        proc_now = float(row_now["처분인원"].values[0]) if "처분인원" in row_now.columns else 0.0
        proc_prev = float(row_prev["처분인원"].values[0]) if not row_prev.empty and "처분인원" in row_prev.columns else proc_now

        raw_검사감소 = safe_divide(proc_prev - proc_now, proc_prev + 1e-9) * 100
        검사_감소율 = clip_score(raw_검사감소 * (6 / year_gap))

        exam_면제 = float(row_now["병역면제"].values[0]) if "병역면제" in row_now.columns else 0.0
        면제율 = clip_score(safe_divide(exam_면제, proc_now) * 100)

        재검 = float(row_now["재신체검사"].values[0]) if "재신체검사" in row_now.columns else 0.0
        재검율 = clip_score(safe_divide(재검, proc_now) * 100)

        miss = float(row_enl["행방불명"].values[0]) if not row_enl.empty and "행방불명" in row_enl.columns else 0.0
        evas = float(row_enl["기피"].values[0]) if not row_enl.empty and "기피" in row_enl.columns else 0.0
        notif = float(row_enl["입영실통지"].values[0]) if not row_enl.empty and "입영실통지" in row_enl.columns else 1.0
        차질률 = clip_score(safe_divide(miss + evas, notif) * 100)

        pop_penalty = 0.0
        if population_df is not None and not population_df.empty and "지방청" in population_df.columns:
            pop_row = population_df[population_df["지방청"] == region]
            if not pop_row.empty and "병역자원비율" in pop_row.columns:
                ratio = float(pop_row["병역자원비율"].values[0])
                pop_penalty = clip_score((7.0 - ratio) / 7.0 * 40)

        score = (
            WEIGHTS_MANPOWER["입영인원_감소율"] * 입영_감소율 +
            WEIGHTS_MANPOWER["병역판정검사_감소율"] * 검사_감소율 +
            WEIGHTS_MANPOWER["병역면제율"] * 면제율 +
            WEIGHTS_MANPOWER["재신체검사율"] * 재검율 +
            WEIGHTS_MANPOWER["입영_차질률"] * 차질률
        )

        # 행안부 인구 보정 10% 반영
        score = score * 0.9 + pop_penalty * 0.1

        results.append({
            "지방청": region,
            "입영인원_감소율": round(입영_감소율, 2),
            "병역판정검사_감소율": round(검사_감소율, 2),
            "병역면제율": round(면제율, 2),
            "재신체검사율": round(재검율, 2),
            "입영_차질률": round(차질률, 2),
            "인구_보정점수": round(pop_penalty, 2),
            "인력Risk": round(clip_score(score), 2),
        })

    return pd.DataFrame(results), warnings


# ─────────────────────────────────────────────
# 감염병 Disruption Coefficient
# ─────────────────────────────────────────────

def calc_disease_dc(
    regional_df: pd.DataFrame,
    national_weighted: float,
    influenza_df: pd.DataFrame,
    ari_series: pd.Series,
    jibang_disease_df: pd.DataFrame = None,
) -> tuple:
    """
    감염병 Disruption Coefficient 계산.
    반환: dc_score_national, regional_df_scored, jibang_dc_df, components, warnings
    """
    warnings = validate_weights(WEIGHTS_DISEASE, "감염병 DC")

    # ② 질병 등급 가중합
    try:
        grade_score = clip_score(
            np.log1p(national_weighted) / np.log1p(10000) * 100
        ) if national_weighted and national_weighted > 0 else 0.0
    except Exception:
        grade_score = 0.0

    # ③ 인플루엔자 유행강도
    if influenza_df is not None and not influenza_df.empty:
        if "최대분율" in influenza_df.columns:
            flu_score = clip_score(float(influenza_df["최대분율"].max()) * 2)
        elif "평균분율" in influenza_df.columns:
            flu_score = clip_score(float(influenza_df["평균분율"].max()) * 2)
        else:
            flu_score = 30.0
    else:
        flu_score = 30.0

    # ④ 급성호흡기 트렌드
    if ari_series is not None and len(ari_series) >= 2:
        vals = ari_series.sort_index().values
        current = vals[-1]
        historical_max = max(vals)
        ari_score = clip_score(safe_divide(current, historical_max) * 60)
    else:
        ari_score = 30.0

    def _dc_from_regional_index(regional_idx: float) -> float:
        return clip_score(
            WEIGHTS_DISEASE["지역별_발생률"] * regional_idx +
            WEIGHTS_DISEASE["질병_등급_가중합"] * grade_score +
            WEIGHTS_DISEASE["인플루엔자_유행_강도"] * flu_score +
            WEIGHTS_DISEASE["급성호흡기_트렌드"] * ari_score
        )

    jibang_dc_df = None

    # 지방청별 감염병 DC
    if jibang_disease_df is not None and not jibang_disease_df.empty and "총발생률" in jibang_disease_df.columns:
        jd = jibang_disease_df.copy()

        RATE_MIN = 800.0
        RATE_MAX = 1600.0

        jd["발생률지수"] = jd["총발생률"].apply(
            lambda x: clip_score(safe_divide(x - RATE_MIN, RATE_MAX - RATE_MIN) * 100)
        )
        jd["감염병DC"] = jd["발생률지수"].apply(_dc_from_regional_index).round(2)

        jibang_dc_df = jd[["지방청", "발생률지수", "감염병DC"]]
        avg_regional = jd["발생률지수"].mean()

    elif regional_df is not None and not regional_df.empty and "총발생률" in regional_df.columns:
        regional_df = regional_df.copy()
        max_rate = regional_df["총발생률"].max()
        regional_df["발생률지수"] = regional_df["총발생률"].apply(
            lambda x: clip_score(safe_divide(x, max_rate) * 100)
        )
        avg_regional = regional_df["발생률지수"].mean()

    else:
        regional_df = pd.DataFrame()
        avg_regional = 50.0

    if regional_df is not None and not regional_df.empty and "총발생률" in regional_df.columns and "발생률지수" not in regional_df.columns:
        regional_df = regional_df.copy()
        max_rate = regional_df["총발생률"].max()
        regional_df["발생률지수"] = regional_df["총발생률"].apply(
            lambda x: clip_score(safe_divide(x, max_rate) * 100)
        )

    dc_score_national = _dc_from_regional_index(avg_regional)

    components = {
        "지역별_발생률_지수(전국평균)": round(avg_regional, 2),
        "질병_등급_가중합_지수": round(grade_score, 2),
        "인플루엔자_강도_지수": round(flu_score, 2),
        "급성호흡기_트렌드_지수": round(ari_score, 2),
    }

    return dc_score_national, regional_df, jibang_dc_df, components, warnings


# ─────────────────────────────────────────────
# 물자 Risk Score
# ─────────────────────────────────────────────

def calc_material_risk(
    domestic: dict,
    foreign: dict,
    strategic: dict,
    bidders: dict = None,
) -> tuple:
    """
    방위사업청 기반 전국 물자 Risk Score.
    지역 단위 없음 → 전국 단일값. 모든 지방청에 동일 적용.

    활용 데이터:
    - 국내조달 계약정보
    - 국외조달 계약정보
    - 국내조달 입찰참여업체정보
    - 전략물자 품목키워드 및 개정정보
    """
    warnings = validate_weights(WEIGHTS_MATERIAL, "물자 Risk")

    if domestic is None:
        domestic = {}
    if foreign is None:
        foreign = {}
    if strategic is None:
        strategic = {}
    if bidders is None:
        bidders = {}

    total_domestic = domestic.get("총건수", 1) or 1
    total_foreign = foreign.get("국외총건수", 0) or 0
    total_all = total_domestic + total_foreign

    if total_all <= 0:
        total_all = 1

    # ① 국내조달 계약 감소율
    # 단년 데이터이므로 국내조달 비중이 낮을수록 위험하다고 대리 산출
    내외비율 = safe_divide(total_domestic, total_all) * 100
    국내감소율지수 = clip_score(100 - 내외비율)

    # ② 국외조달 의존도
    국외의존도 = clip_score(safe_divide(total_foreign, total_all) * 100)

    # ③ 공급업체 집중도
    # 입찰참여업체정보가 있으면 공급업체 다양성지수 우선 사용
    if "공급업체다양성지수" in bidders:
        supplier_diversity = bidders.get("공급업체다양성지수", 50.0)
        집중도 = clip_score(100 - supplier_diversity)
    else:
        supplier_diversity = None
        company_counts = domestic.get("업체별건수", pd.Series(dtype=float))
        if len(company_counts) > 0:
            top5_share = safe_divide(company_counts.head(5).sum(), total_domestic) * 100
            집중도 = clip_score(top5_share)
        else:
            집중도 = 50.0

    # ④ 수의계약 의존도
    수의건수 = domestic.get("수의계약건수", 0) or 0
    수의의존도 = clip_score(safe_divide(수의건수, total_domestic) * 100)

    # ⑤ 전략물자 관련 계약 비율
    전략품목수 = strategic.get("전략물자품목수", 0) or 0
    전략비율 = clip_score(safe_divide(전략품목수, total_domestic) * 1000)

    score = (
        WEIGHTS_MATERIAL["국내조달_계약_감소율"] * 국내감소율지수 +
        WEIGHTS_MATERIAL["국외조달_의존도"] * 국외의존도 +
        WEIGHTS_MATERIAL["공급업체_집중도"] * 집중도 +
        WEIGHTS_MATERIAL["수의계약_의존도"] * 수의의존도 +
        WEIGHTS_MATERIAL["전략물자_관련_계약_비율"] * 전략비율
    )

    components = {
        "국내조달_감소율지수": round(국내감소율지수, 2),
        "국외조달_의존도": round(국외의존도, 2),
        "공급업체_집중도": round(집중도, 2),
        "공급업체_다양성지수": round(supplier_diversity, 2) if supplier_diversity is not None else None,
        "입찰_총업체수": bidders.get("총입찰업체수", 0),
        "입찰_고유업체수": bidders.get("고유업체수", 0),
        "입찰_HHI": bidders.get("허핀달지수", None),
        "수의계약_의존도": round(수의의존도, 2),
        "전략물자_비율_지수": round(전략비율, 2),
    }

    return clip_score(score), components, warnings


# ─────────────────────────────────────────────
# 통합 Risk Score
# ─────────────────────────────────────────────

def calc_integrated_risk(
    manpower_df: pd.DataFrame,
    disease_dc: float,
    material_score: float,
    jibang_dc_df: pd.DataFrame = None,
) -> tuple:
    """
    지방청별 통합 Risk Score 계산.
    """
    warnings = validate_weights(WEIGHTS_INTEGRATED, "통합 Risk")

    if manpower_df is None or manpower_df.empty:
        return pd.DataFrame(), ["인력 Risk 데이터가 없습니다."]

    result = manpower_df.copy()

    if jibang_dc_df is not None and not jibang_dc_df.empty:
        dc_lookup = jibang_dc_df[["지방청", "감염병DC"]].copy().reset_index(drop=True)
        result = result.merge(dc_lookup, on="지방청", how="left")

        unmatched = result[result["감염병DC"].isna()]["지방청"].tolist()
        if unmatched:
            warnings.append(
                f"지방청-DC 매핑 실패. 전국 대표값 적용: {unmatched}"
            )

        result["감염병DC"] = result["감염병DC"].fillna(round(disease_dc, 2))
    else:
        result["감염병DC"] = round(disease_dc, 2)

    result["물자Risk"] = round(material_score, 2)

    result["통합Risk"] = result.apply(
        lambda r: clip_score(
            WEIGHTS_INTEGRATED["인력"] * r["인력Risk"] +
            WEIGHTS_INTEGRATED["감염병"] * r["감염병DC"] +
            WEIGHTS_INTEGRATED["물자"] * material_score
        ),
        axis=1
    ).round(2)

    result["위험등급"] = result["통합Risk"].apply(grade)

    return result, warnings


# ─────────────────────────────────────────────
# 시뮬레이션 데이터 생성
# ─────────────────────────────────────────────

DEMO_REGIONS = [
    "서울", "부산울산", "대구경북", "경인", "광주전남",
    "대전충남", "강원", "충북", "전북", "경남",
    "제주", "인천", "경기북부",
]


def generate_simulation_data(seed: int = 42) -> pd.DataFrame:
    """
    시뮬레이션 데이터 생성.
    실제 데이터가 없을 때 데모용으로 사용.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for region in DEMO_REGIONS:
        mp_score = float(rng.uniform(10, 80))
        dc_score = float(rng.uniform(15, 70))
        mat_score = float(rng.uniform(20, 60))

        integrated = clip_score(
            WEIGHTS_INTEGRATED["인력"] * mp_score +
            WEIGHTS_INTEGRATED["감염병"] * dc_score +
            WEIGHTS_INTEGRATED["물자"] * mat_score
        )

        rows.append({
            "지방청": region,
            "인력Risk": round(mp_score, 2),
            "감염병DC": round(dc_score, 2),
            "물자Risk": round(mat_score, 2),
            "통합Risk": round(integrated, 2),
            "위험등급": grade(integrated),
        })

    return pd.DataFrame(rows)
