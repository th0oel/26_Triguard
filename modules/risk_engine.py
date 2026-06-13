# modules/risk_engine.py
"""
TriGuard AI - Risk Score 계산 엔진
인력 / 감염병 DC / 물자 / 통합 Risk Score
"""

import numpy as np
import pandas as pd
from modules.preprocess import safe_divide

# ─────────────────────────────────────────────
# 가중치 상수 (수정 시 여기만 변경)
# ─────────────────────────────────────────────

WEIGHTS_MANPOWER = {
    "입영인원_감소율":      0.30,
    "병역판정검사_감소율":  0.25,
    "병역면제율":           0.20,
    "재신체검사율":         0.10,
    "입영_차질률":          0.15,
}

WEIGHTS_DISEASE = {
    "지역별_발생률":        0.35,
    "질병_등급_가중합":     0.30,
    "인플루엔자_유행_강도": 0.20,
    "급성호흡기_트렌드":    0.15,
}

WEIGHTS_MATERIAL = {
    "국내조달_계약_감소율": 0.25,
    "국외조달_의존도":      0.25,
    "공급업체_집중도":      0.20,
    "수의계약_의존도":      0.15,
    "전략물자_관련_계약_비율": 0.15,
}

WEIGHTS_INTEGRATED = {
    "인력":   0.40,
    "감염병": 0.40,
    "물자":   0.20,
}

# 위험 등급 임계값
GRADE_DANGER  = 60.0   # 이상 → 위험
GRADE_CAUTION = 35.0   # 이상 → 주의
# 미만 → 정상


def validate_weights(weights: dict, name: str) -> list:
    """가중치 합이 1.0인지 검증. 오차 허용: ±0.01. 경고 메시지 리스트 반환."""
    total = sum(weights.values())
    warnings = []
    if abs(total - 1.0) > 0.01:
        warnings.append(
            f"⚠️ [{name}] 가중치 합계가 {total:.3f}입니다 (기대값: 1.000). "
            f"결과가 부정확할 수 있습니다."
        )
    return warnings


def clip_score(score: float) -> float:
    """Risk Score를 0~100 범위로 clip."""
    return float(np.clip(score, 0.0, 100.0))


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

def calc_manpower_risk(exam_df: pd.DataFrame, enlist_df: pd.DataFrame, exempt_df: pd.DataFrame) -> pd.DataFrame:
    """
    지방청별 인력 Risk Score 계산.

    exam_df: 병역판정검사 (연도, 지방청, 처분인원, 현역, 병역면제, 재신체검사)
    enlist_df: 입영현황 (지방청, 입영실통지, 입영, 행방불명, 기피)
    exempt_df: 병역면제 (지방청, 계)
    """
    warnings = validate_weights(WEIGHTS_MANPOWER, "인력 Risk")
    results = []

    # 연도 범위: 최신 연도 vs 가장 오래된 연도 (최대 7년) 비교
    # → 단순 전년 대비가 아닌 중장기 감소 추세 반영
    years = sorted(exam_df["연도"].unique())
    latest_year = years[-1]
    # 기준 연도: 최신 연도 기준 최대 6년 전 (데이터가 없으면 가장 오래된 연도)
    target_base_year = latest_year - 6
    base_candidates = [y for y in years if y <= target_base_year]
    prev_year = base_candidates[-1] if base_candidates else years[0]
    year_gap = max(latest_year - prev_year, 1)  # 연수 차이 (0 방지)

    latest_exam = exam_df[exam_df["연도"] == latest_year]
    prev_exam   = exam_df[exam_df["연도"] == prev_year]

    regions = latest_exam["지방청"].unique()

    for region in regions:
        row_now  = latest_exam[latest_exam["지방청"] == region]
        row_prev = prev_exam[prev_exam["지방청"] == region]
        row_enl  = enlist_df[enlist_df["지방청"] == region] if enlist_df is not None else pd.DataFrame()
        row_ex   = exempt_df[exempt_df["지방청"] == region] if exempt_df is not None else pd.DataFrame()

        if row_now.empty:
            continue

        # ① 입영인원 감소율 (현역 기준, 기준연도 대비 최신연도)
        exam_now  = float(row_now["현역"].values[0]) if "현역" in row_now.columns else 0
        exam_prev = float(row_prev["현역"].values[0]) if (not row_prev.empty and "현역" in row_prev.columns) else exam_now
        # 연간 평균 감소율로 정규화 (연수 차이로 나눔 → 연평균 % 감소)
        raw_감소율 = safe_divide(exam_prev - exam_now, exam_prev + 1e-9) * 100
        입영_감소율 = clip_score(raw_감소율 * (6 / year_gap))  # 6년 기준 정규화

        # ② 병역판정검사 감소율 (기준연도 대비)
        proc_now  = float(row_now["처분인원"].values[0]) if "처분인원" in row_now.columns else 0
        proc_prev = float(row_prev["처분인원"].values[0]) if (not row_prev.empty and "처분인원" in row_prev.columns) else proc_now
        raw_검사감소 = safe_divide(proc_prev - proc_now, proc_prev + 1e-9) * 100
        검사_감소율 = clip_score(raw_검사감소 * (6 / year_gap))  # 6년 기준 정규화

        # ③ 병역면제율 (면제자 / 처분인원)
        exempt_total = float(row_ex["계"].values[0]) if (not row_ex.empty and "계" in row_ex.columns) else 0
        exam_면제 = float(row_now["병역면제"].values[0]) if "병역면제" in row_now.columns else 0
        면제율 = clip_score(safe_divide(exam_면제, proc_now) * 100)

        # ④ 재신체검사율
        재검 = float(row_now["재신체검사"].values[0]) if "재신체검사" in row_now.columns else 0
        재검율 = clip_score(safe_divide(재검, proc_now) * 100)

        # ⑤ 입영 차질률 ((행방불명+기피) / 입영실통지)
        miss  = float(row_enl["행방불명"].values[0]) if (not row_enl.empty and "행방불명" in row_enl.columns) else 0
        evas  = float(row_enl["기피"].values[0])    if (not row_enl.empty and "기피" in row_enl.columns) else 0
        notif = float(row_enl["입영실통지"].values[0]) if (not row_enl.empty and "입영실통지" in row_enl.columns) else 1
        차질률 = clip_score(safe_divide(miss + evas, notif) * 100)

        score = (
            WEIGHTS_MANPOWER["입영인원_감소율"]     * 입영_감소율 +
            WEIGHTS_MANPOWER["병역판정검사_감소율"] * 검사_감소율 +
            WEIGHTS_MANPOWER["병역면제율"]          * 면제율 +
            WEIGHTS_MANPOWER["재신체검사율"]        * 재검율 +
            WEIGHTS_MANPOWER["입영_차질률"]         * 차질률
        )

        results.append({
            "지방청": region,
            "입영인원_감소율":     round(입영_감소율, 2),
            "병역판정검사_감소율": round(검사_감소율, 2),
            "병역면제율":          round(면제율, 2),
            "재신체검사율":        round(재검율, 2),
            "입영_차질률":         round(차질률, 2),
            "인력Risk": round(clip_score(score), 2),
        })

    result_df = pd.DataFrame(results)
    return result_df, warnings


# ─────────────────────────────────────────────
# 감염병 Disruption Coefficient (DC)
# ─────────────────────────────────────────────

def calc_disease_dc(
    regional_df: pd.DataFrame,              # 시도별 발생현황 (시도, 총발생률) — 표시용
    national_weighted: float,               # 질병별 등급 가중합
    influenza_df: pd.DataFrame,             # 인플루엔자 (절기, 최대분율, 평균분율)
    ari_series: pd.Series,                  # 급성호흡기 (연도 index, 총합계)
    jibang_disease_df: pd.DataFrame = None, # 지방청별 총발생률 (aggregate_disease_by_jibang 결과)
) -> tuple:
    """
    감염병 Disruption Coefficient 계산.
    반환: (dc_score_national, regional_df_scored, jibang_dc_df, components, warnings)
    - dc_score_national: 전국 대표값 (fallback용)
    - jibang_dc_df: 지방청별 감염병DC DataFrame [지방청, 발생률지수, 감염병DC]
    """
    warnings = validate_weights(WEIGHTS_DISEASE, "감염병 DC")

    # ── 공통 국가 수준 지표 ──────────────────────
    # ② 질병 등급 가중합 → 100점 스케일
    # 실데이터 기준: 2358 → 약 45점, 5000 → 약 60점, 10000 → 100점
    grade_score = clip_score(np.log1p(national_weighted) / np.log1p(10000) * 100) if national_weighted > 0 else 0.0

    # ③ 인플루엔자 유행강도
    # 전체 절기 중 최대분율 대비 최신 절기 최대분율 비율로 정규화
    if influenza_df is not None and not influenza_df.empty:
        all_max = influenza_df["최대분율"].dropna()
        latest_max = float(all_max.iloc[-1])
        historical_max = float(all_max.max()) if len(all_max) > 0 else 100
        # 역사적 최대 대비 현재 비율 → 0~60점 스케일
        flu_score = clip_score(safe_divide(latest_max, historical_max) * 60)
    else:
        flu_score = 30.0

    # ④ 급성호흡기 트렌드 (최근 2년 대비 증감)
    if ari_series is not None and len(ari_series) >= 2:
        vals = ari_series.sort_index().values
        # 전체 범위 대비 현재 수준으로 정규화
        current = vals[-1]
        historical_max = max(vals)
        ari_score = clip_score(safe_divide(current, historical_max) * 60)
    else:
        ari_score = 30.0

    def _dc_from_regional_index(regional_idx: float) -> float:
        return clip_score(
            WEIGHTS_DISEASE["지역별_발생률"]        * regional_idx +
            WEIGHTS_DISEASE["질병_등급_가중합"]      * grade_score +
            WEIGHTS_DISEASE["인플루엔자_유행_강도"] * flu_score +
            WEIGHTS_DISEASE["급성호흡기_트렌드"]    * ari_score
        )

    # ── ① 지방청별 발생률 지수 및 DC ────────────
    jibang_dc_df = None
    if jibang_disease_df is not None and not jibang_disease_df.empty and "총발생률" in jibang_disease_df.columns:
        max_rate = jibang_disease_df["총발생률"].max()
        jd = jibang_disease_df.copy()
        # 절대값 기준 정규화: 1000 이하=0점, 1600 이상=100점
        RATE_MIN = 800.0   # 정상 기준
        RATE_MAX = 1600.0  # 위험 기준 (실데이터 최대치 기반)
        jd["발생률지수"] = jd["총발생률"].apply(
            lambda x: clip_score(safe_divide(x - RATE_MIN, RATE_MAX - RATE_MIN) * 100)
        )
        jd["감염병DC"] = jd["발생률지수"].apply(_dc_from_regional_index).round(2)
        jibang_dc_df = jd[["지방청", "발생률지수", "감염병DC"]]
        avg_regional = jd["발생률지수"].mean()
    elif regional_df is not None and not regional_df.empty and "총발생률" in regional_df.columns:
        # 지방청 매핑 없이 시도 평균으로 fallback
        max_rate = regional_df["총발생률"].max()
        regional_df = regional_df.copy()
        regional_df["발생률지수"] = regional_df["총발생률"].apply(
            lambda x: clip_score(safe_divide(x, max_rate) * 100)
        )
        avg_regional = regional_df["발생률지수"].mean()
    else:
        regional_df = pd.DataFrame()
        avg_regional = 50.0

    # ── 시도별 표시용 regional_df 정규화 ─────────
    if regional_df is not None and not regional_df.empty and "총발생률" in regional_df.columns and "발생률지수" not in regional_df.columns:
        max_rate = regional_df["총발생률"].max()
        regional_df = regional_df.copy()
        regional_df["발생률지수"] = regional_df["총발생률"].apply(
            lambda x: clip_score(safe_divide(x, max_rate) * 100)
        )

    # 전국 대표 DC (fallback/디스플레이용)
    dc_score_national = _dc_from_regional_index(avg_regional)

    components = {
        "지역별_발생률_지수(전국평균)": round(avg_regional, 2),
        "질병_등급_가중합_지수":        round(grade_score, 2),
        "인플루엔자_강도_지수":         round(flu_score, 2),
        "급성호흡기_트렌드_지수":       round(ari_score, 2),
    }

    return dc_score_national, regional_df, jibang_dc_df, components, warnings


# ─────────────────────────────────────────────
# 물자 Risk Score (전국 단위)
# ─────────────────────────────────────────────

def calc_material_risk(
    domestic: dict,     # parse_dapa_domestic 결과
    foreign: dict,      # parse_dapa_foreign 결과
    strategic: dict,    # parse_strategic_goods 결과
) -> tuple:
    """
    방위사업청 기반 전국 물자 Risk Score.
    지역 단위 없음 → 전국 단일값. 모든 지방청에 동일 적용.
    """
    warnings = validate_weights(WEIGHTS_MATERIAL, "물자 Risk")

    total_domestic = domestic.get("총건수", 1)
    total_foreign  = foreign.get("국외총건수", 0)
    total_all      = total_domestic + total_foreign

    # ① 국내조달 계약 감소율: 데이터 단년이므로 국외 대비 국내 비중으로 대리
    # (국내 비중이 낮을수록 위험 → 역수)
    내외비율 = safe_divide(total_domestic, total_all) * 100
    국내감소율지수 = clip_score(100 - 내외비율)  # 국내 비중 낮을수록 위험

    # ② 국외조달 의존도
    국외의존도 = clip_score(safe_divide(total_foreign, total_all) * 100)

    # ③ 공급업체 집중도 (상위 5개 업체 점유율)
    company_counts = domestic.get("업체별건수", pd.Series(dtype=float))
    if len(company_counts) > 0:
        top5_share = safe_divide(company_counts.head(5).sum(), total_domestic) * 100
        집중도 = clip_score(top5_share)
    else:
        집중도 = 50.0

    # ④ 수의계약 의존도
    수의건수 = domestic.get("수의계약건수", 0)
    수의의존도 = clip_score(safe_divide(수의건수, total_domestic) * 100)

    # ⑤ 전략물자 관련 계약 비율 (전략물자 품목 수 / 전체계약 × 스케일)
    전략품목수 = strategic.get("전략물자품목수", 0)
    전략비율 = clip_score(safe_divide(전략품목수, total_domestic) * 1000)  # 스케일 조정

    score = (
        WEIGHTS_MATERIAL["국내조달_계약_감소율"]   * 국내감소율지수 +
        WEIGHTS_MATERIAL["국외조달_의존도"]         * 국외의존도 +
        WEIGHTS_MATERIAL["공급업체_집중도"]         * 집중도 +
        WEIGHTS_MATERIAL["수의계약_의존도"]         * 수의의존도 +
        WEIGHTS_MATERIAL["전략물자_관련_계약_비율"] * 전략비율
    )

    components = {
        "국내조달_감소율지수":  round(국내감소율지수, 2),
        "국외조달_의존도":      round(국외의존도, 2),
        "공급업체_집중도":      round(집중도, 2),
        "수의계약_의존도":      round(수의의존도, 2),
        "전략물자_비율_지수":   round(전략비율, 2),
    }

    return clip_score(score), components, warnings


# ─────────────────────────────────────────────
# 통합 Risk Score
# ─────────────────────────────────────────────

def calc_integrated_risk(
    manpower_df: pd.DataFrame,              # 지방청별 인력Risk 포함
    disease_dc: float,                      # 전국 감염병 DC (지방청 매핑 없을 때 fallback)
    material_score: float,                  # 전국 물자 Risk
    jibang_dc_df: pd.DataFrame = None,      # 지방청별 감염병DC [지방청, 감염병DC]
) -> pd.DataFrame:
    """
    지방청별 통합 Risk Score 계산.
    jibang_dc_df가 있으면 지방청별 감염병DC를 각각 적용.
    매핑이 없는 지방청은 전국 대표값(disease_dc)으로 fallback.
    물자는 지역 데이터 없으므로 전국 단일값 동일 적용.
    """
    warnings = validate_weights(WEIGHTS_INTEGRATED, "통합 Risk")

    result = manpower_df.copy()

    if jibang_dc_df is not None and not jibang_dc_df.empty:
        dc_lookup = jibang_dc_df[["지방청", "감염병DC"]].copy().reset_index(drop=True)
        result = result.merge(dc_lookup, on="지방청", how="left")

        unmatched = result[result["감염병DC"].isna()]["지방청"].tolist()
        if unmatched:
            warnings.append(
                f"지방청-DC 매핑 실패 (전국 대표값 적용): {unmatched}. "
                f"jibang_dc_df 보유 지방청: {dc_lookup['지방청'].tolist()}"
            )
        result["감염병DC"] = result["감염병DC"].fillna(round(disease_dc, 2))
    else:
        result["감염병DC"] = round(disease_dc, 2)

    result["물자Risk"] = round(material_score, 2)

    result["통합Risk"] = result.apply(
        lambda r: clip_score(
            WEIGHTS_INTEGRATED["인력"]   * r["인력Risk"] +
            WEIGHTS_INTEGRATED["감염병"] * r["감염병DC"] +
            WEIGHTS_INTEGRATED["물자"]   * material_score
        ), axis=1
    ).round(2)

    result["위험등급"] = result["통합Risk"].apply(grade)
    return result, warnings


# ─────────────────────────────────────────────
# 시뮬레이션 데이터 생성 (실제 데이터 없을 때 데모용)
# ─────────────────────────────────────────────

DEMO_REGIONS = [
    "서울", "부산울산", "대구경북", "경인", "광주전남",
    "대전충남", "강원", "충북", "전북", "경남",
    "제주", "인천", "경기북부",
]


def generate_simulation_data(seed: int = 42) -> pd.DataFrame:
    """
    시뮬레이션 데이터 생성.
    가중치 공식을 그대로 적용해서 현실적 분포 생성.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for region in DEMO_REGIONS:
        mp_score = float(rng.uniform(10, 80))
        dc_score = float(rng.uniform(15, 70))
        mat_score = float(rng.uniform(20, 60))
        integrated = clip_score(
            WEIGHTS_INTEGRATED["인력"]   * mp_score +
            WEIGHTS_INTEGRATED["감염병"] * dc_score +
            WEIGHTS_INTEGRATED["물자"]   * mat_score
        )
        rows.append({
            "지방청":    region,
            "인력Risk":  round(mp_score, 2),
            "감염병DC":  round(dc_score, 2),
            "물자Risk":  round(mat_score, 2),
            "통합Risk":  round(integrated, 2),
            "위험등급":  grade(integrated),
        })

    return pd.DataFrame(rows)
