# app.py
"""
TriGuard AI - 지역별 병력운용 Risk Score 분석 및 조기경보 시스템
2026 병무청·방위사업청·질병관리청 합동 공공데이터·AI 활용 경진대회 출품작
"""

import streamlit as st
import pandas as pd
import traceback

from modules.preprocess import (
    load_csv_from_upload,
    parse_byungmu_exam,
    parse_byungmu_enlist,
    parse_byungmu_exempt,
    parse_influenza,
    parse_ari,
    parse_infectious_disease_regional,
    parse_infectious_disease_national,
    parse_dapa_domestic,
    parse_dapa_foreign,
    parse_strategic_goods,
    aggregate_disease_by_jibang,
)
from modules.risk_engine import (
    calc_manpower_risk,
    calc_disease_dc,
    calc_material_risk,
    calc_integrated_risk,
    generate_simulation_data,
    validate_weights,
    WEIGHTS_MANPOWER,
    WEIGHTS_DISEASE,
    WEIGHTS_MATERIAL,
    WEIGHTS_INTEGRATED,
)
from modules.visualize import (
    render_kpi_cards,
    render_result_table,
    render_bar_chart,
    render_manpower_detail,
    render_disease_components,
    render_material_components,
    render_response_guide,
    render_warnings,
)

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="TriGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ TriGuard AI")
st.caption("감염병·병역자원 데이터 기반 지역별 병력운용 Risk Score 분석 및 조기경보 시스템")
st.divider()

# ─────────────────────────────────────────────
# 사이드바: 파일 업로드 & 모드 선택
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("데이터 업로드")
    mode = st.radio(
        "실행 모드",
        ["실제 데이터 모드", "시뮬레이션 모드"],
        index=0,
    )

    st.subheader("병무청")
    f_exam   = st.file_uploader("병역판정검사_현황.csv",    type="csv", key="exam")
    f_enlist = st.file_uploader("현역병_입영현황.csv",      type="csv", key="enlist")
    f_exempt = st.file_uploader("병역면제자_관리현황.csv",  type="csv", key="exempt")

    st.subheader("질병관리청")
    f_regional = st.file_uploader("지역별_감염병_발생현황.csv",  type="csv", key="regional")
    f_national = st.file_uploader("질병별_감염병_발생현황.csv",  type="csv", key="national")
    f_flu      = st.file_uploader("인플루엔자_표본감시.csv",     type="csv", key="flu")
    f_ari      = st.file_uploader("급성호흡기감염증.csv",        type="csv", key="ari")

    st.subheader("방위사업청")
    f_dapa_dom = st.file_uploader("국내조달_계약정보.csv",    type="csv", key="dapa_dom")
    f_dapa_for = st.file_uploader("국외조달_계약정보.csv",    type="csv", key="dapa_for")
    f_strategic = st.file_uploader("전략물자_품목키워드.csv", type="csv", key="strategic")

    st.divider()
    st.caption("파일 미업로드 시 시뮬레이션 데이터 사용")

# ─────────────────────────────────────────────
# 가중치 검증 (앱 시작 시 1회)
# ─────────────────────────────────────────────

for name, weights in [
    ("인력 Risk", WEIGHTS_MANPOWER),
    ("감염병 DC", WEIGHTS_DISEASE),
    ("물자 Risk", WEIGHTS_MATERIAL),
    ("통합 Risk", WEIGHTS_INTEGRATED),
]:
    for w in validate_weights(weights, name):
        st.warning(w)

# ─────────────────────────────────────────────
# 시뮬레이션 모드
# ─────────────────────────────────────────────

if mode == "시뮬레이션 모드":
    st.info("시뮬레이션 모드: 랜덤 생성 데이터를 사용합니다.")
    seed = st.slider("시뮬레이션 시드", 0, 100, 42)
    result_df = generate_simulation_data(seed=seed)

    st.subheader("📊 통합 Risk Score 현황")
    render_kpi_cards(result_df)
    st.markdown("#### 지방청별 통합 Risk Score")
    render_result_table(result_df)
    st.markdown("#### 바 차트")
    render_bar_chart(result_df, "통합Risk", "지방청별 통합 Risk Score")
    st.markdown("#### 🚨 대응 가이드")
    render_response_guide(result_df)
    st.stop()

# ─────────────────────────────────────────────
# 실제 데이터 모드
# ─────────────────────────────────────────────

# 파일이 하나도 없으면 안내
uploaded_any = any([f_exam, f_enlist, f_exempt, f_regional, f_national, f_flu, f_ari, f_dapa_dom, f_dapa_for, f_strategic])
if not uploaded_any:
    st.info("👈 사이드바에서 CSV 파일을 업로드하거나 시뮬레이션 모드를 선택하세요.")
    st.stop()

# ─────────────────────────────────────────────
# 데이터 파싱
# ─────────────────────────────────────────────

all_warnings = []

# ── 병무청 ──
exam_df    = None
enlist_df  = None
exempt_df  = None

if f_exam:
    try:
        raw = load_csv_from_upload(f_exam)
        exam_df = parse_byungmu_exam(raw)
        st.success(f"병역판정검사 로드 완료: {len(exam_df)}행")
    except KeyError as e:
        st.error(f"병역판정검사: 필수 컬럼 누락 ({e})")
    except Exception as e:
        st.error(f"병역판정검사 로드 오류: {e}")

if f_enlist:
    try:
        raw = load_csv_from_upload(f_enlist)
        enlist_df = parse_byungmu_enlist(raw)
        st.success(f"입영현황 로드 완료: {len(enlist_df)}행")
    except KeyError as e:
        st.error(f"입영현황: 필수 컬럼 누락 ({e})")
    except Exception as e:
        st.error(f"입영현황 로드 오류: {e}")

if f_exempt:
    try:
        raw = load_csv_from_upload(f_exempt)
        exempt_df = parse_byungmu_exempt(raw)
        st.success(f"병역면제 로드 완료: {len(exempt_df)}행")
    except Exception as e:
        st.error(f"병역면제 로드 오류: {e}")

# ── 질병관리청 ──
regional_inc_df   = None
jibang_disease_df = None
national_weighted = 0.0
flu_df     = None
ari_series = None

if f_regional:
    try:
        raw = load_csv_from_upload(f_regional)
        regional_inc_df = parse_infectious_disease_regional(raw)
        st.success(f"지역별 감염병 로드 완료: {len(regional_inc_df)}행")
        jibang_disease_df = aggregate_disease_by_jibang(regional_inc_df)
    except Exception as e:
        st.error(f"지역별 감염병 로드 오류: {e}")

if f_national:
    try:
        raw = load_csv_from_upload(f_national)
        national_weighted = parse_infectious_disease_national(raw)
        st.success(f"질병별 감염병 로드 완료 (가중합: {national_weighted:.1f})")
    except Exception as e:
        st.error(f"질병별 감염병 로드 오류: {e}")

if f_flu:
    try:
        raw = load_csv_from_upload(f_flu)
        flu_df = parse_influenza(raw)
        st.success(f"인플루엔자 로드 완료: {len(flu_df)}절기")
    except Exception as e:
        st.error(f"인플루엔자 로드 오류: {e}")

if f_ari:
    try:
        raw = load_csv_from_upload(f_ari)
        ari_series = parse_ari(raw)
        st.success(f"급성호흡기 로드 완료: {len(ari_series)}년")
    except Exception as e:
        st.error(f"급성호흡기 로드 오류: {e}")

# ── 방위사업청 ──
domestic_info = {"총건수": 1, "총금액": 0, "수의계약건수": 0, "수의계약금액": 0, "업체별건수": pd.Series(dtype=float)}
foreign_info  = {"국외총건수": 0}
strategic_info = {"전략물자품목수": 0}

if f_dapa_dom:
    try:
        raw = load_csv_from_upload(f_dapa_dom)
        domestic_info = parse_dapa_domestic(raw)
        st.success(f"국내조달 로드 완료: {domestic_info['총건수']}건")
    except KeyError as e:
        st.error(f"국내조달: 필수 컬럼 누락 ({e})")
    except Exception as e:
        st.error(f"국내조달 로드 오류: {e}")

if f_dapa_for:
    try:
        raw = load_csv_from_upload(f_dapa_for)
        foreign_info = parse_dapa_foreign(raw)
        st.success(f"국외조달 로드 완료: {foreign_info['국외총건수']}건")
    except Exception as e:
        st.error(f"국외조달 로드 오류: {e}")

if f_strategic:
    try:
        raw = load_csv_from_upload(f_strategic)
        strategic_info = parse_strategic_goods(raw)
        st.success(f"전략물자 로드 완료: {strategic_info['전략물자품목수']}품목")
    except Exception as e:
        st.error(f"전략물자 로드 오류: {e}")

# ─────────────────────────────────────────────
# Risk Score 계산
# ─────────────────────────────────────────────

if exam_df is None:
    st.warning("병역판정검사 데이터가 없어 인력 Risk 계산이 불가합니다. 시뮬레이션 모드를 사용하세요.")
    st.stop()

st.divider()
st.subheader("Risk Score 계산 중...")

with st.spinner("인력 Risk 계산 중..."):
    try:
        manpower_df, mp_warnings = calc_manpower_risk(exam_df, enlist_df, exempt_df)
        all_warnings.extend(mp_warnings)
    except Exception as e:
        st.error(f"인력 Risk 계산 실패: {e}\n{traceback.format_exc()}")
        st.stop()

with st.spinner("감염병 DC 계산 중..."):
    try:
        dc_score, regional_scored, jibang_dc_df, dc_components, dc_warnings = calc_disease_dc(
            regional_inc_df, national_weighted, flu_df, ari_series, jibang_disease_df
        )
        all_warnings.extend(dc_warnings)
    except Exception as e:
        st.error(f"감염병 DC 계산 실패: {e}")
        dc_score, regional_scored, jibang_dc_df, dc_components, dc_warnings = 30.0, None, None, {}, []

with st.spinner("물자 Risk 계산 중..."):
    try:
        mat_score, mat_components, mat_warnings = calc_material_risk(
            domestic_info, foreign_info, strategic_info
        )
        all_warnings.extend(mat_warnings)
    except Exception as e:
        st.error(f"물자 Risk 계산 실패: {e}")
        mat_score, mat_components = 30.0, {}

with st.spinner("통합 Risk Score 계산 중..."):
    try:
        result_df, int_warnings = calc_integrated_risk(manpower_df, dc_score, mat_score, jibang_dc_df)
        all_warnings.extend(int_warnings)
    except Exception as e:
        st.error(f"통합 Risk 계산 실패: {e}")
        st.stop()

render_warnings(all_warnings)

# ─────────────────────────────────────────────
# 대시보드 출력
# ─────────────────────────────────────────────

st.divider()
st.subheader("📊 통합 Risk Score 현황")
render_kpi_cards(result_df)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "통합 결과", "인력 세부", "감염병 DC", "물자 Risk", "대응 가이드"
])

with tab1:
    st.markdown("#### 지방청별 통합 Risk Score")
    render_result_table(result_df)
    st.markdown("#### 바 차트")
    render_bar_chart(result_df, "통합Risk", "지방청별 통합 Risk Score")
    st.markdown("---")
    st.download_button(
        "📥 결과 CSV 다운로드",
        data=result_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="triguard_risk_result.csv",
        mime="text/csv",
    )

with tab2:
    st.markdown("#### 인력 Risk 세부 지표")
    render_manpower_detail(manpower_df)
    st.markdown("##### 바 차트 (인력Risk)")
    if not manpower_df.empty and "인력Risk" in manpower_df.columns:
        render_bar_chart(result_df, "인력Risk", "지방청별 인력 Risk Score")

with tab3:
    st.markdown("#### 감염병 Disruption Coefficient")
    render_disease_components(dc_components, dc_score)
    if regional_scored is not None and not regional_scored.empty and "발생률지수" in regional_scored.columns:
        st.markdown("##### 시도별 감염병 발생률 지수")
        st.dataframe(
            regional_scored[["시도", "총발생률", "발생률지수"]]
            .sort_values("발생률지수", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

with tab4:
    st.markdown("#### 물자 Risk (전국 단위)")
    st.caption("※ 방위사업청 데이터는 지역 단위 없음 → 전국 단일값을 모든 지방청에 동일 적용")
    render_material_components(mat_components, mat_score)

with tab5:
    st.markdown("#### 위험·주의 권역 대응 가이드")
    render_response_guide(result_df)

# ─────────────────────────────────────────────
# 푸터
# ─────────────────────────────────────────────

st.divider()
st.caption(
    "TriGuard AI | 2026 병무청·방위사업청·질병관리청 합동 공공데이터·AI 활용 경진대회 출품작 | "
    "데이터 출처: 병무청, 질병관리청, 방위사업청, 행정안전부, 무역안보관리원"
)
