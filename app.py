# app.py
"""
TriGuard AI - 지역별 병력운용 Risk Score 분석 및 조기경보 시스템
2026 병무청·방위사업청·질병관리청 합동 공공데이터·AI 활용 경진대회 출품작
"""

import os
import streamlit as st
import pandas as pd
import traceback

from modules.preprocess import (
    load_csv,
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
# 데이터 파일 경로 매핑
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATA_FILES = {
    "exam":       "병무청_병역판정검사 현황_20251231.csv",
    "enlist":     "병무청_현역병 지방청별 입영현황_20241231.csv",
    "exempt":     "병무청_병역면제자 관리현황_20241231.csv",
    "regional":   "질병관리청_지역별 감염병 발생현황질병별 연도별 월별 기간별통계_지역별.csv",
    "national":   "질병관리청_질병별 감염병 발생현황 연도별 월별 기간별 통계.csv",
    "flu":        "질병관리청_법정감염병 표본감시_통계_인플루엔자.csv",
    "ari":        "질병관리청_법정감염병 표본감시_통계_급성호흡기감염증.csv",
    "dapa_dom":   "방위사업청_국내조달 계약정보_20251231.csv",
    "dapa_for":   "방위사업청_국외조달 계약정보_20251231.csv",
    "strategic":  "무역안보관리원_전략물자 품목키워드 및 개정정보_20260522.csv",
    "population": "행정안전부_지역별 연령별 주민등록 인구현황_월간.csv",
}


def get_data_path(key: str) -> str | None:
    """data/ 폴더에 해당 파일이 있으면 경로 반환, 없으면 None."""
    filename = DATA_FILES.get(key)
    if not filename:
        return None
    path = os.path.join(DATA_DIR, filename)
    return path if os.path.exists(path) else None


def load_auto(key: str):
    """data/ 폴더에서 자동 로드. 없으면 None 반환."""
    path = get_data_path(key)
    if path:
        try:
            return load_csv(path)
        except Exception:
            return None
    return None


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
# 자동 로드 여부 확인
# ─────────────────────────────────────────────

data_dir_exists = os.path.isdir(DATA_DIR)
auto_files_found = {k: get_data_path(k) for k in DATA_FILES}
auto_count = sum(1 for v in auto_files_found.values() if v)

# ─────────────────────────────────────────────
# 사이드바: 모드 선택 & 수동 업로드
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("실행 설정")
    mode = st.radio(
        "실행 모드",
        ["실제 데이터 모드", "시뮬레이션 모드"],
        index=0,
    )

    if mode == "실제 데이터 모드":
        if auto_count > 0:
            st.success(f"✅ data/ 폴더에서 {auto_count}개 파일 자동 감지")
            use_auto = st.toggle("자동 로드 사용", value=True)
        else:
            st.info("data/ 폴더에 CSV가 없습니다. 직접 업로드하세요.")
            use_auto = False

        if not use_auto or auto_count == 0:
            st.subheader("병무청")
            f_exam   = st.file_uploader("병역판정검사_현황.csv",   type="csv", key="exam")
            f_enlist = st.file_uploader("현역병_입영현황.csv",     type="csv", key="enlist")
            f_exempt = st.file_uploader("병역면제자_관리현황.csv", type="csv", key="exempt")

            st.subheader("질병관리청")
            f_regional = st.file_uploader("지역별_감염병_발생현황.csv",  type="csv", key="regional")
            f_national = st.file_uploader("질병별_감염병_발생현황.csv",  type="csv", key="national")
            f_flu      = st.file_uploader("인플루엔자_표본감시.csv",     type="csv", key="flu")
            f_ari      = st.file_uploader("급성호흡기감염증.csv",        type="csv", key="ari")

            st.subheader("방위사업청")
            f_dapa_dom  = st.file_uploader("국내조달_계약정보.csv",    type="csv", key="dapa_dom")
            f_dapa_for  = st.file_uploader("국외조달_계약정보.csv",    type="csv", key="dapa_for")
            f_strategic = st.file_uploader("전략물자_품목키워드.csv",  type="csv", key="strategic")
        else:
            # 자동 로드 모드에서는 업로드 위젯 숨김
            f_exam = f_enlist = f_exempt = None
            f_regional = f_national = f_flu = f_ari = None
            f_dapa_dom = f_dapa_for = f_strategic = None

        st.divider()
        st.caption("파일 미업로드 시 시뮬레이션 데이터 사용")

# ─────────────────────────────────────────────
# 가중치 검증
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
# 실제 데이터 로드
# ─────────────────────────────────────────────

def _load_raw(key, uploaded_file):
    """자동 로드 우선, 없으면 업로드 파일 사용."""
    if use_auto and auto_files_found.get(key):
        return load_csv(auto_files_found[key])
    if uploaded_file:
        return load_csv_from_upload(uploaded_file)
    return None


# 파일이 하나도 없으면 안내
if not use_auto:
    uploaded_any = any([f_exam, f_enlist, f_exempt, f_regional, f_national,
                        f_flu, f_ari, f_dapa_dom, f_dapa_for, f_strategic])
    if not uploaded_any:
        st.info("👈 사이드바에서 CSV 파일을 업로드하거나 시뮬레이션 모드를 선택하세요.")
        st.stop()

# ─────────────────────────────────────────────
# 데이터 파싱
# ─────────────────────────────────────────────

all_warnings = []

# ── 병무청 ──
exam_df = enlist_df = exempt_df = None

raw = _load_raw("exam", f_exam)
if raw is not None:
    try:
        exam_df = parse_byungmu_exam(raw)
        st.success(f"병역판정검사 로드 완료: {len(exam_df)}행")
    except KeyError as e:
        st.error(f"병역판정검사: 필수 컬럼 누락 ({e})")
    except Exception as e:
        st.error(f"병역판정검사 로드 오류: {e}")

raw = _load_raw("enlist", f_enlist)
if raw is not None:
    try:
        enlist_df = parse_byungmu_enlist(raw)
        st.success(f"입영현황 로드 완료: {len(enlist_df)}행")
    except Exception as e:
        st.error(f"입영현황 로드 오류: {e}")

raw = _load_raw("exempt", f_exempt)
if raw is not None:
    try:
        exempt_df = parse_byungmu_exempt(raw)
        st.success(f"병역면제 로드 완료: {len(exempt_df)}행")
    except Exception as e:
        st.error(f"병역면제 로드 오류: {e}")

# ── 질병관리청 ──
regional_inc_df = jibang_disease_df = None
national_weighted = 0.0
flu_df = ari_series = None

raw = _load_raw("regional", f_regional)
if raw is not None:
    try:
        regional_inc_df = parse_infectious_disease_regional(raw)
        st.success(f"지역별 감염병 로드 완료: {len(regional_inc_df)}행")
        jibang_disease_df = aggregate_disease_by_jibang(regional_inc_df)
    except Exception as e:
        st.error(f"지역별 감염병 로드 오류: {e}")

raw = _load_raw("national", f_national)
if raw is not None:
    try:
        national_weighted = parse_infectious_disease_national(raw)
        st.success(f"질병별 감염병 로드 완료 (가중합: {national_weighted:.1f})")
    except Exception as e:
        st.error(f"질병별 감염병 로드 오류: {e}")

raw = _load_raw("flu", f_flu)
if raw is not None:
    try:
        flu_df = parse_influenza(raw)
        st.success(f"인플루엔자 로드 완료: {len(flu_df)}절기")
    except Exception as e:
        st.error(f"인플루엔자 로드 오류: {e}")

raw = _load_raw("ari", f_ari)
if raw is not None:
    try:
        ari_series = parse_ari(raw)
        st.success(f"급성호흡기 로드 완료: {len(ari_series)}년")
    except Exception as e:
        st.error(f"급성호흡기 로드 오류: {e}")

# ── 방위사업청 ──
domestic_info = {"총건수": 1, "총금액": 0, "수의계약건수": 0, "수의계약금액": 0, "업체별건수": pd.Series(dtype=float)}
foreign_info  = {"국외총건수": 0}
strategic_info = {"전략물자품목수": 0}

raw = _load_raw("dapa_dom", f_dapa_dom)
if raw is not None:
    try:
        domestic_info = parse_dapa_domestic(raw)
        st.success(f"국내조달 로드 완료: {domestic_info['총건수']}건")
    except KeyError as e:
        st.error(f"국내조달: 필수 컬럼 누락 ({e})")
    except Exception as e:
        st.error(f"국내조달 로드 오류: {e}")

raw = _load_raw("dapa_for", f_dapa_for)
if raw is not None:
    try:
        foreign_info = parse_dapa_foreign(raw)
        st.success(f"국외조달 로드 완료: {foreign_info['국외총건수']}건")
    except Exception as e:
        st.error(f"국외조달 로드 오류: {e}")

raw = _load_raw("strategic", f_strategic)
if raw is not None:
    try:
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
        dc_score, regional_scored, jibang_dc_df, dc_components = 30.0, None, None, {}

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
    if not manpower_df.empty and "인력Risk" in manpower_df.columns:
        st.markdown("##### 바 차트 (인력Risk)")
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
