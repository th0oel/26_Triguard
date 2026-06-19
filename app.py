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
    parse_population,
    parse_dapa_bidders,
    aggregate_disease_by_jibang,
)
from modules.ml_engine import (
    train_risk_model,
    predict_risk,
    cross_agency_correlation,
    manpower_disease_trend,
    detect_anomaly_regions,
    forecast_manpower,
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
    render_map,
    render_manpower_detail,
    render_disease_components,
    render_material_components,
    render_response_guide,
    render_warnings,
    render_ml_prediction,
    render_feature_importance,
    render_cross_agency_scatter,
    render_manpower_trend,
    render_anomaly_table,
)

# ─────────────────────────────────────────────
# 데이터 파일 경로 매핑
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATA_FILES = {
    "exam":       "병무청_병역판정검사 현황_20251231.csv",
    "enlist":     "병무청_현역병 지방청별 입영현황_20241231.csv",
    "exempt":     "병무청_병역면제자 관리현황_20241231.csv",
    "regional":   "질병관리청_지역별 감염병 발생현황(질병별 연도별 월별 기간별)통계_지역별.csv",
    "national":   "질병관리청_질병별 감염병 발생현황 (연도별 월별 기간별) 통계.csv",
    "flu":        "질병관리청_법정감염병 표본감시_통계_인플루엔자.csv",
    "ari":        "질병관리청_법정감염병 표본감시_통계_급성호흡기감염증.csv",
    "dapa_dom":   "방위사업청_국내조달 계약정보_20251231.csv",
    "dapa_for":   "방위사업청_국외조달 계약정보_20251231.csv",
    "dapa_bidders": "방위사업청 국내조달 입찰참여업체정보_20251231.csv",
    "strategic":  "무역안보관리원_전략물자 품목키워드 및 개정정보_20260522.csv",
    "population": "행정안전부_지역별 연령별 주민등록 인구현황_월간.csv",
}


def get_data_path(key: str):
    filename = DATA_FILES.get(key)
    if not filename:
        return None
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    for base in [DATA_DIR, ROOT_DIR]:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    return None


def load_auto(key: str):
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

st.title("TriGuard AI")
st.caption("감염병·병역자원 데이터 기반 지역별 병력운용 Risk Score 분석 및 조기경보 시스템")
st.divider()

# ─────────────────────────────────────────────
# 자동 로드 여부 확인
# ─────────────────────────────────────────────

data_dir_exists = os.path.isdir(DATA_DIR)
auto_files_found = {k: get_data_path(k) for k in DATA_FILES}
auto_count = sum(1 for v in auto_files_found.values() if v)

# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("실행 설정")
    mode = st.radio("실행 모드", ["실제 데이터 모드", "시뮬레이션 모드"], index=0)

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
            f_dapa_dom     = st.file_uploader("국내조달_계약정보.csv",            type="csv", key="dapa_dom")
            f_dapa_for     = st.file_uploader("국외조달_계약정보.csv",            type="csv", key="dapa_for")
            f_dapa_bidders = st.file_uploader("국내조달_입찰참여업체정보.csv",    type="csv", key="dapa_bidders")
            f_strategic    = st.file_uploader("전략물자_품목키워드.csv",           type="csv", key="strategic")
            st.subheader("행정안전부")
            f_population   = st.file_uploader("행정안전부_연령별_주민등록_인구현황.csv", type="csv", key="population")
        else:
            f_exam = f_enlist = f_exempt = None
            f_regional = f_national = f_flu = f_ari = None
            f_dapa_dom = f_dapa_for = f_dapa_bidders = f_strategic = None
            f_population = None

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
    st.info("시뮬레이션 모드로 데이터를 생성합니다.")
    seed = st.slider("시뮬레이션 시드", 0, 100, 42)
    result_df = generate_simulation_data(seed=seed)
    st.subheader("통합 Risk Score 현황")
    render_kpi_cards(result_df)
    render_result_table(result_df)
    render_bar_chart(result_df, "통합Risk", "지방청별 통합 Risk Score")
    render_response_guide(result_df)
    st.stop()

# ─────────────────────────────────────────────
# 실제 데이터 로드
# ─────────────────────────────────────────────

def _load_raw(key, uploaded_file):
    if use_auto and auto_files_found.get(key):
        return load_csv(auto_files_found[key])
    if uploaded_file:
        return load_csv_from_upload(uploaded_file)
    return None

if not use_auto:
    uploaded_any = any([f_exam, f_enlist, f_exempt, f_regional, f_national,
                        f_flu, f_ari, f_dapa_dom, f_dapa_for, f_dapa_bidders, f_strategic,
                        f_population])
    if not uploaded_any:
        st.info("사이드바에서 CSV 파일을 업로드하거나 시뮬레이션 모드를 선택하세요.")
        st.stop()

# ─────────────────────────────────────────────
# 데이터 파싱
# ─────────────────────────────────────────────

all_warnings = []
exam_df = enlist_df = exempt_df = None

raw = _load_raw("exam", f_exam)
if raw is not None:
    try:
        exam_df = parse_byungmu_exam(raw)
    except Exception as e:
        st.error(f"병역판정검사 로드 오류: {e}")

raw = _load_raw("enlist", f_enlist)
if raw is not None:
    try:
        enlist_df = parse_byungmu_enlist(raw)
    except Exception as e:
        st.error(f"입영현황 로드 오류: {e}")

raw = _load_raw("exempt", f_exempt)
if raw is not None:
    try:
        exempt_df = parse_byungmu_exempt(raw)
    except Exception as e:
        st.error(f"병역면제 로드 오류: {e}")

regional_inc_df = jibang_disease_df = None
national_weighted = 0.0
flu_df = ari_series = None

raw = _load_raw("regional", f_regional)
if raw is not None:
    try:
        regional_inc_df = parse_infectious_disease_regional(raw)
        jibang_disease_df = aggregate_disease_by_jibang(regional_inc_df)
    except Exception as e:
        st.error(f"지역별 감염병 로드 오류: {e}")

raw = _load_raw("national", f_national)
if raw is not None:
    try:
        national_weighted = parse_infectious_disease_national(raw)
    except Exception as e:
        st.error(f"질병별 감염병 로드 오류: {e}")

raw = _load_raw("flu", f_flu)
if raw is not None:
    try:
        flu_df = parse_influenza(raw)
    except Exception as e:
        st.error(f"인플루엔자 로드 오류: {e}")

raw = _load_raw("ari", f_ari)
if raw is not None:
    try:
        ari_series = parse_ari(raw)
    except Exception as e:
        st.error(f"급성호흡기 로드 오류: {e}")

domestic_info = {"총건수": 1, "총금액": 0, "수의계약건수": 0, "수의계약금액": 0, "업체별건수": pd.Series(dtype=float)}
foreign_info  = {"국외총건수": 0}
strategic_info = {"전략물자품목수": 0}

raw = _load_raw("dapa_dom", f_dapa_dom)
if raw is not None:
    try:
        domestic_info = parse_dapa_domestic(raw)
    except Exception as e:
        st.error(f"국내조달 로드 오류: {e}")

raw = _load_raw("dapa_for", f_dapa_for)
if raw is not None:
    try:
        foreign_info = parse_dapa_foreign(raw)
    except Exception as e:
        st.error(f"국외조달 로드 오류: {e}")

raw = _load_raw("dapa_bidders", f_dapa_bidders)
bidders_info = None
if raw is not None:
    try:
        bidders_info = parse_dapa_bidders(raw)
    except Exception as e:
        st.error(f"입찰참여업체 로드 오류: {e}")

raw = _load_raw("strategic", f_strategic)
if raw is not None:
    try:
        strategic_info = parse_strategic_goods(raw)
    except Exception as e:
        st.error(f"전략물자 로드 오류: {e}")

population_df = None
raw = _load_raw("population", f_population)
if raw is not None:
    try:
        population_df = parse_population(raw)
    except Exception as e:
        st.error(f"인구 데이터 로드 오류: {e}")

# ─────────────────────────────────────────────
# Risk Score 계산
# ─────────────────────────────────────────────

if exam_df is None:
    st.warning("병역판정검사 데이터가 없어 인력 Risk 계산이 불가합니다. 시뮬레이션 모드를 사용하세요.")
    st.stop()

st.divider()
st.subheader("Risk Score 계산")

with st.spinner("인력 Risk 계산 중..."):
    try:
        manpower_df, mp_warnings = calc_manpower_risk(exam_df, enlist_df, exempt_df, population_df)
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
            domestic_info, foreign_info, strategic_info, bidders_info
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
st.subheader("통합 Risk Score 현황")
render_kpi_cards(result_df)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "지도", "통합 결과", "인력 세부", "감염병 DC", "물자 Risk", "AI 분석", "대응 가이드"
])

with tab1:
    st.markdown("#### 지방청별 통합 Risk Score 지도")
    render_map(result_df, "통합Risk", "지방청별 통합 Risk Score")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 인력 Risk")
        render_map(result_df, "인력Risk", "인력 Risk Score")
    with col_b:
        st.markdown("##### 감염병 DC")
        render_map(result_df, "감염병DC", "감염병 Disruption Coefficient")

with tab2:
    st.markdown("#### 지방청별 통합 Risk Score")
    render_result_table(result_df)
    st.markdown("#### 바 차트")
    render_bar_chart(result_df, "통합Risk", "지방청별 통합 Risk Score")
    st.markdown("---")
    st.download_button(
        "결과 CSV 다운로드",
        data=result_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="triguard_risk_result.csv",
        mime="text/csv",
    )

with tab3:
    st.markdown("#### 인력 Risk 세부 지표")
    render_manpower_detail(manpower_df)
    if not manpower_df.empty and "인력Risk" in manpower_df.columns:
        render_bar_chart(result_df, "인력Risk", "지방청별 인력 Risk Score")

with tab4:
    st.markdown("#### 감염병 Disruption Coefficient")
    render_disease_components(dc_components, dc_score)
    if regional_scored is not None and not regional_scored.empty and "발생률지수" in regional_scored.columns:
        st.dataframe(
            regional_scored[["시도", "총발생률", "발생률지수"]]
            .sort_values("발생률지수", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

with tab5:
    st.markdown("#### 물자 Risk (전국 단위)")
    st.caption("※ 방위사업청 데이터는 지역 단위 없음 → 전국 단일값을 모든 지방청에 동일 적용")
    render_material_components(mat_components, mat_score)

with tab6:
    st.markdown("#### AI 위험 예측 분석")

    with st.spinner("RandomForest 모델 학습 중..."):
        try:
            model, scaler, feat_imp_df, cv_score = train_risk_model(result_df)
            pred_df = predict_risk(model, scaler, result_df)
            cross_data = cross_agency_correlation(manpower_df, jibang_dc_df)
            trend_df = manpower_disease_trend(exam_df)
            forecast_df = forecast_manpower(exam_df, forecast_years=3)
            anomaly_df = detect_anomaly_regions(result_df)
            ml_ok = True
        except Exception as e:
            st.error(f"ML 분석 실패: {e}")
            ml_ok = False

    if ml_ok:
        ai_sub = st.tabs(["예측 결과", "피처 중요도", "교차 분석", "트렌드 & 예측", "이상 탐지"])

        with ai_sub[0]:
            st.markdown("##### Rule-based vs ML 예측 비교")
            st.caption("Rule-based 점수와 ML 예측 등급이 다른 경우 추가 검토가 필요합니다.")
            merged_pred = result_df[["지방청", "통합Risk", "위험등급"]].merge(
                pred_df[["지방청", "예측등급", "정상_확률", "주의_확률", "위험_확률"]],
                on="지방청", how="left"
            )
            merged_pred["불일치"] = merged_pred["위험등급"] != merged_pred["예측등급"]
            render_ml_prediction(merged_pred, cv_score)
            mismatch = merged_pred[merged_pred["불일치"]]["지방청"].tolist()
            if mismatch:
                st.warning(f"Rule-based와 ML 예측 불일치 권역: {', '.join(mismatch)}")
            else:
                st.success("전 권역 Rule-based와 ML 예측 일치")

        with ai_sub[1]:
            st.markdown("##### 위험 예측 피처 중요도")
            render_feature_importance(feat_imp_df)

        with ai_sub[2]:
            st.markdown("##### 3개 기관 데이터 교차 분석")
            render_cross_agency_scatter(
                cross_data["scatter_df"],
                cross_data["correlation"] or 0.0,
                cross_data["insight"],
            )
            if cross_data["high_risk_combo"]:
                st.error(f"인력·감염병 복합 위험 권역: {', '.join(cross_data['high_risk_combo'])}")

        with ai_sub[3]:
            st.markdown("##### 전국 처분인원 추세 및 예측")
            render_manpower_trend(forecast_df)
            if not trend_df.empty:
                st.dataframe(trend_df, use_container_width=True)

        with ai_sub[4]:
            st.markdown("##### 이상 권역 탐지 (Z-score 기반)")
            st.caption("통합 Risk Score 기준 Z-score ≥ 1.5인 권역을 이상 권역으로 탐지합니다.")
            render_anomaly_table(anomaly_df)
            if "이상권역" in anomaly_df.columns:
                anomaly_regions = anomaly_df[anomaly_df["이상권역"] == True]["지방청"].tolist()
                if anomaly_regions:
                    st.info(f"이상 탐지 권역: {', '.join(anomaly_regions)}")

with tab7:
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
