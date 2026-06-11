import streamlit as st
from risk_engine import run_engine
from sample_data import get_sample_data
from visualize import render_bar_chart, render_radar_chart, render_grade_summary

st.set_page_config(page_title="TriGuard AI", page_icon="🛡️", layout="wide")

st.title("🛡️ TriGuard AI")
st.markdown("**공공데이터 기반 지역별 병력운용 Risk Score 분석 및 조기경보 시스템**")
st.divider()

df_raw = get_sample_data()
result = run_engine(df_raw)

danger  = len(result[result["위험등급"] == "🔴 위험"])
caution = len(result[result["위험등급"] == "🟡 주의"])
normal  = len(result[result["위험등급"] == "🟢 정상"])

col1, col2, col3 = st.columns(3)
col1.metric("🔴 위험 지역", f"{danger}개")
col2.metric("🟡 주의 지역", f"{caution}개")
col3.metric("🟢 정상 지역", f"{normal}개")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 전체 현황", "🔍 지역별 상세", "📋 데이터 테이블"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(render_bar_chart(result), use_container_width=True)
    with col_b:
        st.plotly_chart(render_grade_summary(result), use_container_width=True)

with tab2:
    region = st.selectbox("지역 선택", result["지역"].tolist())
    row = result[result["지역"] == region].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("인력 Risk",  f"{row['인력_Risk']:.1f}")
    c2.metric("감염병 DC",  f"{row['감염병_DC']:.1f}")
    c3.metric("물자 Risk",  f"{row['물자_Risk']:.1f}")
    c4.metric("통합 Score", f"{row['통합_Score']:.1f}")
    st.plotly_chart(render_radar_chart(result, region), use_container_width=True)

with tab3:
    st.dataframe(result, use_container_width=True, hide_index=True)

st.divider()
st.caption("© 2026 TriGuard AI | 병무청·질병관리청·방위사업청·행정안전부 공공데이터 활용")