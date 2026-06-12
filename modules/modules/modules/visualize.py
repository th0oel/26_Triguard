# modules/visualize.py
"""
TriGuard AI - Streamlit 시각화 모듈
위험등급 색상, 테이블, 차트, 대응 가이드
"""

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# 색상 상수
# ─────────────────────────────────────────────

GRADE_COLORS = {
    "정상": "#2ecc71",   # 초록
    "주의": "#f1c40f",   # 노랑
    "위험": "#e74c3c",   # 빨강
}

GRADE_BG = {
    "정상": "background-color: #d5f5e3; color: #1a5c34;",
    "주의": "background-color: #fef9e7; color: #7d6608;",
    "위험": "background-color: #fadbd8; color: #922b21;",
}

# ─────────────────────────────────────────────
# 위험등급 스타일
# ─────────────────────────────────────────────

def style_grade_cell(val: str) -> str:
    return GRADE_BG.get(val, "")


def style_score_cell(val) -> str:
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v >= 60:
        return "background-color: #fadbd8; color: #922b21;"
    elif v >= 35:
        return "background-color: #fef9e7; color: #7d6608;"
    else:
        return "background-color: #d5f5e3; color: #1a5c34;"


def apply_risk_style(df: pd.DataFrame, score_cols: list = None, grade_col: str = "위험등급"):
    """DataFrame에 위험등급 색상 스타일 적용."""
    score_cols = score_cols or []
    style_dict = {}
    if grade_col in df.columns:
        style_dict[grade_col] = style_grade_cell
    for col in score_cols:
        if col in df.columns:
            style_dict[col] = style_score_cell

    if not style_dict:
        return df.style

    styled = df.style
    for col, fn in style_dict.items():
        styled = styled.applymap(fn, subset=[col])
    return styled


# ─────────────────────────────────────────────
# 요약 KPI 카드
# ─────────────────────────────────────────────

def render_kpi_cards(result_df: pd.DataFrame):
    """상단 KPI 카드 3개: 위험/주의/정상 지역 수."""
    counts = result_df["위험등급"].value_counts()
    danger  = counts.get("위험", 0)
    caution = counts.get("주의", 0)
    normal  = counts.get("정상", 0)
    total   = len(result_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("위험 지역", f"{danger}개", delta=None)
    col2.metric("주의 지역", f"{caution}개", delta=None)
    col3.metric("정상 지역", f"{normal}개", delta=None)
    col4.metric("전체 권역", f"{total}개", delta=None)


# ─────────────────────────────────────────────
# 통합 결과 테이블
# ─────────────────────────────────────────────

def render_result_table(result_df: pd.DataFrame):
    """통합 Risk Score 테이블 (색상 적용)."""
    display_cols = [c for c in [
        "지방청", "인력Risk", "감염병DC", "물자Risk", "통합Risk", "위험등급"
    ] if c in result_df.columns]

    display_df = result_df[display_cols].copy()
    display_df = display_df.sort_values("통합Risk", ascending=False).reset_index(drop=True)

    styled = apply_risk_style(
        display_df,
        score_cols=["인력Risk", "감염병DC", "물자Risk", "통합Risk"],
        grade_col="위험등급"
    )
    st.dataframe(styled, use_container_width=True, height=460)


# ─────────────────────────────────────────────
# 바 차트
# ─────────────────────────────────────────────

def render_bar_chart(result_df: pd.DataFrame, score_col: str = "통합Risk", title: str = "지방청별 통합 Risk Score"):
    """가로 바 차트."""
    if score_col not in result_df.columns or "지방청" not in result_df.columns:
        st.warning(f"차트 컬럼 누락: {score_col}")
        return

    chart_df = result_df[["지방청", score_col, "위험등급"]].copy()
    chart_df = chart_df.sort_values(score_col, ascending=True)

    # Streamlit native bar_chart
    st.bar_chart(
        chart_df.set_index("지방청")[score_col],
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# 세부 지표 테이블
# ─────────────────────────────────────────────

def render_manpower_detail(manpower_df: pd.DataFrame):
    """인력 Risk 세부 지표 테이블."""
    cols = [c for c in [
        "지방청", "입영인원_감소율", "병역판정검사_감소율",
        "병역면제율", "재신체검사율", "입영_차질률", "인력Risk"
    ] if c in manpower_df.columns]

    if not cols:
        st.info("인력 세부 지표 없음")
        return

    display = manpower_df[cols].copy().sort_values("인력Risk", ascending=False).reset_index(drop=True)
    styled = apply_risk_style(display, score_cols=["인력Risk"])
    st.dataframe(styled, use_container_width=True)


def render_disease_components(components: dict, dc_score: float):
    """감염병 DC 구성요소 표시."""
    st.metric("감염병 Disruption Coefficient (전국 대표값)", f"{dc_score:.1f} / 100")
    comp_df = pd.DataFrame(list(components.items()), columns=["지표", "점수"])
    st.dataframe(comp_df, use_container_width=True)


def render_material_components(components: dict, mat_score: float):
    """물자 Risk 구성요소 표시."""
    st.metric("물자 Risk Score (전국)", f"{mat_score:.1f} / 100")
    comp_df = pd.DataFrame(list(components.items()), columns=["지표", "점수"])
    st.dataframe(comp_df, use_container_width=True)


# ─────────────────────────────────────────────
# Rule-based 대응 가이드
# ─────────────────────────────────────────────

RESPONSE_GUIDE = {
    "위험": {
        "인력": [
            "해당 권역 병무청에 병역판정검사 임시 확대 실시 검토",
            "인접 지방청 자원 재배분 가능 여부 긴급 검토",
            "입영 기피자 사유 조사 및 행정 지도 강화",
            "전시근로역·보충역 활용 가능성 검토",
        ],
        "감염병": [
            "감염병 확산 지역 병역판정검사 일정 조정 고려",
            "입영 전 건강검진 항목 추가 및 감염병 검사 의무화",
            "입영 집결지 방역 강화 및 격리시설 사전 확보",
            "질병관리청 긴급대응팀과 정보 공유 체계 가동",
        ],
        "물자": [
            "주요 군수품 긴급 국내 대체 조달 방안 수립",
            "국외 의존 품목 국내 대체 공급선 확보 추진",
            "전략물자 비축 현황 점검 및 긴급 확충 요청",
            "수의계약 과도 의존 품목 경쟁 입찰 전환 검토",
        ],
    },
    "주의": {
        "인력": [
            "병역판정검사 처분인원 감소 추세 모니터링 강화",
            "입영 차질 요인 분석 및 사전 대응 계획 수립",
        ],
        "감염병": [
            "감염병 동향 주간 모니터링 및 예방 지침 배포",
            "유관 기관(질병관리청) 데이터 연계 점검",
        ],
        "물자": [
            "공급업체 집중도 완화를 위한 복수 공급선 발굴",
            "국외 조달 비중 분기별 점검 및 보고",
        ],
    },
    "정상": {
        "인력":   ["현 수준 유지 및 정기 모니터링 지속"],
        "감염병": ["표준 감시 체계 유지"],
        "물자":   ["정기 조달 계획 이행 점검"],
    },
}


def render_response_guide(result_df: pd.DataFrame):
    """
    지방청별 위험등급에 따른 rule-based 대응 가이드 출력.
    ※ 향후 RAG 기반 대응 가이드로 확장 예정
    """
    st.caption(
        "현재 MVP에서는 rule-based 대응 가이드를 제공합니다. "
        "향후 RAG 기반 대응 가이드로 확장 예정입니다."
    )

    danger_regions  = result_df[result_df["위험등급"] == "위험"]["지방청"].tolist()
    caution_regions = result_df[result_df["위험등급"] == "주의"]["지방청"].tolist()

    for region_list, label in [(danger_regions, "위험"), (caution_regions, "주의")]:
        if not region_list:
            continue
        for region in region_list:
            row = result_df[result_df["지방청"] == region].iloc[0]
            with st.expander(f"[{label}] {region} — 통합Risk: {row['통합Risk']}"):
                guide = RESPONSE_GUIDE.get(label, {})

                # 어느 영역이 가장 위험한지 판단
                scores = {
                    "인력":   row.get("인력Risk", 0),
                    "감염병": row.get("감염병DC", 0),
                    "물자":   row.get("물자Risk", 0),
                }
                primary = max(scores, key=scores.get)

                st.markdown(f"**주요 위험 요인:** {primary} (점수: {scores[primary]:.1f})")
                st.markdown("**권고 조치:**")
                for action in guide.get(primary, guide.get("인력", [])):
                    st.markdown(f"- {action}")

    if not danger_regions and not caution_regions:
        st.success("현재 전 권역 정상 수준입니다. 정기 모니터링을 지속하세요.")


# ─────────────────────────────────────────────
# 경고 메시지 렌더링
# ─────────────────────────────────────────────

def render_warnings(warnings: list):
    for w in warnings:
        st.warning(w)
