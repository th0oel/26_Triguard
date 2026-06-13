# modules/visualize.py
"""
TriGuard AI - Streamlit 시각화 모듈
위험등급 색상, 테이블, 차트, 지도, 대응 가이드
"""

import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# 색상 상수
# ─────────────────────────────────────────────

GRADE_COLORS = {
    "정상": "#2ecc71",
    "주의": "#f1c40f",
    "위험": "#e74c3c",
}

GRADE_BG = {
    "정상": "background-color: #d5f5e3; color: #1a5c34;",
    "주의": "background-color: #fef9e7; color: #7d6608;",
    "위험": "background-color: #fadbd8; color: #922b21;",
}

# 지방청 → 시도 역매핑 (choropleth용)
JIBANG_TO_SIDO = {
    "서울":    ["서울특별시"],
    "부산울산": ["부산광역시", "울산광역시"],
    "대구경북": ["대구광역시", "경상북도"],
    "경인":    ["경기도"],
    "경기북부": ["경기도"],
    "광주전남": ["광주광역시", "전라남도"],
    "대전충남": ["대전광역시", "충청남도", "세종특별자치시"],
    "강원":    ["강원도"],
    "강원영동": ["강원도"],
    "충북":    ["충청북도"],
    "전북":    ["전라북도"],
    "경남":    ["경상남도"],
    "제주":    ["제주특별자치도"],
    "인천":    ["인천광역시"],
}

# GeoJSON 경로 (app.py 기준 상대경로)
GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "korea_provinces.json")
GEOJSON_FALLBACK = os.path.join(os.path.dirname(__file__), "..", "korea_provinces.json")


def _load_geojson():
    for path in [GEOJSON_PATH, GEOJSON_FALLBACK]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


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
        styled = styled.map(fn, subset=[col])
    return styled


# ─────────────────────────────────────────────
# KPI 카드
# ─────────────────────────────────────────────

def render_kpi_cards(result_df: pd.DataFrame):
    counts = result_df["위험등급"].value_counts()
    danger  = counts.get("위험", 0)
    caution = counts.get("주의", 0)
    normal  = counts.get("정상", 0)
    total   = len(result_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("위험 지역", f"{danger}개")
    col2.metric("주의 지역", f"{caution}개")
    col3.metric("정상 지역", f"{normal}개")
    col4.metric("전체 권역", f"{total}개")


# ─────────────────────────────────────────────
# 통합 결과 테이블
# ─────────────────────────────────────────────

def render_result_table(result_df: pd.DataFrame):
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
# 바 차트 (plotly — 색상 적용)
# ─────────────────────────────────────────────

def render_bar_chart(result_df: pd.DataFrame, score_col: str = "통합Risk", title: str = "지방청별 통합 Risk Score"):
    if score_col not in result_df.columns or "지방청" not in result_df.columns:
        st.warning(f"차트 컬럼 누락: {score_col}")
        return

    chart_df = result_df[["지방청", score_col, "위험등급"]].copy()
    chart_df = chart_df.sort_values(score_col, ascending=True)
    chart_df["color"] = chart_df["위험등급"].map(GRADE_COLORS)

    fig = go.Figure(go.Bar(
        x=chart_df[score_col],
        y=chart_df["지방청"],
        orientation="h",
        marker_color=chart_df["color"],
        text=chart_df[score_col].apply(lambda v: f"{v:.1f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Risk Score", range=[0, 100]),
        yaxis=dict(title=""),
        height=420,
        margin=dict(l=10, r=40, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_vline(x=60, line_dash="dash", line_color="#e74c3c", annotation_text="위험", annotation_position="top")
    fig.add_vline(x=35, line_dash="dash", line_color="#f1c40f", annotation_text="주의", annotation_position="top")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# 지도 시각화 (choropleth)
# ─────────────────────────────────────────────

def render_map(result_df: pd.DataFrame, score_col: str = "통합Risk", title: str = "지방청별 Risk Score 지도"):
    """
    지방청별 Risk Score를 시도 단위 choropleth 지도로 표시.
    하나의 지방청이 여러 시도에 걸치는 경우 동일값 복사.
    """
    geo = _load_geojson()
    if geo is None:
        st.warning("GeoJSON 파일이 없어 지도를 표시할 수 없습니다. data/ 폴더에 korea_provinces.json을 넣어주세요.")
        return

    # 지방청 → 시도 확장
    rows = []
    for _, row in result_df.iterrows():
        jibang = str(row["지방청"]).strip()
        sido_list = JIBANG_TO_SIDO.get(jibang, [])
        for sido in sido_list:
            rows.append({
                "시도":   sido,
                "지방청": jibang,
                score_col: row[score_col],
                "위험등급": row.get("위험등급", "정상"),
            })

    if not rows:
        st.warning("지도 매핑 데이터가 없습니다.")
        return

    map_df = pd.DataFrame(rows)
    # 같은 시도에 여러 지방청이 매핑될 경우 평균
    map_df = map_df.groupby("시도", as_index=False).agg({
        score_col: "mean",
        "지방청": lambda x: " / ".join(sorted(set(x))),
        "위험등급": "first",
    })

    # GeoJSON feature key
    for feat in geo["features"]:
        feat["id"] = feat["properties"]["name"]

    color_scale = [
        [0.0,  "#2ecc71"],
        [0.35, "#2ecc71"],
        [0.35, "#f1c40f"],
        [0.60, "#f1c40f"],
        [0.60, "#e74c3c"],
        [1.0,  "#e74c3c"],
    ]

    fig = px.choropleth(
        map_df,
        geojson=geo,
        locations="시도",
        featureidkey="properties.name",
        color=score_col,
        color_continuous_scale=color_scale,
        range_color=[0, 100],
        hover_name="시도",
        hover_data={"지방청": True, score_col: ":.1f", "위험등급": True},
        title=title,
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
    )
    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(
            title="Risk Score",
            tickvals=[0, 35, 60, 100],
            ticktext=["0 (정상)", "35 (주의)", "60 (위험)", "100"],
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("※ 지방청 권역이 여러 시도에 걸치는 경우(예: 부산울산, 대구경북) 해당 시도에 동일 점수 표시")


# ─────────────────────────────────────────────
# 세부 지표 테이블
# ─────────────────────────────────────────────

def render_manpower_detail(manpower_df: pd.DataFrame):
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
    st.metric("감염병 Disruption Coefficient (전국 대표값)", f"{dc_score:.1f} / 100")
    comp_df = pd.DataFrame(list(components.items()), columns=["지표", "점수"])
    st.dataframe(comp_df, use_container_width=True)


def render_material_components(components: dict, mat_score: float):
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
# 경고 메시지
# ─────────────────────────────────────────────

def render_warnings(warnings: list):
    for w in warnings:
        st.warning(w)


# ─────────────────────────────────────────────
# ML / AI 예측 시각화
# ─────────────────────────────────────────────

def render_forecast(forecast_df: pd.DataFrame):
    """다음 분기 예측 결과 테이블 + 차트."""
    # 테이블
    def color_change(val):
        if "악화" in str(val):
            return "color: #e74c3c; font-weight: bold;"
        elif "개선" in str(val):
            return "color: #27ae60;"
        return ""

    styled = forecast_df.style.applymap(color_change, subset=["등급변화"])
    st.dataframe(styled, use_container_width=True)

    # 현재 vs 예측 바 차트
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="현재 Score",
        x=forecast_df["지방청"],
        y=forecast_df["현재Score"],
        marker_color="#95a5a6",
    ))
    fig.add_trace(go.Bar(
        name="예측 Score (다음 분기)",
        x=forecast_df["지방청"],
        y=forecast_df["예측Score(다음분기)"],
        marker_color="#e67e22",
    ))
    fig.add_hline(y=60, line_dash="dash", line_color="#e74c3c", annotation_text="위험")
    fig.add_hline(y=35, line_dash="dash", line_color="#f1c40f", annotation_text="주의")
    fig.update_layout(
        title="현재 vs 다음 분기 예측 Risk Score",
        barmode="group",
        height=420,
        yaxis=dict(range=[0, 100]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cross_agency(cross: dict):
    """3개 기관 교차 분석 결과 시각화."""

    # 기관별 기여도 파이 차트
    contrib_df = cross.get("agency_contribution")
    if contrib_df is not None:
        fig = go.Figure(go.Pie(
            labels=contrib_df["기관"],
            values=contrib_df["기여도"],
            hole=0.4,
            marker_colors=["#3498db", "#e74c3c", "#f39c12"],
            textinfo="label+percent",
        ))
        fig.update_layout(
            title="통합Risk Score 기관별 기여도",
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(contrib_df, use_container_width=True)

    # 상관계수 표시
    col1, col2 = st.columns(2)
    corr_mp_dc  = cross.get("correlation_mp_dc")
    corr_dc_int = cross.get("correlation_dc_integrated")
    if corr_mp_dc is not None:
        col1.metric("인력Risk ↔ 감염병DC 상관계수", f"{corr_mp_dc:.3f}",
                    help="양수: 감염병 증가 시 인력 위험도 함께 증가 경향")
    if corr_dc_int is not None:
        col2.metric("감염병DC ↔ 통합Risk 상관계수", f"{corr_dc_int:.3f}")

    # 위험 패턴
    pattern_counts = cross.get("pattern_counts", {})
    if pattern_counts:
        pat_df = pd.DataFrame(list(pattern_counts.items()), columns=["패턴", "지방청수"])
        pat_df = pat_df.sort_values("지방청수", ascending=False)
        fig2 = go.Figure(go.Bar(
            x=pat_df["패턴"], y=pat_df["지방청수"],
            marker_color=["#e74c3c" if "복합" in p else "#f1c40f" if "위험" in p else "#2ecc71"
                          for p in pat_df["패턴"]],
        ))
        fig2.update_layout(
            title="지방청별 위험 패턴 분포",
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 히트맵
    heatmap = cross.get("jibang_risk_matrix")
    if heatmap is not None and not heatmap.empty:
        fig3 = go.Figure(go.Heatmap(
            z=heatmap.values,
            x=list(heatmap.columns),
            y=list(heatmap.index),
            colorscale=[[0, "#2ecc71"], [0.35, "#f1c40f"], [0.6, "#e74c3c"], [1.0, "#922b21"]],
            zmin=0, zmax=100,
            text=heatmap.values.round(1),
            texttemplate="%{text}",
        ))
        fig3.update_layout(
            title="지방청 × 기관 Risk Score 히트맵",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3, use_container_width=True)


def render_scenario(scenario_df: pd.DataFrame, region: str):
    """시나리오 분석 결과 시각화."""
    colors = [GRADE_COLORS.get(g, "#95a5a6") for g in scenario_df["등급"]]
    fig = go.Figure(go.Bar(
        x=scenario_df["시나리오"],
        y=scenario_df["예측Score"],
        marker_color=colors,
        text=scenario_df["예측Score"].apply(lambda v: f"{v:.1f}"),
        textposition="outside",
    ))
    fig.add_hline(y=60, line_dash="dash", line_color="#e74c3c")
    fig.add_hline(y=35, line_dash="dash", line_color="#f1c40f")
    fig.update_layout(
        title=f"[{region}] 시나리오별 예측 Risk Score",
        yaxis=dict(range=[0, 100]),
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(scenario_df, use_container_width=True)


# ─────────────────────────────────────────────
# ML 예측 결과 시각화
# ─────────────────────────────────────────────

def render_ml_prediction(pred_df: pd.DataFrame, cv_score: float):
    """RandomForest 예측 등급 + 확률 테이블."""
    st.metric("모델 교차검증 정확도 (5-fold)", f"{cv_score*100:.1f}%")
    st.caption("실데이터 14개 + 시뮬레이션 300개 증강 학습 기준")

    display = pred_df.copy()
    def color_grade(val):
        return GRADE_BG.get(val, "")

    styled = display.style.applymap(color_grade, subset=["예측등급"])
    st.dataframe(styled, use_container_width=True)


def render_feature_importance(feat_imp_df: pd.DataFrame):
    """피처 중요도 가로 바 차트."""
    fig = go.Figure(go.Bar(
        x=feat_imp_df["중요도"],
        y=feat_imp_df["피처"],
        orientation="h",
        marker_color="#3498db",
        text=feat_imp_df["중요도"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
    ))
    fig.update_layout(
        title="위험 예측 모델 피처 중요도",
        xaxis=dict(title="중요도", range=[0, feat_imp_df["중요도"].max() * 1.2]),
        yaxis=dict(title=""),
        height=360,
        margin=dict(l=10, r=40, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cross_agency_scatter(scatter_df: pd.DataFrame, correlation: float, insight: str):
    """인력Risk ↔ 감염병DC 산점도."""
    if scatter_df.empty:
        st.info(insight)
        return

    fig = px.scatter(
        scatter_df,
        x="인력Risk",
        y="감염병DC",
        text="지방청",
        title=f"인력 Risk ↔ 감염병 DC 상관관계 (r = {correlation:.2f})",
        labels={"인력Risk": "인력 Risk Score", "감염병DC": "감염병 DC"},
        color="감염병DC",
        color_continuous_scale=[[0, "#2ecc71"], [0.5, "#f1c40f"], [1, "#e74c3c"]],
        range_color=[0, 100],
    )
    fig.update_traces(textposition="top center", marker_size=12)
    fig.add_hline(y=35, line_dash="dash", line_color="#f1c40f", annotation_text="주의선")
    fig.add_hline(y=60, line_dash="dash", line_color="#e74c3c", annotation_text="위험선")
    fig.add_vline(x=35, line_dash="dash", line_color="#f1c40f")
    fig.add_vline(x=60, line_dash="dash", line_color="#e74c3c")
    fig.update_layout(
        height=460,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"💡 {insight}")


def render_manpower_trend(trend_df: pd.DataFrame):
    """연도별 전국 처분인원 + 예측 라인 차트."""
    if trend_df.empty:
        st.info("트렌드 데이터 없음")
        return

    actual = trend_df[trend_df["구분"] == "실측"]
    forecast = trend_df[trend_df["구분"] == "예측"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual["연도"], y=actual["전국처분인원"],
        mode="lines+markers+text",
        name="실측",
        line=dict(color="#3498db", width=3),
        marker=dict(size=8),
        text=actual["전국처분인원"].apply(lambda v: f"{int(v):,}"),
        textposition="top center",
    ))
    if not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast["연도"], y=forecast["전국처분인원"],
            mode="lines+markers+text",
            name="예측 (RandomForest)",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
            text=forecast["전국처분인원"].apply(lambda v: f"{int(v):,}"),
            textposition="top center",
        ))
    fig.update_layout(
        title="전국 병역판정검사 처분인원 추세 및 예측",
        xaxis=dict(title="연도"),
        yaxis=dict(title="처분인원 (명)"),
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_table(anomaly_df: pd.DataFrame):
    """이상 권역 탐지 테이블."""
    if anomaly_df.empty:
        st.info("이상 탐지 데이터 없음")
        return

    def highlight_anomaly(row):
        if row.get("이상권역", False):
            return ["background-color: #fadbd8"] * len(row)
        return [""] * len(row)

    styled = anomaly_df.style.apply(highlight_anomaly, axis=1)
    st.dataframe(styled, use_container_width=True)
