import plotly.express as px
import plotly.graph_objects as go

COLOR_MAP = {
    "🔴 위험": "#e74c3c",
    "🟡 주의": "#f39c12",
    "🟢 정상": "#27ae60"
}

def render_bar_chart(result_df):
    colors = result_df["위험등급"].map(COLOR_MAP)
    fig = go.Figure(go.Bar(
        x=result_df["지역"],
        y=result_df["통합_Score"],
        marker_color=colors,
        text=result_df["통합_Score"],
        textposition="outside"
    ))
    fig.update_layout(
        title="지역별 통합 Risk Score",
        xaxis_title="지역",
        yaxis_title="Risk Score (0~100)",
        yaxis_range=[0, 110],
        height=450
    )
    return fig

def render_radar_chart(result_df, region):
    row = result_df[result_df["지역"] == region].iloc[0]
    categories = ["인력 Risk", "감염병 DC", "물자 Risk"]
    values = [row["인력_Risk"], row["감염병_DC"], row["물자_Risk"]]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]
    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        line_color="#2980b9"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100])),
        title=f"{region} 위험도 상세",
        height=400
    )
    return fig

def render_grade_summary(result_df):
    grade_counts = result_df["위험등급"].value_counts().reset_index()
    grade_counts.columns = ["위험등급", "지역수"]
    fig = px.pie(
        grade_counts,
        names="위험등급",
        values="지역수",
        color="위험등급",
        color_discrete_map=COLOR_MAP,
        title="위험등급 분포"
    )
    return fig