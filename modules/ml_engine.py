# modules/ml_engine.py
"""
TriGuard AI - ML 예측 엔진
- RandomForest 기반 위험 등급 예측
- 3개 기관 교차 분석 (감염병 ↔ 인력 상관관계)
- 시계열 트렌드 분석
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import warnings as pywarnings


# ─────────────────────────────────────────────
# 1. RandomForest 위험 예측 모델
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "입영인원_감소율",
    "병역판정검사_감소율",
    "병역면제율",
    "재신체검사율",
    "입영_차질률",
    "감염병DC",
    "물자Risk",
]

LABEL_COL = "위험등급"
GRADE_ORDER = {"정상": 0, "주의": 1, "위험": 2}
GRADE_REVERSE = {0: "정상", 1: "주의", 2: "위험"}


def _encode_grades(series: pd.Series) -> np.ndarray:
    return series.map(GRADE_ORDER).fillna(0).astype(int).values


def train_risk_model(result_df: pd.DataFrame, n_simulated: int = 300, seed: int = 42):
    """
    실데이터 + 시뮬레이션 증강 데이터로 RandomForest 학습.
    실데이터가 적을 때(14개 지방청) 과적합 방지를 위해 시뮬레이션 데이터 증강.
    반환: (model, scaler, feature_importance_df, cv_score)
    """
    rng = np.random.default_rng(seed)

    # ── 실데이터 준비 ──
    real_df = result_df.copy()
    avail_features = [c for c in FEATURE_COLS if c in real_df.columns]

    # 없는 피처는 0으로 채움
    for col in FEATURE_COLS:
        if col not in real_df.columns:
            real_df[col] = 0.0

    real_X = real_df[FEATURE_COLS].fillna(0).values
    real_y = _encode_grades(real_df[LABEL_COL]) if LABEL_COL in real_df.columns else np.zeros(len(real_df), dtype=int)

    # ── 시뮬레이션 증강 ──
    sim_rows = []
    for _ in range(n_simulated):
        feat = {
            "입영인원_감소율":     float(rng.uniform(0, 80)),
            "병역판정검사_감소율": float(rng.uniform(0, 60)),
            "병역면제율":          float(rng.uniform(0, 40)),
            "재신체검사율":        float(rng.uniform(0, 30)),
            "입영_차질률":         float(rng.uniform(0, 20)),
            "감염병DC":            float(rng.uniform(10, 80)),
            "물자Risk":            float(rng.uniform(15, 65)),
        }
        # 가중 통합점수로 라벨 생성 (risk_engine 공식과 동일)
        score = (
            0.40 * (
                0.30 * feat["입영인원_감소율"] +
                0.25 * feat["병역판정검사_감소율"] +
                0.20 * feat["병역면제율"] +
                0.10 * feat["재신체검사율"] +
                0.15 * feat["입영_차질률"]
            ) +
            0.40 * feat["감염병DC"] +
            0.20 * feat["물자Risk"]
        )
        if score >= 60:
            label = 2
        elif score >= 35:
            label = 1
        else:
            label = 0
        sim_rows.append({**feat, "label": label})

    sim_df = pd.DataFrame(sim_rows)
    sim_X = sim_df[FEATURE_COLS].values
    sim_y = sim_df["label"].values

    # ── 학습 ──
    X = np.vstack([real_X, sim_X])
    y = np.concatenate([real_y, sim_y])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
    )

    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        model.fit(X_scaled, y)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

    # 피처 중요도
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "피처": FEATURE_COLS,
        "중요도": importances,
    }).sort_values("중요도", ascending=False).reset_index(drop=True)

    return model, scaler, feat_imp_df, float(cv_scores.mean())


def predict_risk(
    model,
    scaler,
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    학습된 모델로 각 지방청의 위험 등급 및 확률 예측.
    반환: DataFrame with [지방청, 예측등급, 정상확률, 주의확률, 위험확률]
    """
    df = result_df.copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_COLS].fillna(0).values
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)

    # 클래스 순서 보장
    classes = model.classes_
    proba_df = pd.DataFrame(proba, columns=[f"cls_{c}" for c in classes])

    out = pd.DataFrame({
        "지방청": df["지방청"].values,
        "예측등급": [GRADE_REVERSE.get(p, "정상") for p in preds],
    })

    for cls_idx, cls_label in GRADE_REVERSE.items():
        col_name = f"cls_{cls_idx}"
        if col_name in proba_df.columns:
            out[f"{cls_label}_확률"] = (proba_df[col_name].values * 100).round(1)
        else:
            out[f"{cls_label}_확률"] = 0.0

    return out


# ─────────────────────────────────────────────
# 2. 3개 기관 교차 분석
# ─────────────────────────────────────────────

def cross_agency_correlation(
    manpower_df: pd.DataFrame,
    jibang_dc_df: pd.DataFrame = None,
) -> dict:
    """
    인력 Risk ↔ 감염병DC 상관관계 분석 (지방청 단위).
    반환: {
        correlation: float,           # 피어슨 상관계수
        scatter_df: DataFrame,        # 산점도용
        high_risk_combo: list,        # 두 지표 모두 높은 지방청
        insight: str,                 # 자동 인사이트 문장
    }
    """
    if jibang_dc_df is None or jibang_dc_df.empty:
        return {
            "correlation": None,
            "scatter_df": pd.DataFrame(),
            "high_risk_combo": [],
            "insight": "감염병 지방청 데이터 없음 — 교차 분석 불가",
        }

    merged = manpower_df[["지방청", "인력Risk"]].merge(
        jibang_dc_df[["지방청", "감염병DC"]],
        on="지방청",
        how="inner",
    )

    if len(merged) < 3:
        return {
            "correlation": None,
            "scatter_df": merged,
            "high_risk_combo": [],
            "insight": "매핑된 지방청이 부족하여 상관관계 분석 불가",
        }

    corr = merged["인력Risk"].corr(merged["감염병DC"])

    # 두 지표 모두 주의 이상인 지방청
    high_combo = merged[
        (merged["인력Risk"] >= 35) & (merged["감염병DC"] >= 35)
    ]["지방청"].tolist()

    # 자동 인사이트
    if abs(corr) >= 0.7:
        strength = "강한"
    elif abs(corr) >= 0.4:
        strength = "중간 수준의"
    else:
        strength = "약한"

    direction = "양의" if corr >= 0 else "음의"
    insight = (
        f"인력 Risk와 감염병 DC 간 {strength} {direction} 상관관계가 있습니다 "
        f"(r = {corr:.2f}). "
    )
    if high_combo:
        insight += f"양 지표 모두 주의 이상인 권역: {', '.join(high_combo)}."
    else:
        insight += "두 지표 모두 주의 이상인 권역은 없습니다."

    return {
        "correlation": round(corr, 3),
        "scatter_df": merged,
        "high_risk_combo": high_combo,
        "insight": insight,
    }


def manpower_disease_trend(
    exam_df: pd.DataFrame,
    regional_dc_series: pd.Series = None,
) -> pd.DataFrame:
    """
    연도별 인력(처분인원 전국합) + 감염병 트렌드 비교.
    exam_df: 병역판정검사 (연도, 지방청, 처분인원)
    regional_dc_series: 연도별 감염병 수치 Series (index=연도) — 없으면 생략
    반환: DataFrame [연도, 전국처분인원, 감염병지수(있으면)]
    """
    if exam_df.empty or "연도" not in exam_df.columns:
        return pd.DataFrame()

    trend = (
        exam_df.groupby("연도")["처분인원"].sum()
        .reset_index()
        .rename(columns={"처분인원": "전국처분인원"})
    )

    if regional_dc_series is not None and len(regional_dc_series) > 0:
        dc_df = regional_dc_series.reset_index()
        dc_df.columns = ["연도", "감염병지수"]
        trend = trend.merge(dc_df, on="연도", how="left")

    trend = trend.sort_values("연도").reset_index(drop=True)
    return trend


# ─────────────────────────────────────────────
# 3. 시계열 트렌드 & 이상 탐지
# ─────────────────────────────────────────────

def detect_anomaly_regions(result_df: pd.DataFrame, threshold_z: float = 1.5) -> pd.DataFrame:
    """
    통합Risk 기준 Z-score로 이상 권역 탐지.
    반환: DataFrame [지방청, 통합Risk, Z점수, 이상여부]
    """
    if "통합Risk" not in result_df.columns or len(result_df) < 3:
        return result_df.copy()

    scores = result_df["통합Risk"].values.astype(float)
    mean, std = scores.mean(), scores.std()

    out = result_df[["지방청", "통합Risk"]].copy()
    out["Z점수"] = ((scores - mean) / (std + 1e-9)).round(2)
    out["이상권역"] = out["Z점수"].abs() >= threshold_z
    return out.sort_values("Z점수", ascending=False).reset_index(drop=True)


def forecast_manpower(exam_df: pd.DataFrame, forecast_years: int = 3) -> pd.DataFrame:
    """
    전국 처분인원 단순 선형 추세 예측 (향후 N년).
    반환: DataFrame [연도, 전국처분인원, 구분(실측/예측)]
    """
    if exam_df.empty or "연도" not in exam_df.columns or "처분인원" not in exam_df.columns:
        return pd.DataFrame()

    trend = (
        exam_df.groupby("연도")["처분인원"]
        .sum()
        .reset_index()
        .sort_values("연도")
    )

    if len(trend) < 2:
        return trend.assign(구분="실측")

    X = trend["연도"].values.reshape(-1, 1)
    y = trend["전국처분인원"].values if "전국처분인원" in trend.columns else trend["처분인원"].values

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y)

    last_year = int(trend["연도"].max())
    future_years = np.array(range(last_year + 1, last_year + forecast_years + 1)).reshape(-1, 1)
    future_preds = reg.predict(future_years)

    actual = trend.rename(columns={"처분인원": "전국처분인원"}).assign(구분="실측")
    forecast = pd.DataFrame({
        "연도": future_years.flatten(),
        "전국처분인원": future_preds.round(0),
        "구분": "예측",
    })

    return pd.concat([actual, forecast], ignore_index=True)
