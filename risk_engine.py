import pandas as pd
import numpy as np

def normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([50.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val) * 100

def compute_manpower_risk(df):
    score = (
        normalize(df["입영인원_감소율"])     * 0.30 +
        normalize(df["병역판정_감소율"])     * 0.25 +
        normalize(df["병역면제율"])          * 0.20 +
        normalize(df["입영차질률"])          * 0.15 +
        normalize(df["재신체검사율"])        * 0.10
    )
    return score.clip(0, 100)

def compute_disease_risk(df):
    score = (
        normalize(df["감염병_발생률"])       * 0.35 +
        normalize(df["질병등급_가중합"])     * 0.30 +
        normalize(df["인플루엔자_유행강도"]) * 0.20 +
        normalize(df["급성호흡기_트렌드"])   * 0.15
    )
    return score.clip(0, 100)

def compute_supply_risk(df):
    score = (
        normalize(df["국내조달_감소율"])     * 0.25 +
        normalize(df["국외조달_의존도"])     * 0.25 +
        normalize(df["공급업체_집중도"])     * 0.20 +
        normalize(df["수의계약_의존도"])     * 0.15 +
        normalize(df["전략물자_계약비율"])   * 0.15
    )
    return score.clip(0, 100)

def get_grade(score):
    if score >= 70:
        return "🔴 위험"
    elif score >= 40:
        return "🟡 주의"
    else:
        return "🟢 정상"

def run_engine(df):
    result = df[["지역"]].copy()
    result["인력_Risk"]  = compute_manpower_risk(df).round(1)
    result["감염병_DC"]  = compute_disease_risk(df).round(1)
    result["물자_Risk"]  = compute_supply_risk(df).round(1)
    result["통합_Score"] = (
        result["인력_Risk"]  * 0.40 +
        result["감염병_DC"]  * 0.40 +
        result["물자_Risk"]  * 0.20
    ).round(1).clip(0, 100)
    result["위험등급"] = result["통합_Score"].apply(get_grade)
    return result.sort_values("통합_Score", ascending=False).reset_index(drop=True)