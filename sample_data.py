import pandas as pd
import numpy as np

REGIONS = [
    "서울", "부산", "대구", "인천", "광주",
    "대전", "울산", "세종", "경기", "강원",
    "충북", "충남", "전북", "전남", "경북",
    "경남", "제주"
]

def get_sample_data():
    np.random.seed(42)
    n = len(REGIONS)
    df = pd.DataFrame({
        "지역": REGIONS,
        "입영인원_감소율":     np.random.uniform(0, 30, n),
        "병역판정_감소율":     np.random.uniform(0, 25, n),
        "병역면제율":          np.random.uniform(5, 20, n),
        "입영차질률":          np.random.uniform(0, 15, n),
        "재신체검사율":        np.random.uniform(1, 10, n),
        "감염병_발생률":       np.random.uniform(10, 80, n),
        "질병등급_가중합":     np.random.uniform(5, 60, n),
        "인플루엔자_유행강도": np.random.uniform(0, 50, n),
        "급성호흡기_트렌드":   np.random.uniform(-10, 30, n),
        "국내조달_감소율":     np.random.uniform(0, 40, n),
        "국외조달_의존도":     np.random.uniform(10, 60, n),
        "공급업체_집중도":     np.random.uniform(20, 80, n),
        "수의계약_의존도":     np.random.uniform(5, 40, n),
        "전략물자_계약비율":   np.random.uniform(1, 20, n),
    })
    return df