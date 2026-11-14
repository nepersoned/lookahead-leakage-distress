"""지표 패널 — A와 B를 여러 지표로 한 번에 비교한다.

PR-AUC 하나로 말하면 두 가지를 못 본다.
  1. 절대값이 작아 "격차가 없다"처럼 보인다 (무작위 기준선이 사건률이라 그렇다)
  2. 상위 구간에 격차가 몰리는 성질이 안 보인다

그래서 순위 기반(AUC·KS), 희소사건용(PR-AUC), 실무용(상위 k개), 그리고
복제 대상 논문이 쓴 지표(정확도·정밀도·재현율·F1·Kappa)를 함께 낸다.

사용:
    python src/metrics.py --month 1                # 기본 변수 10개
    python src/metrics.py --month 1 --vars 21      # 논문 변수 21개
    python src/metrics.py --month 1 --rule e2      # 공통표본 요건 해제
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, cohen_kappa_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)

import gate3e_full as g

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
OUT = ROOT / "results"

VARS21 = [f"P{i}" for i in range(1, 8)] + [f"S{i}" for i in range(1, 8)] + \
         [f"A{i}" for i in range(1, 5)] + [f"G{i}" for i in range(1, 4)]


def panel(y, p, prevalence=None) -> dict:
    prevalence = float(y.mean()) if prevalence is None else prevalence
    fpr, tpr, _ = roc_curve(y, p)
    order = np.argsort(-p)
    m = {
        "AUC": roc_auc_score(y, p),
        "KS": float(np.max(tpr - fpr)),
        "PR-AUC": average_precision_score(y, p),
        "PR-AUC/무작위": average_precision_score(y, p) / prevalence,
    }
    for k in (50, 100, 200):
        hit = int(y[order[:k]].sum())
        m[f"상위{k} 적중"] = hit
        m[f"상위{k} 정밀도"] = hit / k
        m[f"상위{k} 재현율"] = hit / max(y.sum(), 1)
        m[f"상위{k} Lift"] = (hit / k) / prevalence

    # 복제 대상 논문이 쓴 지표 — 임계값은 사건 수만큼 상위를 잡는 방식으로 둔다
    cut = np.zeros_like(y)
    cut[order[:int(y.sum())]] = 1
    pr, rc, f1, _ = precision_recall_fscore_support(y, cut, average="binary",
                                                    zero_division=0)
    m.update({"정밀도": pr, "재현율": rc, "F1": f1,
              "Kappa": cohen_kappa_score(y, cut),
              "정확도": float((cut == y).mean())})
    return m


def _fit(X, M, seed=42):
    import lightgbm as lgb
    tr = M["pred_year"].isin(g.TRAIN_YEARS).values
    te = M["pred_year"].isin(g.TEST_YEARS).values
    lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15,
                             min_child_samples=10, class_weight="balanced",
                             verbose=-1, random_state=seed)
    clf.fit(X[tr].clip(lo, hi, axis=1), M.loc[tr, "y"].values)
    return clf.predict_proba(X[te].clip(lo, hi, axis=1))[:, 1], M.loc[te, "y"].values


def load_ratios(which: str):
    if which == "21":
        df = pd.read_csv(FULL / "vars21_full.csv", dtype={"corp_code": str})
        df["corp_code"] = df["corp_code"].str.zfill(8)
        return df, VARS21
    df = pd.read_csv(FULL / "ratios_full.csv", dtype={"corp_code": str})
    df["corp_code"] = df["corp_code"].str.zfill(8)
    return df, g.RATIO_COLS


def build_e2(ratios, cols, pub, labels, month, horizon=365):
    """A와 B를 각각의 최대 표본으로 따로 만든다(공통표본 요건 해제)."""
    r = ratios.set_index(["corp_code", "bsns_year"])
    by: dict[str, list] = {}
    for c, fy, d in zip(pub.corp_code, pub.bsns_year, pub.pub_dt):
        by.setdefault(c, []).append((fy, d))
    ev = labels.groupby("corp_code")["event_dt"].apply(list).to_dict()
    rows = {"A": [], "B": []}
    meta = {"A": [], "B": []}
    for y in g.TRAIN_YEARS + g.TEST_YEARS:
        T = pd.Timestamp(year=y, month=month, day=1)
        for code in sorted(set(ratios.corp_code)):
            av = [(fy, d) for fy, d in by.get(code, []) if d < T]
            dates = ev.get(code, [])
            hit = any(T <= d < T + pd.Timedelta(days=horizon) for d in dates)
            if any(d < T for d in dates) and not hit:
                continue
            md = {"corp_code": code, "pred_year": y, "y": int(hit)}
            if (code, y - 1) in r.index:
                rows["A"].append(r.loc[(code, y - 1), cols].to_dict())
                meta["A"].append(md)
            if av:
                fyb = max(fy for fy, _ in av)
                if (code, fyb) in r.index:
                    rows["B"].append(r.loc[(code, fyb), cols].to_dict())
                    meta["B"].append(md)
    return {k: (pd.DataFrame(rows[k]), pd.DataFrame(meta[k])) for k in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--vars", default="10", choices=["10", "21"])
    ap.add_argument("--rule", default="common", choices=["common", "e2"])
    a = ap.parse_args()

    ratios, cols = load_ratios(a.vars)
    _, pub, labels = g.load()

    if a.rule == "common":
        A, B, M = g.build(ratios.assign(**{}), pub, labels,
                          g.TRAIN_YEARS + g.TEST_YEARS, a.month) \
            if a.vars == "10" else _build_generic(ratios, cols, pub, labels, a.month)
        pa, y = _fit(A, M)
        pb, _ = _fit(B, M)
        prev = float(y.mean())
        res = {"A(관행)": panel(y, pa, prev), "B(정합)": panel(y, pb, prev)}
    else:
        d = build_e2(ratios, cols, pub, labels, a.month)
        res = {}
        for tag, key in (("A(관행)", "A"), ("B(정합)", "B")):
            X, M = d[key]
            p, y = _fit(X, M)
            res[tag] = panel(y, p)

    df = pd.DataFrame(res)
    df["격차"] = df["A(관행)"] - df["B(정합)"]
    df["상대차"] = (df["격차"] / df["B(정합)"].replace(0, np.nan) * 100)
    print(f"\n=== {a.month}월 · 변수 {a.vars}개 · 표본규칙 {a.rule} ===")
    print(df.round(3).to_string())
    OUT.mkdir(exist_ok=True)
    name = f"metrics_m{a.month}_v{a.vars}_{a.rule}.csv"
    df.to_csv(OUT / name, encoding="utf-8-sig")
    print(f"\n→ results/{name}")


def _build_generic(ratios, cols, pub, labels, month, horizon=365):
    """gate3e.build의 변수 세트 일반화판."""
    r = ratios.set_index(["corp_code", "bsns_year"])
    by: dict[str, list] = {}
    for c, fy, d in zip(pub.corp_code, pub.bsns_year, pub.pub_dt):
        by.setdefault(c, []).append((fy, d))
    ev = labels.groupby("corp_code")["event_dt"].apply(list).to_dict()
    A, B, M = [], [], []
    for y in g.TRAIN_YEARS + g.TEST_YEARS:
        T = pd.Timestamp(year=y, month=month, day=1)
        for code in sorted(set(ratios.corp_code)):
            av = [(fy, d) for fy, d in by.get(code, []) if d < T]
            if not av:
                continue
            fy_b = max(fy for fy, _ in av)
            if (code, y - 1) not in r.index or (code, fy_b) not in r.index:
                continue
            dates = ev.get(code, [])
            hit = any(T <= d < T + pd.Timedelta(days=horizon) for d in dates)
            if any(d < T for d in dates) and not hit:
                continue
            A.append(r.loc[(code, y - 1), cols].to_dict())
            B.append(r.loc[(code, fy_b), cols].to_dict())
            M.append({"corp_code": code, "pred_year": y, "y": int(hit)})
    return pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(M)


if __name__ == "__main__":
    main()
