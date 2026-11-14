"""확대 표본 재분석 — 코스닥(2011~2024)·코스피(2011~2024) × 평가창 A/B.

표본 확대의 목적은 평가 구간 사건 수를 늘려 신뢰구간을 좁히는 것이다.
기간만 늘리면 평가 사건이 그대로이므로 평가창 두 가지를 함께 돌린다.

    A  학습 2013~2021 / 평가 2022~2024   (기존과 같은 평가창)
    B  학습 2013~2017 / 평가 2018~2024   (평가창을 앞으로 이동)

코스피와 코스닥은 섞지 않는다. 상장·퇴출 기준과 기업 규모가 달라 합치면
시장 효과가 격차에 섞인다. 코스피는 1~4월 집중도가 낮아(44%) 전제가 약한
시장이라, 격차가 작게 나오는 것 자체가 비교 대상이 된다.

기존 모듈(gate3e_full·metrics·models6)은 학습/평가 연도를 모듈 전역값으로
읽으므로, 조건마다 그 값을 바꿔 끼워 그대로 재사용한다.

사용: python src/expand_analysis.py [--boot 2000] [--market KOSDAQ KOSPI]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import gate3e_full as g
import metrics as mx
import models6 as m6

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
OUT = ROOT / "results"

MARKETS = {
    "KOSDAQ": {"ratios": ["ratios_full.csv", "ratios_kosdaq_old.csv"],
               "reports": ["reports_full.csv", "reports_kosdaq_old.csv"],
               "labels": ["kind_events.csv", "kind_events_kosdaq_old.csv"]},
    "KOSPI": {"ratios": ["ratios_kospi.csv"],
              "reports": ["reports_kospi.csv"],
              "labels": ["kind_events_kospi.csv", "kind_events_kospi_old.csv"]},
}
COLS10 = list(g.RATIO_COLS)
SPLITS = {"A": (list(range(2013, 2022)), [2022, 2023, 2024]),
          "B": (list(range(2013, 2018)), list(range(2018, 2025)))}


def _read(names, **kw):
    parts = [pd.read_csv(FULL / n, **kw) for n in names if (FULL / n).exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_market(market, vars_="10"):
    spec = MARKETS[market]
    names = spec["ratios"]
    if vars_ == "21":
        names = [n.replace("ratios", "vars21") for n in names]
    elif vars_ == "classic":
        names = [n.replace("ratios", "classic") for n in names]
    ratios = _read(names, dtype={"corp_code": str})
    ratios["corp_code"] = ratios["corp_code"].str.zfill(8)
    ratios = ratios.drop_duplicates(["corp_code", "bsns_year"])

    rep = _read(spec["reports"], dtype={"corp_code": str})
    rep["corp_code"] = rep["corp_code"].str.zfill(8)
    rep["rcept_dt"] = pd.to_datetime(rep["rcept_dt"].astype(str).str[:10],
                                     format="mixed", errors="coerce")
    rep["is_amended"] = rep["is_amended"].astype(str).str.lower().isin(["true", "1"])
    ann = rep[rep["report_type"] == "사업"]
    orig = ann[~ann["is_amended"]]
    pub = (pd.concat([orig, ann]).sort_values("rcept_dt")
           .groupby(["corp_code", "bsns_year"], as_index=False)["rcept_dt"].first()
           .rename(columns={"rcept_dt": "pub_dt"}))

    from dart import Dart
    kind = _read(spec["labels"], dtype={"stock_code": str}, parse_dates=["rcept_dt"])
    kind = kind[kind["event"] == "지정"].copy()
    import kind_codes
    kind = kind_codes.fill(kind)          # 외국 기업의 빈 종목코드를 회사명으로 채운다
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes["stock_code"].astype(str).str.strip().str.zfill(6)
    cmap = (codes[codes["stock_code"].str.len() == 6]
            .drop_duplicates("stock_code").set_index("stock_code")["corp_code"])
    kind["corp_code"] = kind["stock_code"].map(cmap)
    kind = kind.dropna(subset=["corp_code"])
    kind["corp_code"] = kind["corp_code"].astype(str).str.zfill(8)
    kind["year"] = kind["rcept_dt"].dt.year
    labels = (kind.sort_values("rcept_dt")
              .groupby(["corp_code", "year"], as_index=False).first()
              .rename(columns={"rcept_dt": "event_dt"})[["corp_code", "event_dt"]])
    return ratios, pub, labels


def cols_for(vars_):
    if vars_ == "21":
        return mx.VARS21
    if vars_ == "classic":
        from vars_classic import COLS
        return COLS
    return COLS10


def set_split(split):
    tr, te = SPLITS[split]
    g.TRAIN_YEARS, g.TEST_YEARS = tr, te


def boot_ci(y, pa, pb, groups, n_boot, rng):
    def stat(yy, a, b):
        oa, ob = np.argsort(-a), np.argsort(-b)
        return {"PR-AUC": average_precision_score(yy, a) - average_precision_score(yy, b),
                "AUC": roc_auc_score(yy, a) - roc_auc_score(yy, b),
                "상위50": int(yy[oa[:50]].sum()) - int(yy[ob[:50]].sum()),
                "상위100": int(yy[oa[:100]].sum()) - int(yy[ob[:100]].sum())}
    obs = stat(y, pa, pb)
    idx_by = pd.Series(range(len(groups))).groupby(groups.values).indices
    keys = list(idx_by)
    draws = {k: [] for k in obs}
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([idx_by[keys[j]] for j in pick])
        if y[idx].sum() < 5:
            continue
        for k, v in stat(y[idx], pa[idx], pb[idx]).items():
            draws[k].append(v)
    rows = []
    for k, v in obs.items():
        arr = np.array(draws[k])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append({"지표": k, "격차": round(float(v), 4), "하한": round(float(lo), 4),
                     "상한": round(float(hi), 4),
                     "0초과": round(float((arr > 0).mean()), 3),
                     "유의": bool(lo > 0)})
    return rows


def analyse(market, split, ratios, pub, labels, n_boot, rng):
    set_split(split)
    years = g.TRAIN_YEARS + g.TEST_YEARS
    out = {"market": market, "split": split}

    # 시점 스윕 — 4월 소멸이 확대 표본에서도 재현되는가
    sweep = []
    for month in range(1, 8):
        A, B, M = g.build(ratios, pub, labels, years, month)
        if M.empty or M.loc[M.pred_year.isin(g.TEST_YEARS), "y"].sum() < 5:
            continue
        for r in g.evaluate(A, M, "A") + g.evaluate(B, M, "B"):
            r.update(month=month, n_test_event=int(M.loc[M.pred_year.isin(g.TEST_YEARS), "y"].sum()))
            sweep.append(r)
    sweep = pd.DataFrame(sweep)
    sweep["market"], sweep["split"] = market, split

    # 1월 — 지표 패널 + 기업 단위 부트스트랩
    A, B, M = g.build(ratios, pub, labels, years, 1)
    te = M.pred_year.isin(g.TEST_YEARS).values
    pa, y = mx._fit(A, M)
    pb, _ = mx._fit(B, M)
    out["평가행"] = int(te.sum())
    out["평가사건"] = int(y.sum())
    out["평가기업"] = int(M.loc[te, "corp_code"].nunique())
    ci = boot_ci(y, pa, pb, M.loc[te, "corp_code"].reset_index(drop=True), n_boot, rng)
    for r in ci:
        r.update(market=market, split=split)
    prev = float(y.mean())
    panel = pd.DataFrame({"A": mx.panel(y, pa, prev), "B": mx.panel(y, pb, prev)})
    panel["격차"] = panel["A"] - panel["B"]

    # 모델 6종
    m6rows = []
    for name in m6.build_models():
        r = {"market": market, "split": split, "모델": name}
        for tag, X in (("A", A), ("B", B)):
            p, yy = m6.run_one(m6.build_models()[name], X, M)
            r[f"{tag} PR-AUC"] = round(average_precision_score(yy, p), 4)
            r[f"{tag} 상위50"] = int(yy[np.argsort(-p)[:50]].sum())
        r["격차 PR-AUC"] = round(r["A PR-AUC"] - r["B PR-AUC"], 4)
        m6rows.append(r)
    return out, sweep, pd.DataFrame(ci), panel, pd.DataFrame(m6rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--market", nargs="+", default=["KOSDAQ", "KOSPI"])
    ap.add_argument("--split", nargs="+", default=["A", "B"])
    ap.add_argument("--vars", default="21", choices=["10", "21", "classic"])
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    OUT.mkdir(exist_ok=True)

    summ, sweeps, cis, m6s = [], [], [], []
    for market in a.market:
        g.RATIO_COLS = cols_for(a.vars)
        ratios, pub, labels = load_market(market, a.vars)
        print(f"\n### {market} — 재무비율 {len(ratios):,}건 / 라벨 {len(labels):,}건 "
              f"({labels.event_dt.dt.year.min()}~{labels.event_dt.dt.year.max()})")
        for split in a.split:
            print(f"  평가창 {split} {SPLITS[split]} ...", flush=True)
            out, sw, ci, panel, m6r = analyse(market, split, ratios, pub, labels, a.boot, rng)
            summ.append(out); sweeps.append(sw); cis.append(ci); m6s.append(m6r)
            panel.to_csv(OUT / f"expand_panel_v{a.vars}_{market}_{split}.csv", encoding="utf-8-sig")
            print(f"    평가 {out['평가행']:,}행 · 사건 {out['평가사건']} · {out['평가기업']:,}개사")
            print(ci[["지표", "격차", "하한", "상한", "유의"]].to_string(index=False))
            print(f"    모델 6종 중 A>B: {int((m6r['격차 PR-AUC'] > 0).sum())}/6")

    pd.DataFrame(summ).to_csv(OUT / f"expand_summary_v{a.vars}.csv", index=False, encoding="utf-8-sig")
    pd.concat(sweeps).to_csv(OUT / f"expand_sweep_v{a.vars}.csv", index=False, encoding="utf-8-sig")
    pd.concat(cis).to_csv(OUT / f"expand_ci_v{a.vars}.csv", index=False, encoding="utf-8-sig")
    pd.concat(m6s).to_csv(OUT / f"expand_models6_v{a.vars}.csv", index=False, encoding="utf-8-sig")

    sw = pd.concat(sweeps)
    piv = sw[sw.model == "LightGBM"].pivot_table(index=["market", "split", "month"],
                                                columns="exp", values="ap")
    piv["격차"] = piv["A"] - piv["B"]
    print("\n=== 시점 스윕 (LightGBM PR-AUC 격차) ===")
    print(piv["격차"].unstack("month").round(3).to_string())
    print("\n→ results/expand_*.csv")


if __name__ == "__main__":
    main()
