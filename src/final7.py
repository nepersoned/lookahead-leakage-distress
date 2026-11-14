"""최종 실험 v4 — final6 + 검토 반영.

1. 수시공시 정규식 순서: embezzle을 ceo 위로 (횡령·배임 제목의 '대표이사'가 ceo로 먹혔다),
   acquire의 '분할'에서 주식분할·액면분할 제외
2. 시가총액 시점: 재무제표 회계연도 12월 말이 아니라 **T 직전 월말**을 A·B 모두에 쓴다.
   B도 T 시점의 최신 시장 정보를 갖는다 — 재무제표만 낡았다
3. --start 2018: INTWO·CHIN이 비는 2017년을 학습에서 뺀 강건성 판
4. --sweep: 예측 시점 1~7월 A·B 스윕 (본문 설계 그대로)

사용: python src/final7.py [--start 2017] [--boot 1000] [--sweep]
출력: results/final7_s{start}_*.csv, results/final7_sweep.csv
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

import disclosure_mock as dm
import expand_analysis as ea
import feature_screen as fs
import gate3e_full as g
from dart import Dart
from final4 import BASE

warnings.filterwarnings("ignore")

C = [(c, p) for c, p in fs.CATS]
_emb = C.pop([c for c, _ in C].index("embezzle"))
C.insert([c for c, _ in C].index("ceo"), _emb)
C = [(c, r"(타법인주식|출자증권|영업|유형자산)\s*.*양수|합병|(?<!액면)(?<!주식)분할")
     if c == "acquire" else (c, p) for c, p in C]
fs.CATS = C
import final5 as f5  # noqa: E402  (fs.CATS 교체 뒤에 불러야 BEHAV/RULE이 새 순서를 쓴다)

f5.CATS = [(c, p) for c, p in C if c != "pledge"]
_i = [c for c, _ in f5.CATS].index("loan")
f5.CATS[_i:_i] = [("guarantee", r"채무보증"), ("pledge", r"담보제공")]
fs.CATS = f5.CATS
f5.BEHAV = [c for c, _ in f5.CATS if not c.startswith("L_") and c != "contract"]
f5.RULE = [c for c, _ in f5.CATS if c.startswith("L_")]

K = 10


def load():
    ratios, pub, labels = ea.load_market("KOSDAQ", "classic")
    o = ratios
    ratios["O_SCORE"] = (-1.32 - 0.407 * o.O_SIZE + 6.03 * o.O_TLTA - 1.43 * o.O_WCTA
                         + 0.0757 * o.O_CLCA - 1.72 * o.O_OENEG - 2.37 * o.O_NITA
                         - 1.83 * o.O_FUTL_CFO + 0.285 * o.O_INTWO - 0.521 * o.O_CHIN)
    mc = pd.read_csv(ea.FULL / "marketcap.csv", dtype={"ticker": str, "date": str})
    mc["ticker"] = mc.ticker.str.zfill(6)
    mc["dt"] = pd.to_datetime(mc.date, format="%Y%m%d")
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s = (codes[codes.stock_code.str.len() == 6]
           .drop_duplicates("corp_code").set_index("corp_code")["stock_code"])
    return ratios, pub, labels, mc, c2s


def refresh_mcap(X, M, ratios, mc, c2s, fy_col, month):
    """K_METL을 T 직전 월말 시가총액으로 다시 쓴다. 총부채는 원래 비율에서 되돌린다."""
    X = X.copy()
    fy_mc = mc[mc.dt.dt.month == 12].assign(year=lambda d: d.dt.dt.year)
    old = {(t, y): v for t, y, v in zip(fy_mc.ticker, fy_mc.year, fy_mc.mcap)}
    T = pd.to_datetime(dict(year=M.pred_year, month=month, day=1))
    snap = T - pd.offsets.MonthEnd(1)
    cur = {(t, d): v for t, d, v in zip(mc.ticker, mc.dt.dt.to_period("M"), mc.mcap)}
    tick = M.corp_code.map(c2s)
    new_metl = []
    for i, (tk, fy, s) in enumerate(zip(tick, fy_col, snap)):
        m_old, m_new = old.get((tk, fy)), cur.get((tk, s.to_period("M")))
        k = X.at[i, "K_METL"]
        new_metl.append(k * m_new / m_old if m_old and m_new and pd.notna(k) else np.nan)
    new_metl = pd.Series(new_metl)
    X["K_METL"] = new_metl
    # K-Score: 상장 기업이므로 원문의 K2(시가총액/총부채)를 원문 계수 그대로 쓴다.
    # Altman, Eom & Kim (1994), NYU Salomon Center WP FIN-94-002, p.9.
    # (2026-11-14 이전 판은 K1 계수 −17.9·1.5·3.0·14.8·1.5에 K2의 시가총액 변수를 섞어 썼다 — 재현 오류, final52)
    X["KSCORE"] = (-18.696 + 1.501 * X.K_lnTA + 2.706 * X.K_lnSLTA
                   + 19.760 * X.K_RETA + 1.146 * X.K_METL)
    return X


def design(ratios, pub, labels, mc, c2s, years, month):
    g.RATIO_COLS = BASE
    A, B, M = g.build(ratios, pub, labels, years, month)
    fy_a = M.pred_year - 1
    fy_b = fy_a - M.lag
    return (refresh_mcap(A, M, ratios, mc, c2s, fy_a, month),
            refresh_mcap(B, M, ratios, mc, c2s, fy_b, month), M)


def rolling(X, M, start, test, seed=42):
    p = np.full(len(M), np.nan)
    for y in test:
        g.TRAIN_YEARS, g.TEST_YEARS = list(range(start, y)), [y]
        _, _, py, _ = fs.fit_predict(X, M, seed)
        p[M.pred_year.values == y] = py
    return p[M.pred_year.isin(test).values]


def main_run(a, data, rng):
    ratios, pub, labels, mc, c2s = data
    sel_years = [a.start, a.start + 1]
    test = list(range(a.start + 2, 2025))
    A, B, M = design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    out = f"results/final7_s{a.start}_"
    print(f"\n=== 학습 시작 {a.start} / 선별 {sel_years} / 채점 {test[0]}~{test[-1]}", flush=True)
    print("K_METL 결측 A/B:", round(A.K_METL.isna().mean(), 3), round(B.K_METL.isna().mean(), 3))

    orig, amend = dm.load_filings()
    aud = dm.load_auditors()
    f = fs.load_filings_all()
    f["cat"] = fs.classify(f["report_nm"])
    D = dm.features(M, 1, orig, amend, aud).drop(columns=["amend_1y", "amend_3y"])
    Eb = fs.event_features(M, 1, f, f5.BEHAV).drop(columns=["amend_any_1y", "amend_any_3y", "amend_share_3y"])
    CB = pd.concat([D, Eb, f5.extra_features(M, f, orig)], axis=1).reset_index(drop=True)
    CB = CB.loc[:, CB.notna().any() & (CB.nunique() > 1)]
    Er = fs.event_features(M, 1, f, f5.RULE).reset_index(drop=True)
    CR = Er[[c for c in Er.columns if c.startswith("L_")]]
    CR = CR.loc[:, CR.nunique() > 1]

    import shap
    g.TRAIN_YEARS, g.TEST_YEARS = sel_years, [test[0]]
    clf, Xtr, _, _ = fs.fit_predict(pd.concat([B, CB], axis=1), M)
    sv = shap.TreeExplainer(clf).shap_values(Xtr)
    sv = sv[1] if isinstance(sv, list) else sv
    rank = pd.Series(np.abs(sv).mean(0), index=Xtr.columns)[CB.columns].sort_values(ascending=False)
    rank.rename("mean_abs_shap").to_csv(out + "shap.csv", encoding="utf-8-sig")
    top = list(rank.index[:K])
    f5.SEL_YEARS = sel_years
    counts = [c for c in CB.columns if c.endswith(("_1y", "_3y")) and "share" not in c]
    sp = f5.sparse_table(CB, M, counts)
    n_sel = int(M.pred_year.isin(sel_years).sum())
    sp["강제포함"] = (sp.켜진표본 >= 30) & (sp.켜진표본 <= .1 * n_sel) & (sp.하한 > 1)
    sp.to_csv(out + "sparse.csv", index=False, encoding="utf-8-sig")
    sel = top + [c for c in sp[sp.강제포함].변수 if c not in top]
    print("SHAP 상위:", top, "\n강제포함:", sel[K:], flush=True)
    print(sp[sp.변수.str.startswith(("embezzle", "acquire", "ceo"))].to_string(), flush=True)

    conds = {"A": A, "A+": pd.concat([A, CB[sel], CR], axis=1),
             "B": B, "B+": pd.concat([B, CB[sel], CR], axis=1),
             "B+행동": pd.concat([B, CB[sel]], axis=1)}
    te = M.pred_year.isin(test).values
    y, yr = M.loc[te, "y"].values, M.loc[te, "pred_year"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    P, rows = {}, []
    for name, X in conds.items():
        P[name] = rolling(X, M, a.start, test)
        rows.append({"조건": name, "행": int(te.sum()), "사건": int(y.sum()),
                     **{k: round(float(v), 4) for k, v in dm.score(y, P[name]).items()}})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out + "score.csv", index=False, encoding="utf-8-sig")
    yrows = [{"연도": yy, "조건": n, **{k: round(float(v), 4) for k, v in dm.score(y[yr == yy], p[yr == yy]).items()}}
             for yy in test for n, p in P.items()]
    by = pd.DataFrame(yrows)
    by.to_csv(out + "by_year.csv", index=False, encoding="utf-8-sig")
    print(by.pivot(index="연도", columns="조건", values="PR-AUC").round(3).to_string())
    cis = []
    for x1, x2, what in [("A", "B", "누출 효과"), ("B+", "B", "공시변수 기여"), ("B+", "A", "B+가 관행을 넘는가"),
                         ("A+", "B+", "같은 변수 뒤 격차"), ("A+", "A", "참고"),
                         ("B+행동", "B", "행동만"), ("B+", "B+행동", "규정 추가")]:
        for r in ea.boot_ci(y, P[x1], P[x2], grp, a.boot, rng):
            cis.append({"비교": f"{x1}−{x2}", "의미": what, **r})
    ci = pd.DataFrame(cis)
    ci.to_csv(out + "ci.csv", index=False, encoding="utf-8-sig")
    print(ci.to_string(), flush=True)


def sweep(a, data, rng):
    ratios, pub, labels, mc, c2s = data
    test = list(range(a.start + 2, 2025))
    rows, cis = [], []
    for month in range(1, 8):
        A, B, M = design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), month)
        te = M.pred_year.isin(test).values
        y = M.loc[te, "y"].values
        pa, pb = rolling(A, M, a.start, test), rolling(B, M, a.start, test)
        sa, sb = dm.score(y, pa), dm.score(y, pb)
        rows.append({"월": month, "행": int(te.sum()), "사건": int(y.sum()),
                     "lag>0 비율": round(float((M.loc[te, "lag"] > 0).mean()), 3),
                     **{f"A_{k}": round(float(v), 4) for k, v in sa.items()},
                     **{f"B_{k}": round(float(v), 4) for k, v in sb.items()}})
        for r in ea.boot_ci(y, pa, pb, M.loc[te, "corp_code"].reset_index(drop=True), a.boot, rng):
            cis.append({"월": month, **r})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/final7_sweep.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv("results/final7_sweep_ci.csv", index=False, encoding="utf-8-sig")
    print(ci[ci.지표.isin(["PR-AUC", "AUC", "상위50"])].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2017)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    data = load()
    (sweep if a.sweep else main_run)(a, data, rng)
