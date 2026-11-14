"""심사 의견에 대한 강건성 — 본문 설계(final27)를 한 가지씩 바꿔 핵심 비교 세 개를 다시 잰다.

    누출            A − B
    정직 > 관행     B+시장 − A
    남는 격차       A+시장 − B+시장

변형
    본문                 final27과 같음 (재현 확인)
    영업현금흐름 제외     O_FUTL_CFO와 그것이 들어간 O_SCORE를 A·B에서 뺀다 (API 정정본 의존 제거)
    로지스틱 회귀         LightGBM 대신 L2 로지스틱 (중앙값 대치 · 표준화 · class_weight balanced)
    재무적 사유만         감사·결산 확정 사유(감사의견·자본잠식·손실지속·매출액미달·보고서미제출)만 사건.
                         다른 사유로 지정된 기업·연도는 제외
    산업 더미             KSIC 대분류 더미를 모든 조건에 추가
    금융업·스팩 제외      KSIC 64~66과 스팩을 모집단에서 뺀다
    선별 없음             SHAP 선별 대신 공시 후보 전부

출력: results/final30_score.csv, final30_ci.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final7 as f7
import final27 as f27
import final28 as f28
import final29 as f29
import disclosure_mock as dm
import expand_analysis as ea
import feature_screen as fs
import gate3e_full as g
from dart import Dart

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
START, TEST = 2017, list(range(2019, 2025))
FIN_GROUP = "감사·결산 확정"

KSIC = [("A", 1, 3), ("B", 5, 8), ("C", 10, 34), ("D", 35, 35), ("E", 36, 39), ("F", 41, 42),
        ("G", 45, 47), ("H", 49, 52), ("I", 55, 56), ("J", 58, 63), ("K", 64, 66), ("L", 68, 68),
        ("M", 70, 73), ("N", 74, 76), ("O", 84, 84), ("P", 85, 85), ("Q", 86, 87), ("R", 90, 91),
        ("S", 94, 96)]


def section(d2):
    try:
        d2 = int(d2)
    except (TypeError, ValueError):
        return "?"
    return next((s for s, lo, hi in KSIC if lo <= d2 <= hi), "?")


def logit_rolling(X, M):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    yr, yv = M.pred_year.values, M.y.values
    p = np.full(len(M), np.nan)
    for Y in TEST:
        tr, te = (yr >= START) & (yr < Y), yr == Y
        lo, hi = X[tr].quantile(.01), X[tr].quantile(.99)
        Xc = X.clip(lo, hi, axis=1).replace([np.inf, -np.inf], np.nan)
        keep = Xc.columns[Xc[tr].notna().any()]
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))
        m.fit(Xc.loc[tr, keep], yv[tr])
        p[te] = m.predict_proba(Xc.loc[te, keep])[:, 1]
    return p[np.isin(yr, TEST)]


def evaluate(name, A, B, BP, AP, M, rng, rows, cis, model="lgbm"):
    run = (lambda X: f7.rolling(X.reset_index(drop=True), M.reset_index(drop=True), START, TEST)) \
        if model == "lgbm" else (lambda X: logit_rolling(X.reset_index(drop=True), M.reset_index(drop=True)))
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    P = {"A": run(A), "B": run(B), "B+시장": run(BP), "A+시장": run(AP)}
    for k, p in P.items():
        rows.append({"변형": name, "조건": k, "채점": int(te.sum()), "사건": int(y.sum()),
                     **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}})
    for a, b, what in [("A", "B", "누출"), ("B+시장", "A", "정직 > 관행"), ("A+시장", "B+시장", "남는 격차")]:
        for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            f_ = r["0초과"]
            cis.append({"변형": name, "비교": what, **r, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    print(name, {k: round(float(dm.score(y, p)["PR-AUC"]), 4) for k, p in P.items()}, flush=True)


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    ratios = pd.read_csv(FULL / "classic_orig.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios.corp_code.str.zfill(8)
    A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    A, B, M = A.reset_index(drop=True), B.reset_index(drop=True), M.reset_index(drop=True)

    CB = f27.disclosure(M)
    top = list(pd.read_csv("results/final27_shap.csv")["변수"][:10])
    sp = pd.read_csv("results/final27_sparse.csv")
    sel = top + [c for c in sp[sp.강제포함].변수 if c not in top]
    H = f28.history(M, labels)
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s2 = (codes[codes.stock_code.str.len() == 6].drop_duplicates("corp_code")
            .set_index("corp_code")["stock_code"].to_dict())
    F = f29.market(M, c2s2)

    def plus(base, cb=None, extra=None):
        parts = [base, CB[sel] if cb is None else cb, H, F] + ([extra] if extra is not None else [])
        return pd.concat(parts, axis=1)

    rows, cis = [], []
    evaluate("본문", A, B, plus(B), plus(A), M, rng, rows, cis)

    drop = ["O_FUTL_CFO", "O_SCORE"]
    evaluate("영업현금흐름 제외", A.drop(columns=drop), B.drop(columns=drop),
             plus(B.drop(columns=drop)), plus(A.drop(columns=drop)), M, rng, rows, cis)

    evaluate("로지스틱 회귀", A, B, plus(B), plus(A), M, rng, rows, cis, model="logit")

    ev = f27.reason_groups(M[M.y == 1][["corp_code", "pred_year", "y"]].copy())
    other = set(zip(ev.loc[ev.사유군 != FIN_GROUP, "corp_code"], ev.loc[ev.사유군 != FIN_GROUP, "pred_year"]))
    keep = ~np.array([(c, yy) in other for c, yy in zip(M.corp_code, M.pred_year)])
    print(f"재무적 사유 사건 {int((M.y[keep]).sum())} / 다른 사유로 빠진 행 {int((~keep).sum())}", flush=True)
    k = keep
    evaluate("재무적 사유만", A[k], B[k], plus(B)[k], plus(A)[k], M[k], rng, rows, cis)

    ind = pd.read_csv(FULL / "industry.csv", dtype=str)
    ind["corp_code"] = ind.corp_code.str.zfill(8)
    sec = M.corp_code.map(dict(zip(ind.corp_code, ind.induty_2.map(section)))).fillna("?")
    D = pd.get_dummies(sec, prefix="ksic", dtype=float)
    evaluate("산업 더미", A.join(D), B.join(D), plus(B, extra=D), plus(A, extra=D), M, rng, rows, cis)

    names = dict(zip(ind.corp_code, ind.corp_name))
    spac = M.corp_code.map(names).fillna("").str.contains("스팩|기업인수목적")
    fin = sec == "K"
    k = ~(spac | fin).values
    print(f"금융업 {int(fin.sum())}행 · 스팩 {int(spac.sum())}행 제외", flush=True)
    evaluate("금융업·스팩 제외", A[k], B[k], plus(B)[k], plus(A)[k], M[k], rng, rows, cis)

    evaluate("선별 없음 (공시 후보 전부)", A, B, plus(B, cb=CB), plus(A, cb=CB), M, rng, rows, cis)

    pd.DataFrame(rows).to_csv("results/final30_score.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv("results/final30_ci.csv", index=False, encoding="utf-8-sig")
    print(ci[ci.지표 == "PR-AUC"].to_string(), flush=True)


if __name__ == "__main__":
    main()
