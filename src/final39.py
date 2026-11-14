"""final39 — 2025년 홀드아웃 뒤의 탐색적·보조 분석 (사전 계획에 없음, 결과를 본 뒤 수행).

(1) 탐색적: 2025년 채점 사건을 사유군·지정 월로 나누어 A, B, B+, B+시장의 상위 100위 적중을 본다.
    사유군은 KIND 공시 제목의 사유 분류(kind.py classify_reason)로 나눈다. 2025년에는 본문의 세부 사유
    (final15_subreason)가 없으므로 '상장폐지사유'는 기타로 남는다.
(2) 보조: 예측 연도 2019–2025를 본문과 같은 확장 창으로 모아 핵심 비교 세 개를 다시 계산한다.
출력: results/final39_groups.csv, final39_score.csv, final39_ci.csv, final39.log
실행: 저장소 루트에서 `python src/final39.py`
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final38 as f38  # MARKETS 확장(2025 자료)을 함께 적용한다
import final7 as f7
import final27 as f27
import final28 as f28
import final29 as f29
import disclosure_mock as dm
import expand_analysis as ea
from dart import Dart

warnings.filterwarnings("ignore")
FULL = ea.FULL
START, TEST = 2017, list(range(2019, 2026))


def group_of(reason: str) -> str:
    r = str(reason)
    if any(k in r for k in ("감사", "자본잠식", "손실", "매출액", "미제출", "사업보고서")):
        return "감사·결산 확정"
    if any(k in r for k in ("회생", "파산")):
        return "법적 절차"
    if any(k in r for k in ("실질", "적격성", "적경성")):
        return "거래소 재량"
    return "기타·불명"


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    r0 = pd.read_csv(FULL / "classic_orig.csv", dtype={"corp_code": str})
    r1 = pd.read_csv(FULL / "classic_orig_fy2024.csv", dtype={"corp_code": str})
    ratios = pd.concat([r0, r1[r0.columns.intersection(r1.columns)]], ignore_index=True)
    ratios["corp_code"] = ratios.corp_code.str.zfill(8)
    ratios = ratios.drop_duplicates(["corp_code", "bsns_year"], keep="first")
    A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2026)), 1)
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
    plus = lambda base, mkt: pd.concat([base, CB[sel], H] + ([F] if mkt else []), axis=1)
    X = {"A": A, "B": B, "B+": plus(B, False), "A+": plus(A, False),
         "B+시장": plus(B, True), "A+시장": plus(A, True)}
    P = {k: f7.rolling(v, M, START, TEST) for k, v in X.items()}
    te = M.pred_year.isin(TEST).values
    Mt = M[te].reset_index(drop=True)
    y = Mt.y.values
    grp = Mt.corp_code

    # (2) 2019–2025 전수
    scores = [{"조건": k, **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}} for k, p in P.items()]
    cis = []
    for a, b, what in [("A", "B", "누출"), ("B+시장", "A", "B+시장 − A"), ("A+시장", "B+시장", "남는 격차")]:
        for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            f_ = r["0초과"]
            cis.append({"비교": what, **r, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    yr = []
    for Y in TEST:
        m = (Mt.pred_year == Y).values
        yr.append({"연도": Y, "사건": int(y[m].sum()),
                   **{k: round(float(dm.score(y[m], p[m])["PR-AUC"]), 4) for k, p in P.items()}})

    # (1) 2025 사건 구성 (탐색적)
    kind = pd.read_csv(FULL.parent / "raw" / "kind_manual" / "kind_0350_kosdaq_2025.csv", dtype=str)
    kind = kind[kind.event == "지정"].copy()
    lab25 = labels[labels.event_dt.dt.year == 2025][["corp_code", "event_dt"]]
    m25 = (Mt.pred_year == 2025).values
    ev = Mt[m25 & (Mt.y == 1).values][["corp_code"]].merge(lab25, on="corp_code", how="left")
    kind["rcept_dt"] = pd.to_datetime(kind.rcept_dt.str[:19], errors="coerce")
    ev = ev.merge(kind[["rcept_dt", "reason"]].rename(columns={"rcept_dt": "event_dt"}), on="event_dt", how="left")
    ev["사유군"] = ev.reason.map(group_of)
    ev["월"] = ev.event_dt.dt.month
    rank = {}
    idx25 = np.where(m25)[0]
    for k in ("A", "B", "B+", "B+시장"):
        p = P[k][idx25]
        order = pd.Series(p).rank(ascending=False, method="first").values
        rank[k] = dict(zip(Mt.corp_code.values[idx25], order))
    for k in rank:
        ev[f"{k} 상위100"] = ev.corp_code.map(rank[k]).le(100)
    g1 = ev.groupby("사유군").agg(사건=("corp_code", "size"), **{f"{k}": (f"{k} 상위100", "sum") for k in rank})
    g2 = ev.assign(월구간=pd.cut(ev.월, [0, 3, 7, 12], labels=["1–3월", "4–7월", "8–12월"])) \
        .groupby("월구간").agg(사건=("corp_code", "size"), **{f"{k}": (f"{k} 상위100", "sum") for k in rank})

    pd.DataFrame(scores).to_csv("results/final39_score.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cis).to_csv("results/final39_ci.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yr).to_csv("results/final39_by_year.csv", index=False, encoding="utf-8-sig")
    pd.concat([g1.assign(구분="사유군"), g2.assign(구분="월구간")]).to_csv("results/final39_groups.csv", encoding="utf-8-sig")
    print("== 2019–2025 전수"); print(pd.DataFrame(scores).to_string())
    print(pd.DataFrame(cis)[lambda d: d.지표 == "PR-AUC"].to_string())
    print("== 연도별 PR-AUC"); print(pd.DataFrame(yr).to_string())
    print("== 2025 사건 구성 (상위 100위 적중 수)"); print(g1.to_string()); print(g2.to_string())
    print("사유 미연결:", int(ev.reason.isna().sum()))


if __name__ == "__main__":
    main()
