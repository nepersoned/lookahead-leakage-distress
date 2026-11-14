"""추가 강건성 두 가지.

1. 2019년 예측 제외 — 채점을 2020-2024로 한정한다. 확장창이라 각 연도의 학습 구간은 그대로이고
   (2020년 예측은 2017-2019로 학습), 2020-2024 예측값도 본문과 같다. 그래서 final27_preds.csv를 다시 채점한다.
2. SHAP 공시 변수 선별의 안정성 — 선별 구간(예측연도 2017-2018)을 기업 단위로 복원추출해 SHAP 상위 10을
   다시 고른다(B_SEL회). 변수별 선택 빈도와 본문 목록과의 겹침(Jaccard)을 보고, 그중 B_EVAL회는 고른 목록으로
   본문 설계를 다시 돌려 핵심 비교 두 개(B+시장 − A, A+시장 − B+시장)의 분포를 본다.
   희소 강제포함 변수는 규칙으로 정해지므로 본문과 같게 둔다. LightGBM 설정이 결정적이라 시드 변경은 하지 않는다.

출력: results/final35_excl2019.csv, final35_shap_freq.csv, final35_shap_eval.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final7 as f7
import final21 as f21
import final27 as f27
import final28 as f28
import disclosure_mock as dm
import expand_analysis as ea
from dart import Dart

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
B_SEL, B_EVAL = 100, 20


def ci_row(name, y, pa, pb, grp, rng):
    r = [x for x in ea.boot_ci(y, pa, pb, grp, 1000, rng) if x["지표"] == "PR-AUC"][0]
    f_ = r["0초과"]
    return {"비교": name, "격차": round(r["격차"], 4), "하한": round(r["하한"], 4), "상한": round(r["상한"], 4),
            "p": round(max(2 * min(f_, 1 - f_), 0.002), 3)}


def excl2019(rng):
    P = pd.read_csv(OUT / "final27_preds.csv", dtype={"corp_code": str})
    rows = []
    for tag, S in [("2019-2024 (본문)", P), ("2020-2024 (2019 제외)", P[P.pred_year >= 2020])]:
        S = S.reset_index(drop=True)
        y, grp = S.y.values, S.corp_code
        base = {"채점": tag, "행": len(S), "사건": int(y.sum()),
                **{f"{k} PR-AUC": round(float(dm.score(y, S[k].values)["PR-AUC"]), 4) for k in ("A", "B", "B+시장", "A+시장")}}
        for a, b, what in [("A", "B", "누출"), ("B+시장", "A", "정직 > 관행"), ("A+시장", "B+시장", "남는 격차")]:
            rows.append({**base, **ci_row(what, y, S[a].values, S[b].values, grp, rng)})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "final35_excl2019.csv", index=False, encoding="utf-8-sig")
    print(T.to_string(index=False), flush=True)


def shap_top(Xs, ys, cols, k=10):
    import lightgbm as lgb
    import shap
    lo, hi = Xs.quantile(.01), Xs.quantile(.99)
    Xc = Xs.clip(lo, hi, axis=1)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, min_child_samples=10,
                             class_weight="balanced", verbose=-1, random_state=42).fit(Xc, ys)
    sv = shap.TreeExplainer(clf).shap_values(Xc)
    sv = sv[1] if isinstance(sv, list) else sv
    return list(pd.Series(np.abs(sv).mean(0), index=Xc.columns)[cols].sort_values(ascending=False).index[:k])


def shap_stability(rng):
    _, pub, labels, mc, c2s = f7.load()
    ratios = pd.read_csv(ROOT / "data" / "full" / "classic_orig.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios.corp_code.str.zfill(8)
    A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    CB = f27.disclosure(M)
    H = f28.history(M, labels)
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s2 = (codes[codes.stock_code.str.len() == 6].drop_duplicates("corp_code")
            .set_index("corp_code")["stock_code"].to_dict())
    F = f21.features(M, f21.load_market("ohlcv", range(2014, 2025)), None, c2s2).reset_index(drop=True)

    main_top = list(pd.read_csv(OUT / "final27_shap.csv")["변수"][:10])
    sp = pd.read_csv(OUT / "final27_sparse.csv")
    forced = [c for c in sp[sp.강제포함].변수 if c not in main_top]
    print("본문 상위 10:", main_top, "\n강제포함:", forced, flush=True)

    sel = M.pred_year.isin(f27.SEL_YEARS).values
    XB = pd.concat([B, CB], axis=1)[sel].reset_index(drop=True)
    ys, firms = M.y.values[sel], M.corp_code.values[sel]
    uniq = np.unique(firms)
    pos = {f: np.where(firms == f)[0] for f in uniq}
    tops = []
    for i in range(B_SEL):
        rows = np.concatenate([pos[f] for f in rng.choice(uniq, len(uniq), replace=True)])
        tops.append(shap_top(XB.iloc[rows].reset_index(drop=True), ys[rows], list(CB.columns)))
        if (i + 1) % 20 == 0:
            print(f"  선별 부트스트랩 {i + 1}/{B_SEL}", flush=True)
    freq = pd.Series([v for t in tops for v in t]).value_counts().div(B_SEL).rename("상위10 선택 빈도")
    freq = freq.rename_axis("변수").reset_index()
    freq["본문 목록"] = freq.변수.isin(main_top)
    jac = [len(set(t) & set(main_top)) / len(set(t) | set(main_top)) for t in tops]
    freq.to_csv(OUT / "final35_shap_freq.csv", index=False, encoding="utf-8-sig")
    print(freq.head(20).to_string(index=False), flush=True)
    print(f"본문 목록과의 Jaccard: 평균 {np.mean(jac):.2f}, 중앙 {np.median(jac):.2f}, "
          f"본문 10개 중 평균 {np.mean([len(set(t) & set(main_top)) for t in tops]):.1f}개 일치", flush=True)

    te = M.pred_year.isin(f27.TEST).values
    y, grp = M.loc[te, "y"].values, M.loc[te, "corp_code"].reset_index(drop=True)
    pA = f7.rolling(A, M, f27.START, f27.TEST)
    ev = []
    for i, t in enumerate(tops[:B_EVAL]):
        cols = t + [c for c in forced if c not in t]
        bp = pd.concat([B, CB[cols], H, F], axis=1)
        apx = pd.concat([A, CB[cols], H, F], axis=1)
        pb, pa = f7.rolling(bp, M, f27.START, f27.TEST), f7.rolling(apx, M, f27.START, f27.TEST)
        sb, sa, s0 = (float(dm.score(y, p)["PR-AUC"]) for p in (pb, pa, pA))
        ev.append({"반복": i, "본문과 겹침": len(set(t) & set(main_top)), "B+시장": sb, "A+시장": sa, "A": s0,
                   "B+시장 − A": sb - s0, "A+시장 − B+시장": sa - sb})
        print(ev[-1], flush=True)
    E = pd.DataFrame(ev).round(4)
    E.to_csv(OUT / "final35_shap_eval.csv", index=False, encoding="utf-8-sig")
    for c in ("B+시장 − A", "A+시장 − B+시장"):
        print(f"{c}: 평균 {E[c].mean():+.3f}, 최소 {E[c].min():+.3f}, 최대 {E[c].max():+.3f}, 양수 {int((E[c] > 0).sum())}/{len(E)}")


def main():
    rng = np.random.default_rng(42)
    excl2019(rng)
    shap_stability(rng)


if __name__ == "__main__":
    main()
