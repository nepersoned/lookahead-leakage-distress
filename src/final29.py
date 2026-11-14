"""효과가 없던 시도들을 최종 설계(final27)에서 다시 잰다.

기준선: 원문판 재무 + 공시 16 + 지정 이력 3 + 시장 10 (B+시장 · A+시장), 고친 모집단.
각 시도 X를 **양쪽에 똑같이** 더하고 두 가지를 본다.
    ① (B+시장 + X) − B+시장          정직 모형이 좋아지나
    ② (A+시장 + X) − (B+시장 + X)    남는 격차(누출)가 더 희석되나

시도
    리픽싱·최대주주 변경 담보제공                         final16 범주
    장 마감 후 공시                                        final14.features

출력: results/final29_score.csv, final29_ci.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final7 as f7
import final5 as f5
import final21 as f21
import final27 as f27
import final28 as f28
import disclosure_mock as dm
import expand_analysis as ea
import feature_screen as fs
import gate3e_full as g
from dart import Dart

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
START, TEST = 2017, list(range(2019, 2025))


def o_score(d):
    return (-1.32 - 0.407 * d.O_SIZE + 6.03 * d.O_TLTA - 1.43 * d.O_WCTA + 0.0757 * d.O_CLCA
            - 1.72 * d.O_OENEG - 2.37 * d.O_NITA - 1.83 * d.O_FUTL_CFO + 0.285 * d.O_INTWO
            - 0.521 * d.O_CHIN)


def market(M, c2s2):
    return f21.features(M, f21.load_market("ohlcv", range(2014, 2025)), None, c2s2).reset_index(drop=True)


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    ratios = pd.read_csv(FULL / "classic_orig.csv", dtype={"corp_code": str})
    ratios["corp_code"] = ratios.corp_code.str.zfill(8)
    A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    te = M.pred_year.isin(TEST).values
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)

    CB = f27.disclosure(M)
    top = list(pd.read_csv("results/final27_shap.csv")["변수"][:10])
    sp = pd.read_csv("results/final27_sparse.csv")
    sel = top + [c for c in sp[sp.강제포함].변수 if c not in top]
    H = f28.history(M, labels)
    codes = pd.DataFrame(Dart().corp_codes())
    codes["stock_code"] = codes.stock_code.astype(str).str.strip().str.zfill(6)
    c2s2 = (codes[codes.stock_code.str.len() == 6].drop_duplicates("corp_code")
            .set_index("corp_code")["stock_code"].to_dict())
    F = market(M, c2s2)
    BP = pd.concat([B, CB[sel], H, F], axis=1)
    AP = pd.concat([A, CB[sel], H, F], axis=1)

    # ── 시도별 변수
    X = {}
    import final14 as f14
    _, _, lab = ea.load_market("KOSDAQ", "classic")
    X["장 마감 후 공시"] = f14.features(M, f14.load_exchange()).reset_index(drop=True)
    # 임원 네트워크 후보(final18)는 개인정보가 담긴 임원 명단을 읽으므로 공개판에서 뺐다. 원고에 쓰이지 않는다.
    import final16  # noqa: F401  (범주 추가 — CB를 만든 뒤에 불러야 기준선이 바뀌지 않는다)
    fr = fs.load_filings_all()
    fr["cat"] = fs.classify(fr["report_nm"])
    X["리픽싱·담보제공 분리"] = fs.event_features(M, 1, fr, ["refix", "owner_pledge"]).reset_index(drop=True)
    for k in list(X):
        X[k] = X[k].loc[:, ~X[k].columns.isin(BP.columns)]      # 기준선과 겹치는 열(예: all_filings_1y)은 뺀다
        v = X[k]
        print(f"{k}: 변수 {v.shape[1]} / 채점 결측 {v[te].isna().all(axis=1).mean():.1%}", flush=True)

    # ── 점수
    P = {"B+시장": f7.rolling(BP, M, START, TEST), "A+시장": f7.rolling(AP, M, START, TEST)}
    for k, v in X.items():
        P[f"B+시장 + {k}"] = f7.rolling(pd.concat([BP, v], axis=1), M, START, TEST)
        P[f"A+시장 + {k}"] = f7.rolling(pd.concat([AP, v], axis=1), M, START, TEST)
        print(k, round(float(dm.score(y, P[f"B+시장 + {k}"])["PR-AUC"]), 4),
              round(float(dm.score(y, P[f"A+시장 + {k}"])["PR-AUC"]), 4), flush=True)

    rows = [{"조건": k, **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}} for k, p in P.items()]
    pd.DataFrame(rows).to_csv("results/final29_score.csv", index=False, encoding="utf-8-sig")
    cis = []
    for k in X:
        for a, b, what in [(f"B+시장 + {k}", "B+시장", "정직 모형 기여"),
                           (f"A+시장 + {k}", f"B+시장 + {k}", "남는 격차")]:
            for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
                f_ = r["0초과"]
                cis.append({"시도": k, "비교": what, **r, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})
    ci = pd.DataFrame(cis)
    ci.to_csv("results/final29_ci.csv", index=False, encoding="utf-8-sig")
    print(ci[ci.지표 == "PR-AUC"].to_string(), flush=True)


if __name__ == "__main__":
    main()
