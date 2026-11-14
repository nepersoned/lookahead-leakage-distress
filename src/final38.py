"""final38 — 2025년 홀드아웃 (사전 계획: results/holdout2025_plan.md, 커밋 d1d1479).

본문(final27/final30)과 같은 설계·변수·모형으로 예측 연도 2017–2024를 학습하고, 2025년 1월 1일 예측을 한 번만 채점한다.
추가 자료(덮어쓰지 않고 이어 붙임):
    data/full/reports_full_2025.csv          2025년 접수 정기보고서(FY2024 사업보고서 접수일)
    data/full/classic_orig_fy2024.csv        FY2024 최초 공시본 기준선 변수
    data/raw/kind_manual/kind_0350_kosdaq_2025.csv   2025년 KIND 관리종목 공시(2024년분 100% 대조 통과)
보고(이것만): 여섯 조건 성능, 핵심 비교 세 개의 부트스트랩 CI, 표본·빠진 행.
출력: results/final38_score.csv, final38_ci.csv, final38_sample.csv, final38.log
실행: 저장소 루트에서 `python src/final38.py`
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import expand_analysis as ea

# 1) 자료 경로를 2025까지 확장 (모든 하위 모듈이 같은 MARKETS를 쓰도록 import 전에 바꾼다)
ea.MARKETS["KOSDAQ"]["reports"] = ea.MARKETS["KOSDAQ"]["reports"] + ["reports_full_2025.csv"]
ea.MARKETS["KOSDAQ"]["labels"] = ea.MARKETS["KOSDAQ"]["labels"] + ["../raw/kind_manual/kind_0350_kosdaq_2025.csv"]

import final7 as f7          # noqa: E402
import final27 as f27        # noqa: E402
import final28 as f28        # noqa: E402
import final29 as f29        # noqa: E402
import disclosure_mock as dm  # noqa: E402
from dart import Dart        # noqa: E402

warnings.filterwarnings("ignore")
FULL = ea.FULL
START, TEST = 2017, [2025]


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
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)

    scores = [{"조건": k, **{a: round(float(b), 4) for a, b in dm.score(y, p).items()}} for k, p in P.items()]
    cis = []
    for a, b, what in [("A", "B", "누출"), ("B+시장", "A", "B+시장 − A"), ("A+시장", "B+시장", "남는 격차")]:
        for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
            f_ = r["0초과"]
            cis.append({"비교": what, **r, "p(양측)": round(max(2 * min(f_, 1 - f_), 0.002), 3)})

    # 표본: 2025년 지정 기업 중 채점에 들어간 수, A가 없어 빠진 기업
    lab25 = labels[labels.event_dt.dt.year == 2025]
    scored = set(M.loc[te, "corp_code"])
    sample = {"채점 행": int(te.sum()), "채점 사건": int(y.sum()),
              "2025 KIND 지정 기업(코드 연결)": int(lab25.corp_code.nunique()),
              "그중 채점 표본 밖": int(len(set(lab25.corp_code) - scored)),
              "FY2024 기준선 행": int(len(r1))}
    pd.DataFrame(scores).to_csv("results/final38_score.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cis).to_csv("results/final38_ci.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([sample]).to_csv("results/final38_sample.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(scores).to_string())
    print(pd.DataFrame(cis)[lambda d: d.지표 == "PR-AUC"].to_string())
    print(sample)


if __name__ == "__main__":
    main()
