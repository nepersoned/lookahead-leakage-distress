"""장 마감 후 공시 변수 — 접수번호 일별 순위로 '늦게 낸 공시'를 잰다.

부록 T: 거래소공시(접수번호 일련번호 90/60/70 대역)는 같은 날·같은 대역 안에서 일련번호와
게시 시각의 순위상관이 ρ≈0.995~0.999다. DART 직접 제출분(00 대역)은 ρ≈0.83~0.87이라 쓰지
않는다.

절대 시각(15:30)으로 자르려면 날짜마다 컷오프를 실제 시각으로 맞춰야 한다. 대신
**그날 그 대역 안에서의 백분위**를 쓴다. 보정이 필요 없고, 뜻도 같다 — 그날 늦게 낸 공시다.

변수 (모두 rcept_dt < T)
    late_share_1y/3y    거래소공시 중 그날 백분위 0.8 이상인 비율
    late_cnt_1y/3y      그 건수
    pct_mean_1y/3y      백분위 평균
    late_fri_3y         금요일 + 그날 후반부 (악재 몰아내기)
    late_bad_3y         위험 범주(감자·소송·횡령·최대주주 변경·유상증자·CB) 중 후반부 비율

출력: results/final14_score.csv, final14_ci.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final8 as f8
import feature_screen as fs
import disclosure_mock as dm
import expand_analysis as ea

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TEST = f8.TEST
RISK = {"capred", "lawsuit", "embezzle", "owner", "rights", "cb", "asset_sale", "small_offer"}


def load_exchange():
    """거래소공시만 골라 그날·그 대역 안의 백분위를 매긴다."""
    f = pd.read_csv(ROOT / "data" / "full" / "filings_all.csv",
                    dtype={"corp_code": str, "rcept_no": str})
    f["corp_code"] = f.corp_code.str.zfill(8)
    f["rcept_dt"] = pd.to_datetime(f.rcept_dt.astype(str), format="%Y%m%d", errors="coerce")
    f["block"] = f.rcept_no.str[8:10]
    ex = f[f.block.isin(["90", "60", "70"])].copy()
    ex["serial"] = ex.rcept_no.str[8:].astype(int)
    # 같은 날·같은 대역 안에서의 백분위 (0=그날 첫 공시, 1=마지막)
    g = ex.groupby([ex.rcept_dt, "block"])["serial"]
    ex["pct"] = g.rank(pct=True)
    ex["n_day"] = g.transform("size")
    ex = ex[ex.n_day >= 5]                      # 표본이 너무 적은 날은 백분위가 무의미
    ex["cat"] = fs.classify(ex.report_nm)
    ex["late"] = ex.pct >= 0.8
    ex["fri"] = ex.rcept_dt.dt.weekday == 4
    print(f"거래소공시 {len(ex):,}건 / 날짜 {ex.rcept_dt.nunique():,}일 "
          f"/ 후반부 비율 {ex.late.mean():.1%}", flush=True)
    return ex


def features(M, ex):
    by = {c: d for c, d in ex.groupby("corp_code")}
    rows = []
    for code, y in zip(M.corp_code, M.pred_year):
        T = pd.Timestamp(year=y, month=1, day=1)
        d = by.get(code)
        r = {}
        if d is not None:
            for lab, yrs in (("1y", 1), ("3y", 3)):
                w = d[(d.rcept_dt < T) & (d.rcept_dt >= T - pd.DateOffset(years=yrs))]
                r[f"ex_cnt_{lab}"] = float(len(w))
                if len(w):
                    r[f"late_share_{lab}"] = float(w.late.mean())
                    r[f"late_cnt_{lab}"] = float(w.late.sum())
                    r[f"pct_mean_{lab}"] = float(w.pct.mean())
                    if lab == "3y":
                        r["late_fri_3y"] = float((w.late & w.fri).mean())
                        bad = w[w.cat.isin(RISK)]
                        r["late_bad_3y"] = float(bad.late.mean()) if len(bad) else np.nan
                        r["bad_cnt_3y"] = float(len(bad))
        rows.append(r)
    return pd.DataFrame(rows, index=M.index)


def main():
    rng = np.random.default_rng(42)
    ex = load_exchange()
    A, B, M, CB, CR = f8.kosdaq_design()
    sel = f8.selected()
    F = features(M, ex).reset_index(drop=True)
    te = M.pred_year.isin(TEST).values
    print(f"변수 {F.shape[1]}개 / 채점 결측\n{(F[te].isna().mean() * 100).round(1).to_string()}",
          flush=True)

    tr = M.pred_year.isin([2017, 2018]).values
    cmp_ = pd.DataFrame({"사건": F[tr & (M.y == 1).values].mean(),
                         "정상": F[tr & (M.y == 0).values].mean()}).round(3)
    cmp_["배수"] = (cmp_.사건 / cmp_.정상.replace(0, np.nan)).round(2)
    print("\n=== 학습 구간 비교 ===\n" + cmp_.to_string())

    base = pd.concat([B, CB[sel], CR], axis=1)
    conds = {"B+ (기준)": base, "A+": pd.concat([A, CB[sel], CR], axis=1),
             "+ 마감후 공시": pd.concat([base, F], axis=1)}
    y = M.loc[te, "y"].values
    grp = M.loc[te, "corp_code"].reset_index(drop=True)
    P, rows = {}, []
    for name, X in conds.items():
        P[name] = f8.f7.rolling(X, M, f8.START, TEST)
        rows.append({"조건": name, "변수": X.shape[1],
                     **{k: round(float(v), 4) for k, v in dm.score(y, P[name]).items()}})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "final14_score.csv", index=False, encoding="utf-8-sig")
    cis = []
    for r in ea.boot_ci(y, P["+ 마감후 공시"], P["B+ (기준)"], grp, 1000, rng):
        cis.append({"비교": "마감후 − B+", **r})
    ci = pd.DataFrame(cis)
    ci.to_csv(RES / "final14_ci.csv", index=False, encoding="utf-8-sig")
    print(ci.to_string())


if __name__ == "__main__":
    main()
