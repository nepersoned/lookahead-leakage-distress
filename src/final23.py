"""최초 공시본 재무로 다시 잰다 — 정정본 문제를 본문에서 해결.

orig_fin2.py가 원문 12,598건에서 재무 수치를 뽑았다(실패 4.1%). 정정이 없는 기업·연도에서
API 값과 대조해 이익잉여금 97.0% · 총자산회전율 95.5% · 부채비율 94.6% · ROA 90.2% 일치.
정정이 있는 건에서는 8~11%p 낮다 — 최초본과 정정본이 실제로 다르다는 뜻이다.

여기서는 그 값으로 K-Score·Ohlson을 다시 만들어 A·B·A+·B+를 재측정한다.

    API판   지금 본문 (수집 시점의 최신 정정본)
    원문판  **최초 공시본 수치만** — 예측일에 실제로 볼 수 있던 숫자

원문 요약표에는 영업현금흐름이 없다. Ohlson FUTL은 API의 영업현금흐름을 그대로 쓴다
(그 변수만 정정본일 수 있음을 한계로 남긴다).

출력: results/final23_score.csv, final23_ci.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final7 as f7
import final8 as f8
import disclosure_mock as dm
import expand_analysis as ea
import feature_screen as fs
import final5 as f5
import gate3e_full as g
from final4 import BASE

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
RES = ROOT / "results"
TEST, START = f8.TEST, f8.START


def build_classic(doc: pd.DataFrame, api: pd.DataFrame, mcap, c2s) -> pd.DataFrame:
    """원문 수치 + (영업현금흐름만 API) → K-Score·Ohlson 변수."""
    d = doc.copy()
    d["corp_code"] = d.corp_code.astype(str).str.zfill(8)
    cfo = api.set_index(["corp_code", "bsns_year"])["O_FUTL_CFO"]
    tl = api.set_index(["corp_code", "bsns_year"])["O_TLTA"]     # 참고용
    rows = []
    prev = {(c, y): v for c, y, v in zip(d.corp_code, d.bsns_year, d.net_income)}
    for r in d.itertuples():
        ta, li, eq = r.assets, r.liabilities, r.equity
        ca, cl = r.cur_assets, r.cur_liab
        rev, ni, pre = r.revenue, r.net_income, r.pretax
        if not ta or pd.isna(ta) or ta <= 0:
            continue
        wc = None if pd.isna(ca) or pd.isna(cl) else ca - cl
        ni_p = prev.get((r.corp_code, r.bsns_year - 1))
        tic = c2s.get(r.corp_code)
        mc = mcap.get((tic, r.bsns_year))
        o = {"corp_code": r.corp_code, "bsns_year": int(r.bsns_year),
             "K_lnTA": np.log(ta),
             "K_lnSLTA": np.log(rev / ta) if rev and rev > 0 else np.nan,
             "K_RETA": r.retained / ta if not pd.isna(r.retained) else np.nan,
             "K_METL": mc / li if mc and li and li > 0 else np.nan,
             "O_SIZE": np.log(ta),
             "O_TLTA": li / ta if not pd.isna(li) else np.nan,
             "O_WCTA": wc / ta if wc is not None else np.nan,
             "O_CLCA": cl / ca if ca and ca > 0 else np.nan,
             "O_OENEG": float(li > ta) if not pd.isna(li) else np.nan,
             "O_NITA": ni / ta if not pd.isna(ni) else np.nan,
             "O_FUTL_CFO": float(cfo.get((r.corp_code, int(r.bsns_year)), np.nan)),
             "O_INTWO": (np.nan if pd.isna(ni) or ni_p is None or pd.isna(ni_p)
                         else float(ni < 0 and ni_p < 0)),
             "O_CHIN": (np.nan if pd.isna(ni) or ni_p is None or pd.isna(ni_p)
                        or abs(ni) + abs(ni_p) == 0 else (ni - ni_p) / (abs(ni) + abs(ni_p))),
             }
        k = [o["K_lnTA"], o["K_lnSLTA"], o["K_RETA"], o["K_METL"]]
        o["KSCORE"] = (np.nan if any(pd.isna(v) for v in k) else
                       -17.9 + 1.5 * k[0] + 3.0 * k[1] + 14.8 * k[2] + 1.5 * k[3])
        oh = [o[c] for c in ("O_SIZE", "O_TLTA", "O_WCTA", "O_CLCA", "O_OENEG",
                             "O_NITA", "O_FUTL_CFO", "O_INTWO", "O_CHIN")]
        o["O_SCORE"] = (np.nan if any(pd.isna(v) for v in oh) else
                        -1.32 - 0.407 * oh[0] + 6.03 * oh[1] - 1.43 * oh[2] + 0.0757 * oh[3]
                        - 1.72 * oh[4] - 2.37 * oh[5] - 1.83 * oh[6] + 0.285 * oh[7]
                        - 0.521 * oh[8])
        rows.append(o)
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(42)
    ratios_api, pub, labels, mc, c2s = f7.load()
    o = ratios_api
    ratios_api["O_SCORE"] = (-1.32 - 0.407 * o.O_SIZE + 6.03 * o.O_TLTA - 1.43 * o.O_WCTA
                             + 0.0757 * o.O_CLCA - 1.72 * o.O_OENEG - 2.37 * o.O_NITA
                             - 1.83 * o.O_FUTL_CFO + 0.285 * o.O_INTWO - 0.521 * o.O_CHIN)
    fy_mc = mc[mc.dt.dt.month == 12].assign(year=lambda d: d.dt.dt.year)
    mcap = {(t, y): v for t, y, v in zip(fy_mc.ticker, fy_mc.year, fy_mc.mcap)}

    doc = pd.read_csv(FULL / "orig_fin2.csv", dtype={"corp_code": str})
    ratios_doc = build_classic(doc, ratios_api, mcap, c2s)
    print(f"원문판 재무 {len(ratios_doc):,}행 / {ratios_doc.corp_code.nunique():,}개사 "
          f"{int(ratios_doc.bsns_year.min())}~{int(ratios_doc.bsns_year.max())}", flush=True)
    print("결측률\n" + (ratios_doc[BASE].isna().mean() * 100).round(1).to_string(), flush=True)
    ratios_doc.to_csv(FULL / "classic_orig.csv", index=False, encoding="utf-8-sig")

    sel = f8.selected()
    orig, amend = dm.load_filings()
    aud = dm.load_auditors()
    f = fs.load_filings_all()
    f["cat"] = fs.classify(f["report_nm"])

    out, cis = [], []
    for tag, ratios in (("API판 (본문)", ratios_api), ("원문판 (최초 공시본)", ratios_doc)):
        A, B, M = f7.design(ratios, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
        te = M.pred_year.isin(TEST).values
        y = M.loc[te, "y"].values
        grp = M.loc[te, "corp_code"].reset_index(drop=True)
        D = dm.features(M, 1, orig, amend, aud).drop(columns=["amend_1y", "amend_3y"])
        Eb = fs.event_features(M, 1, f, f5.BEHAV).drop(
            columns=["amend_any_1y", "amend_any_3y", "amend_share_3y"])
        CB = pd.concat([D, Eb, f5.extra_features(M, f, orig)], axis=1).reset_index(drop=True)
        Er = fs.event_features(M, 1, f, f5.RULE).reset_index(drop=True)
        CR = Er[[c for c in Er.columns if c.startswith("L_")]]
        CR = CR.loc[:, CR.nunique() > 1]
        cols = [c for c in sel if c in CB.columns]
        conds = {"A": A, "B": B,
                 "A+": pd.concat([A, CB[cols], CR], axis=1),
                 "B+": pd.concat([B, CB[cols], CR], axis=1)}
        P = {}
        print(f"\n=== {tag} — 행 {len(M):,} / 채점 사건 {int(y.sum())} ===", flush=True)
        for name, X in conds.items():
            P[name] = f7.rolling(X, M, START, TEST)
            out.append({"자료": tag, "조건": name, "행": int(te.sum()), "사건": int(y.sum()),
                        **{k: round(float(v), 4) for k, v in dm.score(y, P[name]).items()}})
            print(out[-1], flush=True)
        for a, b, what in [("A", "B", "누출"), ("B+", "B", "복구"),
                           ("A+", "B+", "남는 격차"), ("B+", "A", "B+가 A를 넘는가")]:
            for r in ea.boot_ci(y, P[a], P[b], grp, 1000, rng):
                cis.append({"자료": tag, "비교": f"{a}−{b}", "의미": what, **r})
    pd.DataFrame(out).to_csv(RES / "final23_score.csv", index=False, encoding="utf-8-sig")
    ci = pd.DataFrame(cis)
    ci.to_csv(RES / "final23_ci.csv", index=False, encoding="utf-8-sig")
    print("\n" + ci[ci.지표.isin(["PR-AUC", "AUC"])].to_string())


if __name__ == "__main__":
    main()
