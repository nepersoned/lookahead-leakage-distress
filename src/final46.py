"""final46 — 사건이 사업보고서에 묶인 정도가 누출 크기를 정하는가 (일반화, 탐색적).

같은 모집단·같은 A/B 설계·같은 모형(본문 LightGBM, 1월 예측, 2019–2024 채점)에서 사건 정의만 바꾼다.
과제
    전체 지정           본문
    감사·결산 확정 지정  사유군(본문 규칙)          다른 사유로 지정된 기업·연도는 뺀다(오염된 음성 방지)
    거래소 재량 지정     〃
    법적 절차 지정       〃
    기타 지정            〃 (스팩요건·주식분산 등)
    상장폐지             KIND·DART 비자발적 상장폐지(final32 라벨), 같은 모집단
    감사의견 비적정      FY(Y−1) 감사의견이 한정·부적정·의견거절 (사건일 = FY(Y−1) 사업보고서 접수일)
모형 전에 계산하는 진단값 (라벨과 공시일만으로)
    D_before  사건이 그 기업의 FY(Y−1) 사업보고서 접수일 '이전이거나 같은 날' 일어난 비율
    D_near    접수일 ±30일 안에 일어난 비율
결과: 과제별 A − B (PR-AUC, AUC, 상대 누출 (A−B)/B)와 신뢰구간, 진단값. 과제 간 순위상관.
출력: results/final46_tasks.csv, final46.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import final7 as f7
import final27 as f27
import disclosure_mock as dm
import expand_analysis as ea

warnings.filterwarnings("ignore")
START, TEST = 2017, list(range(2019, 2025))


def main():
    rng = np.random.default_rng(42)
    _, pub, labels, mc, c2s = f7.load()
    r = pd.read_csv(ea.FULL / "classic_orig.csv", dtype={"corp_code": str})
    r["corp_code"] = r.corp_code.str.zfill(8)
    A, B, M = f7.design(r, pub, labels, mc, c2s, list(range(2017, 2025)), 1)
    A, B, M = A.reset_index(drop=True), B.reset_index(drop=True), M.reset_index(drop=True)
    key = list(zip(M.corp_code, M.pred_year))

    # FY(Y−1) 사업보고서 접수일
    pb = pub.copy()
    pb["corp_code"] = pb.corp_code.astype(str).str.zfill(8)
    fdate = pb.groupby(["corp_code", "bsns_year"]).pub_dt.min().to_dict()
    M["file_dt"] = [fdate.get((c, y - 1)) for c, y in key]

    # 지정 사유군
    ev = f27.reason_groups(M[M.y == 1][["corp_code", "pred_year", "y"]].copy())
    gmap = dict(zip(zip(ev.corp_code, ev.pred_year), ev.사유군))
    edt = dict(zip(zip(ev.corp_code, ev.pred_year), pd.to_datetime(ev.rcept_dt)))
    grp = np.array([gmap.get(k, "") for k in key])

    tasks = {}
    tasks["전체 지정"] = (M.y.values.copy(), np.ones(len(M), bool), np.array([edt.get(k) for k in key], dtype=object))
    for gname in ["감사·결산 확정", "거래소 재량", "법적 절차", "기타·불명"]:
        y = ((M.y.values == 1) & (grp == gname)).astype(int)
        keep = ~((M.y.values == 1) & (grp != gname))
        tasks[f"지정: {gname}"] = (y, keep, np.array([edt.get(k) for k in key], dtype=object))

    # 감사·결산 확정을 지정 규칙의 성격으로 다시 나눈다
    rs = dict(zip(zip(ev.corp_code, ev.pred_year), ev.사유.astype(str)))
    sb = dict(zip(zip(ev.corp_code, ev.pred_year), ev["sub"].fillna("").astype(str)))
    def fine(k):
        r_, s_ = rs.get(k, ""), sb.get(k, "")
        if s_ == "감사의견" or "감사" in r_:
            return "감사의견"
        if s_ == "보고서미제출" or "미제출" in r_:
            return "보고서 미제출"
        if s_ == "자본잠식" or any(w in r_ for w in ("손실", "자본잠식", "매출액")):
            return "재무수치 규칙"
        return ""
    fg = np.array([fine(k) if M.y.values[i] == 1 else "" for i, k in enumerate(key)])
    for gname in ["재무수치 규칙", "감사의견", "보고서 미제출"]:
        y = ((M.y.values == 1) & (fg == gname)).astype(int)
        keep = ~((M.y.values == 1) & (fg != gname))
        tasks[f"지정: {gname}"] = (y, keep, np.array([edt.get(k) for k in key], dtype=object))

    # 상장폐지 (final32 라벨, 기업별 최초)
    dl = pd.read_csv("results/final32_labels.csv", dtype={"corp_code": str}, parse_dates=["event_dt"])
    dl["corp_code"] = dl.corp_code.str.zfill(8)
    dmap = dl.groupby("corp_code").event_dt.min().to_dict()
    yd, dd = [], []
    for c, yy in key:
        d = dmap.get(c)
        T = pd.Timestamp(year=yy, month=1, day=1)
        hit = d is not None and T <= d < T + pd.Timedelta(days=365)
        yd.append(int(hit)); dd.append(d if hit else None)
    tasks["상장폐지"] = (np.array(yd), np.ones(len(M), bool), np.array(dd, dtype=object))

    # 감사의견 비적정 (FY(Y−1), 당기 보고서 기준)
    au = pd.read_csv(ea.FULL / "auditors_full.csv", dtype=str)
    # 감사인 자료는 3년마다 받은 보고서의 당기·전기·전전기 행이다. 연도마다 가장 가까운 보고서의 값을 쓴다.
    au["rk"] = au.src.map({"당기": 0, "전기": 1, "전전기": 2})
    au = au.sort_values("rk").drop_duplicates(["corp_code", "bsns_year"]).copy()
    au["corp_code"] = au.corp_code.str.zfill(8)
    au["bad"] = au.adt_opinion.fillna("").str.contains("거절|한정|부적정")
    bad = au.groupby(["corp_code", "bsns_year"]).bad.max()
    bad.index = [(c, int(y)) for c, y in bad.index]
    ya = np.array([int(bad.get((c, y - 1), False)) for c, y in key])
    known = np.array([(c, y - 1) in bad.index for c, y in key])
    tasks["감사의견 비적정"] = (ya, known, np.array([fdate.get((c, y - 1)) if v else None
                                                for (c, y), v in zip(key, ya)], dtype=object))

    te_all = M.pred_year.isin(TEST).values
    rows = []
    for name, (y, keep, dts) in tasks.items():
        Mk = M[keep].reset_index(drop=True).assign(y=y[keep])
        Ak, Bk = A[keep].reset_index(drop=True), B[keep].reset_index(drop=True)
        te = Mk.pred_year.isin(TEST).values
        yt = Mk.y.values[te]
        if yt.sum() < 10:
            print(name, "사건 부족", int(yt.sum()))
            continue
        pa, pb_ = f7.rolling(Ak, Mk, START, TEST), f7.rolling(Bk, Mk, START, TEST)
        sa, sb = dm.score(yt, pa), dm.score(yt, pb_)
        ci = {c["지표"]: c for c in ea.boot_ci(yt, pa, pb_, Mk.loc[te, "corp_code"].reset_index(drop=True), 1000, rng)}
        # 진단값: 채점 구간 사건 기준
        sel = keep & te_all & (y == 1)
        ed = pd.to_datetime(pd.Series(dts[sel]))
        fd = pd.to_datetime(M.file_dt.values[sel])
        okd = ed.notna().values & ~pd.isna(fd)
        diff = (ed.values[okd] - fd[okd]) / np.timedelta64(1, "D")
        rows.append({"과제": name, "채점 행": int(te.sum()), "사건": int(yt.sum()),
                     "사건률": round(float(yt.mean()), 4),
                     "D_before": round(float((diff <= 0).mean()), 3) if okd.any() else np.nan,
                     "D_near": round(float((np.abs(diff) <= 30).mean()), 3) if okd.any() else np.nan,
                     "날짜 확인 사건": int(okd.sum()),
                     "A PR-AUC": round(sa["PR-AUC"], 4), "B PR-AUC": round(sb["PR-AUC"], 4),
                     "누출 PR-AUC": ci["PR-AUC"]["격차"], "하한": ci["PR-AUC"]["하한"], "상한": ci["PR-AUC"]["상한"],
                     "상대 누출": round((sa["PR-AUC"] - sb["PR-AUC"]) / sb["PR-AUC"], 3),
                     "누출 AUC": ci["AUC"]["격차"], "AUC 하한": ci["AUC"]["하한"], "AUC 상한": ci["AUC"]["상한"]})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    from scipy.stats import spearmanr
    rho = {m: spearmanr(t.D_before, t[m]) for m in ("누출 PR-AUC", "상대 누출", "누출 AUC")}
    t.to_csv("results/final46_tasks.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    txt = t.to_string(index=False) + "\n\n" + "\n".join(
        f"Spearman(D_before, {m}) = {v.statistic:.3f} (p={v.pvalue:.3f}, n={len(t)})" for m, v in rho.items())
    open("results/final46.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
