"""final50 — '정정 성분'(final41)이 실제 정정공시 때문인지, 연결·별도 선택이나 파싱 차이 때문인지 점검.

대상: final41과 같은 행에서 FY(Y−1) 값이 최초본(classic_orig)과 정정본(DART API, classic_full)에서 다른 기업·연도.
보는 것
  ① 그 사업연도 사업보고서에 [기재정정] 등 정정 제출이 있는가(reports_full.is_amended)
  ② 차이의 크기 유형: 총자산 로그가 0.05 넘게 다르면 '규모 차이'(연결·별도 선택 차이 의심),
     총자산은 같고(≤0.001) 손익·자본 비율만 다르면 '손익·자본만 차이'(정정 또는 계정 선택 차이)
  ③ 지정 기업과 정상 기업으로 나눠 ①·② 비율
  ④ 지정 기업 사례 20건 목록(원문 확인용): 최초 접수번호, 정정 보고서 이름·접수일, 주요 비율 두 값
출력: results/final50_summary.csv, final50_cases.csv, final50.log
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import expand_analysis as ea

FULL = ea.FULL


def main():
    o = pd.read_csv(FULL / "classic_orig.csv", dtype={"corp_code": str}); o["corp_code"] = o.corp_code.str.zfill(8)
    a = pd.read_csv(FULL / "classic_full.csv", dtype={"corp_code": str}); a["corp_code"] = a.corp_code.str.zfill(8)
    f2 = pd.read_csv(FULL / "orig_fin2.csv", dtype={"corp_code": str, "src_rcept": str}); f2["corp_code"] = f2.corp_code.str.zfill(8)
    rep = pd.read_csv(FULL / "reports_full.csv", dtype={"corp_code": str, "rcept_no": str}); rep["corp_code"] = rep.corp_code.str.zfill(8)
    rep = rep[rep.report_type == "사업"].copy()
    rep["bsns_year"] = rep.bsns_year.astype(int)
    amend = rep[rep.is_amended.astype(str) == "True"].groupby(["corp_code", "bsns_year"]).agg(
        정정건수=("rcept_no", "size"), 첫정정일=("rcept_dt", "min"), 정정보고서=("report_nm", "first"))
    P = pd.read_csv("results/final27_preds.csv", dtype={"corp_code": str}); P["corp_code"] = P.corp_code.str.zfill(8)
    P["bsns_year"] = P.pred_year - 1
    m = P[["corp_code", "bsns_year", "y"]].merge(o, on=["corp_code", "bsns_year"]).merge(
        a[["corp_code", "bsns_year", "O_SIZE", "O_NITA", "O_TLTA", "K_RETA", "O_SCORE"]],
        on=["corp_code", "bsns_year"], suffixes=("", "_api"))
    d = {c: (m[c] - m[f"{c}_api"]).abs() for c in ["O_SIZE", "O_NITA", "O_TLTA", "K_RETA", "O_SCORE"]}
    m["다름"] = pd.concat([d[c] > 1e-6 for c in d], axis=1).any(axis=1)
    m["유형"] = np.where(~m.다름, "같음", np.where(d["O_SIZE"] > 0.05, "규모 차이(연결·별도 의심)",
                         np.where(d["O_SIZE"] <= 0.001, "총자산 같고 손익·자본만 차이", "총자산 소폭 차이")))
    m = m.merge(amend.reset_index(), on=["corp_code", "bsns_year"], how="left")
    m["정정 제출"] = m.정정건수.fillna(0) > 0
    summ = (m.groupby(["y", "유형"]).agg(행=("corp_code", "size"), 정정제출비율=("정정 제출", "mean"))
            .round(3).reset_index())
    summ["y"] = summ.y.map({1: "지정", 0: "정상"})
    tot = m.groupby("y").agg(행=("corp_code", "size"), 다름비율=("다름", "mean"), 정정제출비율=("정정 제출", "mean")).round(3)
    diff_ev = m[(m.y == 1) & m.다름].merge(f2[["corp_code", "bsns_year", "src_rcept"]], on=["corp_code", "bsns_year"], how="left")
    cases = diff_ev.sample(min(20, len(diff_ev)), random_state=1)[
        ["corp_code", "bsns_year", "유형", "src_rcept", "정정 제출", "정정보고서", "첫정정일",
         "O_NITA", "O_NITA_api", "O_TLTA", "O_TLTA_api", "O_SIZE", "O_SIZE_api"]].round(4)
    summ.to_csv("results/final50_summary.csv", index=False, encoding="utf-8-sig")
    cases.to_csv("results/final50_cases.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    txt = "\n\n".join([tot.to_string(), summ.to_string(index=False), cases.to_string(index=False)])
    open("results/final50.log", "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
