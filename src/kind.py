"""KIND(kind.krx.co.kr) 공시 수집 — 관리종목 지정·상장폐지 등 시장조치 공시.

DART OpenAPI의 거래소공시(pblntf_ty=I)에는 관리종목 **확정 지정** 공시가 없다.
파일럿에서 2022~23년 KOSDAQ 공시 47,122건을 전수 확인한 결과 '관리종목'이 들어간
73건이 전부 사유발생·우려 성격이었다. 확정 지정은 KIND에만 실린다.

KIND의 상세검색(details.do)은 날짜 범위 + 공시유형 코드로 조회할 수 있다.
공시유형 코드는 사이트의 상세검색 폼에서 확인한 값이다.

    0350  관리종목
    0328  상장폐지
    0311  매매거래정지
    0356  투자주의환기종목
    0313  불성실공시

사용:
    python src/kind.py --bgn 2015-01-01 --end 2024-12-31 --type 0350
    python src/kind.py --type 0350 0328 --out data/full
"""
from __future__ import annotations

import argparse
import gzip
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw"

URL = "https://kind.krx.co.kr/disclosure/details.do"
REFERER = URL + "?method=searchDetailsMain"

# 상세검색 폼에서 확인한 공시유형 코드
DISCLOSURE_TYPES = {
    "0350": "관리종목",
    "0356": "투자주의환기종목",
    "0313": "불성실공시",
    "0311": "매매거래정지",
    "0328": "상장폐지",
}

MARKETS = {"K": "2", "Y": "1", "": ""}   # K=코스닥, Y=유가증권, 빈값=전체

MIN_INTERVAL = 0.4        # 서버 부담을 줄인다
_last_call = 0.0


def _post(params: dict) -> str:
    """KIND에 POST하고 EUC-KR 응답을 문자열로 돌려준다."""
    global _last_call
    gap = time.monotonic() - _last_call
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)
    _last_call = time.monotonic()

    data = urllib.parse.urlencode(params, encoding="euc-kr").encode()
    req = urllib.request.Request(
        URL, data=data,
        headers={"User-Agent": "Mozilla/5.0", "Referer": REFERER,
                 "X-Requested-With": "XMLHttpRequest"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    body = resp.read()
    if resp.headers.get("Content-Encoding") == "gzip":
        body = gzip.decompress(body)
    # KIND는 화면마다 인코딩이 다르다. details.do는 UTF-8, adminissue.do는 EUC-KR.
    for enc in ("utf-8", "euc-kr"):
        try:
            return body.decode(enc)
        except UnicodeDecodeError:
            continue
    return body.decode("euc-kr", errors="replace")


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _parse(html: str) -> tuple[list[dict], int]:
    """결과 표를 행 목록으로, 그리고 전체 페이지 수를 돌려준다."""
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(tds) < 5:
            continue
        # 공시 제목 칸에서 상세 링크의 접수번호를 뽑는다
        m = re.search(r"acptno=(\d+)", tds[3]) or re.search(r"'(\d{14})'", tds[3])
        # 기업명 칸에는 KIND 내부 발행회사코드가 들어 있다.
        # 5자리이고 뒤에 0을 붙이면 종목코드가 된다(보통주는 끝자리가 0).
        # 이름으로 맞추면 개명·퇴출된 기업을 놓치므로 이 코드로 맞춘다.
        c = re.search(r"companysummary_open\('(\d+)'\)", tds[2])
        isur = c.group(1) if c else ""
        rows.append({
            "rcept_dt": _strip(tds[1]),
            "corp_name": _strip(tds[2]),
            "stock_code": (isur + "0") if len(isur) == 5 else isur,
            "report_nm": _strip(tds[3]),
            "submitter": _strip(tds[4]),
            "acpt_no": m.group(1) if m else "",
        })

    m = re.search(r"총\s*<em>\s*([\d,]+)\s*</em>", html) or \
        re.search(r"총\s*([\d,]+)\s*건", html)
    total = int(m.group(1).replace(",", "")) if m else len(rows)
    return rows, total


def search(bgn: str, end: str, dtype: str = "0350", market: str = "K",
           page_size: int = 100, verbose: bool = True) -> pd.DataFrame:
    """기간·공시유형으로 KIND 공시를 전부 훑는다.

    bgn, end : 'YYYY-MM-DD'
    dtype    : DISCLOSURE_TYPES 의 키
    market   : 'K'=코스닥, 'Y'=유가증권, ''=전체
    """
    rows, page = [], 1
    while True:
        params = {
            "method": "searchDetailsSub", "forward": "details_sub",
            "currentPageSize": str(page_size), "pageIndex": str(page),
            "orderMode": "1", "orderStat": "D",
            "searchCodeType": "", "repIsuSrtCd": "", "searchCorpName": "",
            "fromDate": bgn, "toDate": end,
            "marketType": MARKETS.get(market, ""),
            "disclosureType02": dtype, "disclosureTypeArr02": dtype,
        }
        html = _post(params)
        page_rows, total = _parse(html)
        rows.extend(page_rows)
        if verbose:
            print(f"  {bgn}~{end} {DISCLOSURE_TYPES.get(dtype, dtype)} "
                  f"p{page}: {len(page_rows)}건 (누적 {len(rows)})", flush=True)
        # 한 페이지가 덜 찼으면 마지막 페이지다. 총건수 파싱은 신뢰하지 않는다.
        if len(page_rows) < page_size:
            break
        page += 1
        if page > 200:                      # 안전장치
            print("  ※ 200페이지 초과 — 중단")
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["rcept_dt", "corp_name", "report_nm"])
    df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], errors="coerce")
    df["disclosure_type"] = DISCLOSURE_TYPES.get(dtype, dtype)
    return df.sort_values("rcept_dt").reset_index(drop=True)


def search_years(bgn: str, end: str, dtype: str = "0350", market: str = "K",
                 verbose: bool = True) -> pd.DataFrame:
    """긴 기간은 연 단위로 쪼개서 훑는다(한 번에 너무 많으면 페이지가 잘린다)."""
    b, e = pd.Timestamp(bgn), pd.Timestamp(end)
    out = []
    for year in range(b.year, e.year + 1):
        y_bgn = max(b, pd.Timestamp(year=year, month=1, day=1)).strftime("%Y-%m-%d")
        y_end = min(e, pd.Timestamp(year=year, month=12, day=31)).strftime("%Y-%m-%d")
        df = search(y_bgn, y_end, dtype=dtype, market=market, verbose=verbose)
        if not df.empty:
            out.append(df)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).drop_duplicates(
        subset=["rcept_dt", "corp_name", "report_nm"]).reset_index(drop=True)


# ── 사건 유형 ────────────────────────────────────────────────────
# 공시 제목 앞부분으로 갈린다. 라벨로 쓸 것은 '지정'뿐이다.
_P = r"^(?:\[[^\]]*\]\s*)?"          # 앞에 붙는 [정정] 같은 머리말은 건너뛴다

EVENT_RULES = [
    # 해제가 가장 먼저다. '관리종목지정해제'처럼 '지정'을 품고 있어서
    # 순서를 뒤로 두면 지정으로 잘못 샌다.
    ("해제",   _P + r"관리종목\s*(?:지정)?\s*(?:사유\s*(?:일부)?)?\s*해제"),
    # 코스닥은 '사유추가', 유가증권은 '사유변경'으로 쓴다. 둘 다 이미 지정된
    # 기업의 사유가 바뀌는 것이므로 새 사건이 아니다.
    ("사유변경", _P + r"관리종목\s*지정\s*사유\s*(?:추가|변경|일부)"),
    ("지정",   _P + r"관리종목\s*지정"),
    ("상장폐지", _P + r".*상장\s*폐지"),
]


def classify_event(title: str) -> str:
    for label, pat in EVENT_RULES:
        if re.search(pat, str(title)):
            return label
    return "기타"


# ── 지정 사유 ────────────────────────────────────────────────────
# 공시 제목의 괄호 안에 붙는다. 위에서부터 먼저 맞는 것을 쓴다.
REASON_RULES = {
    "감사의견":     r"감사의견|의견\s*거절|부적정|한정",
    "자본잠식":     r"자본\s*잠식",
    "매출액미달":   r"매출액",
    "손실지속":     r"계속사업손실|법인세비용차감전|영업손실|당기순손실",
    "보고서미제출": r"보고서\s*미제출|사업보고서|반기보고서|분기보고서",
    "상장폐지사유": r"상장폐지\s*사유",
    "상장적격성":   r"상장적격성",
    "회생파산":     r"회생|파산|부도|당좌거래",
    "시가총액미달": r"시가총액",
    "주식분산미달": r"주식\s*분산|주식분포",
    "공시위반":     r"불성실공시|공시\s*위반",
    "스팩요건":     r"SPAC|스팩|예비심사청구서",
    "합병소멸":     r"피흡수합병|합병",
    "기간만료":     r"만료|존속기한",
}

# 재무제표가 공개되어야 비로소 알 수 있는 사유.
# 이 비율이 높을수록 "사건과 재무제표가 같은 문서에서 나온다"는 주장이 강해진다.
FROM_STATEMENTS = {"감사의견", "자본잠식", "매출액미달", "손실지속", "보고서미제출"}


def classify_reason(text: str) -> str:
    for label, pat in REASON_RULES.items():
        if re.search(pat, str(text)):
            return label
    return "미상"


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """사건 유형·사유·재무제표 유래 여부를 붙인다."""
    df = df.copy()
    df["event"] = df["report_nm"].map(classify_event)
    df["reason"] = df["report_nm"].map(classify_reason)
    df["from_statements"] = df["reason"].isin(FROM_STATEMENTS)
    df["year"] = df["rcept_dt"].dt.year
    df["month"] = df["rcept_dt"].dt.month
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bgn", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--type", nargs="+", default=["0350"],
                   help=" ".join(f"{k}={v}" for k, v in DISCLOSURE_TYPES.items()))
    p.add_argument("--market", default="K", help="K=코스닥 Y=유가증권")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--name", default="kind_events.csv")
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for dtype in a.type:
        print(f"\n[{DISCLOSURE_TYPES.get(dtype, dtype)}] 수집")
        df = search_years(a.bgn, a.end, dtype=dtype, market=a.market)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("수집 0건")
        return

    df = annotate(pd.concat(frames, ignore_index=True))
    df.to_csv(out_dir / a.name, index=False, encoding="utf-8-sig")

    print(f"\n총 {len(df):,}건 → {out_dir / a.name}")
    print(f"고유 기업 {df['corp_name'].nunique():,}개")
    print("\n[공시유형별]")
    print(df["disclosure_type"].value_counts().to_string())
    print("\n[연도별]")
    print(df["year"].value_counts().sort_index().to_string())
    print("\n[월별]  ← 1~4월 집중이 보여야 한다")
    print(df["month"].value_counts().sort_index().to_string())
    print("\n[사건 유형]")
    print(df["event"].value_counts().to_string())

    desig = df[df["event"] == "지정"]
    print(f"\n── 라벨로 쓸 '지정' {len(desig):,}건 "
          f"(고유기업 {desig['corp_name'].nunique():,}개) ──")
    print("\n[지정의 월별 분포]  ← 1~4월 집중이 보여야 한다")
    m = desig["month"].value_counts().sort_index()
    print(m.to_string())
    if len(desig):
        print(f"1~4월 집중도 {m.reindex(range(1, 5)).sum():.0f}/{len(desig)}"
              f" = {m.reindex(range(1, 5)).sum() / len(desig):.1%}")
    print("\n[지정 사유]")
    print(desig["reason"].value_counts().to_string())
    known = desig[desig["reason"] != "미상"]
    if len(known):
        print(f"\n재무제표 유래 비율(미상 제외): "
              f"{known['from_statements'].mean():.1%}")
    print(f"미상 비율: {(desig['reason'] == '미상').mean():.1%}")


if __name__ == "__main__":
    main()
