"""T1 — 사업보고서 원문에서 「II. 사업의 내용」을 잘라낸다.

실험 C의 관문이다. 여기서 섹션이 안 잘리면 텍스트 변수를 만들 수 없다.

DART document.xml은 ZIP으로 오고 그 안에 XML이 여러 개 들어 있다.
본문 XML은 SECTION-1 / SECTION-2 / TITLE 태그로 목차가 구조화돼 있어
정규식으로 제목을 훑는 것보다 태그를 타고 들어가는 편이 안전하다.

사용:
    python src/t1_parse.py --n 5              # 판정용 표본 실행
    python src/t1_parse.py --rcept 20230323001407
"""
from __future__ import annotations

import argparse
import io
import os
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
CACHE = ROOT / "data" / "cache" / "docs"
OUT = ROOT / "data" / "text"

# 「II. 사업의 내용」의 표기 흔들림을 모두 받는다.
# 로마자 II를 한글 'Ⅱ'로 쓰는 경우, 마침표가 없는 경우, 공백 차이가 있다.
RE_TARGET = re.compile(r"^\s*(?:II|Ⅱ|2)\s*[.．]?\s*사업의\s*내용\s*$")
RE_NEXT = re.compile(r"^\s*(?:III|Ⅲ|3)\s*[.．]?\s*재무에\s*관한\s*사항\s*$")


_SESSION = requests.Session()
_LAST = [0.0]
MIN_INTERVAL = 0.35   # 원문 API는 응답이 무거워 목록 API보다 여유를 둔다


def fetch(rcept_no: str, key: str, retries: int = 4) -> bytes:
    """원문 ZIP. 한 번 받은 것은 캐시에 둔다 — 재실행 시 API를 안 부른다.

    끊김이 잦아 재시도를 넣는다. 재시도 없이 돌리면 수천 건이 ConnectionError로
    통째로 날아간다(실제로 한 번 겪었다).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{rcept_no}.zip"
    if p.exists():
        return p.read_bytes()

    last = None
    for attempt in range(retries):
        gap = time.monotonic() - _LAST[0]
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        _LAST[0] = time.monotonic()
        try:
            r = _SESSION.get("https://opendart.fss.or.kr/api/document.xml",
                             params={"crtfc_key": key, "rcept_no": rcept_no},
                             timeout=120)
            r.raise_for_status()
            if r.content[:2] != b"PK":
                raise RuntimeError(f"ZIP이 아님: {r.content[:200]!r}")
            p.write_bytes(r.content)
            return r.content
        except Exception as exc:                       # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)                   # 1, 2, 4, 8초
    raise last


def main_xml(blob: bytes, rcept_no: str) -> str:
    """분리 제출된 첨부(_00760 등)가 아니라 본문 XML을 고른다.

    인코딩 주의 — XML 선언은 전부 `encoding="utf-8"`이라고 써 있지만
    2021년 이전 문서는 실제로 EUC-KR이다. 선언을 믿고 utf-8로 읽으면
    제목이 깨져 섹션을 못 찾는다(FY2018~2021이 통째로 실패했다).
    """
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = [n for n in z.namelist() if n.endswith(".xml")]
    exact = [n for n in names if n == f"{rcept_no}.xml"]
    pick = exact[0] if exact else max(names, key=lambda n: z.getinfo(n).file_size)
    raw = z.read(pick)
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp949", "replace")


def _strip(html: str) -> str:
    """표·이미지를 버리고 문장만 남긴다.

    TABLE을 지우는 이유 — 「II. 사업의 내용」의 절반 이상이 매출 실적표다.
    임베딩에 넣으면 숫자가 문장을 덮는다. 서술만 쓴다.
    """
    t = re.sub(r"<TABLE.*?</TABLE>", " ", html, flags=re.S | re.I)
    t = re.sub(r"<(IMAGE|IMG)[^>]*>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "\n", t)
    t = (t.replace("&nbsp;", " ").replace("&amp;", "&")
          .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))
    t = re.sub(r"[ \t\u00a0]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def extract(xml: str) -> tuple[str, str]:
    """(섹션 텍스트, 판정 사유). 실패하면 빈 문자열."""
    titles = [(m.start(), m.end(), re.sub(r"<[^>]+>", "", m.group(1)).strip())
              for m in re.finditer(r"<TITLE[^>]*>(.*?)</TITLE>", xml, re.S)]
    if not titles:
        return "", "TITLE 태그 없음"

    start = end = None
    for i, (s, e, name) in enumerate(titles):
        if RE_TARGET.match(name):
            start = e
            for s2, _, n2 in titles[i + 1:]:
                if RE_NEXT.match(n2):
                    end = s2
                    break
            break
    if start is None:
        return "", "「II. 사업의 내용」 제목을 못 찾음"
    if end is None:
        return "", "다음 절(III)을 못 찾아 끝을 못 정함"

    body = _strip(xml[start:end])
    if RE_NEXT.search(body):
        return body, "경고: 잘린 본문에 III 제목이 섞임"
    return body, "정상"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="판정용 표본 기업 수")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--rcept", help="특정 접수번호 하나만")
    a = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.getenv("DART_API_KEY", "").strip()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.rcept:
        targets = [("(직접지정)", a.rcept)]
    else:
        rep = pd.read_csv(FULL / "reports_full.csv", dtype={"corp_code": str})
        ann = rep[(rep.report_type == "사업") & (~rep.is_amended)
                  & (rep.bsns_year == a.year)].drop_duplicates("rcept_no")
        pick = ann.sample(n=a.n, random_state=42)
        targets = list(zip(pick.corp_name, pick.rcept_no.astype(str)))

    rows = []
    for name, rno in targets:
        try:
            xml = main_xml(fetch(rno, key), rno)
            body, why = extract(xml)
        except Exception as exc:                      # noqa: BLE001
            body, why = "", f"오류: {exc}"
        ok = len(body) >= 2000 and why == "정상"
        rows.append({"기업": name, "접수번호": rno, "글자수": len(body),
                     "판정": "통과" if ok else "실패", "사유": why})
        if body:
            (OUT / f"{rno}.txt").write_text(body, encoding="utf-8")
        print(f"  {name:14s} {len(body):>7,}자  {'통과' if ok else '실패'}  {why}",
              flush=True)

    df = pd.DataFrame(rows)
    n_ok = int((df["판정"] == "통과").sum())
    print(f"\n판정 — {n_ok}/{len(df)}곳 통과 (기준: 표·이미지 제외 2,000자 이상)")
    print(f"→ 섹션 텍스트 {OUT}/, 원문 캐시 {CACHE}/")
    df.to_csv(ROOT / "results" / "t1_parse.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
