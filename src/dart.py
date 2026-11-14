"""DART OpenAPI 얇은 클라이언트.

파일럿 코드가 그대로 본 수집 코드가 되므로(계획서 관문1), 여기서
- 호출 속도 제한
- 응답 캐시(재실행 시 API 재호출 없음)
- 상태코드 해석
을 한 번만 제대로 해 둔다.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import requests
from dotenv import load_dotenv

BASE = "https://opendart.fss.or.kr/api"
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"

# DART 공식 한도: 인증키당 분당 1,000건. 여유 있게 초당 5건으로 묶어 둔다.
MIN_INTERVAL = 0.2

# 상태코드 → 의미 (공식 문서)
STATUS = {
    "000": "정상",
    "010": "등록되지 않은 키",
    "011": "사용할 수 없는 키",
    "012": "접근할 수 없는 IP",
    "013": "조회된 데이터 없음",
    "014": "파일이 존재하지 않음",
    "020": "요청 제한 초과",
    "021": "조회 가능한 회사 개수 초과",
    "100": "부적절한 필드값",
    "101": "부적절한 접근",
    "800": "시스템 점검 중",
    "900": "정의되지 않은 오류",
    "901": "사용자 계정의 개인정보보유기간 만료",
}


class DartError(RuntimeError):
    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(f"[{status}] {STATUS.get(status, '?')} — {message}")


class Dart:
    def __init__(self, api_key: str | None = None, use_cache: bool = True):
        if api_key is None:
            load_dotenv(ROOT / ".env")
            api_key = os.getenv("DART_API_KEY", "").strip()
        if not api_key or api_key.startswith("여기에"):
            raise SystemExit(
                "DART_API_KEY가 없습니다. .env.example을 .env로 복사하고 "
                "https://opendart.fss.or.kr 에서 발급받은 키를 넣어주세요."
            )
        self.key = api_key
        self.use_cache = use_cache
        self.session = requests.Session()
        self._last_call = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 저수준 ----------
    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        self._last_call = time.monotonic()

    def _cache_path(self, endpoint: str, params: dict) -> Path:
        keyless = {k: v for k, v in sorted(params.items()) if k != "crtfc_key"}
        digest = hashlib.sha1(
            f"{endpoint}?{json.dumps(keyless, ensure_ascii=False)}".encode()
        ).hexdigest()[:16]
        return CACHE_DIR / f"{endpoint}_{digest}.json"

    def get_json(self, endpoint: str, **params: Any) -> dict:
        """endpoint 예: 'list.json'. status 013(데이터 없음)은 예외 대신 빈 list로 돌려준다."""
        cache = self._cache_path(endpoint, params)
        if self.use_cache and cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))

        self._throttle()
        params["crtfc_key"] = self.key
        for attempt in range(10):
            try:
                resp = self.session.get(f"{BASE}/{endpoint}", params=params, timeout=30)
                resp.raise_for_status()
            except requests.exceptions.RequestException:
                # SSL EOF·연결 끊김·DNS 실패는 일시적이다. 재시도 없이 돌리면 긴 수집이
                # 중간에 죽는다. 인터넷이 몇 분 끊겨도 버티도록 최대 60초까지 기다린다.
                if attempt == 9:
                    raise
                time.sleep(min(2 ** attempt, 60))
                continue
            data = resp.json()
            status = data.get("status", "000")
            if status == "020":  # 요청 제한 — 잠깐 쉬고 재시도
                time.sleep(5 * (attempt + 1))
                continue
            break

        status = data.get("status", "000")
        if status == "013":
            data = {"status": "013", "message": "조회된 데이터 없음", "list": []}
        elif status != "000":
            raise DartError(status, data.get("message", ""))

        if self.use_cache:
            cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    # ---------- 고수준 ----------
    def list_filings(
        self,
        bgn_de: str,
        end_de: str,
        corp_cls: str | None = None,
        pblntf_ty: str | None = None,
        corp_code: str | None = None,
        last_reprt_at: str = "N",
    ) -> Iterator[dict]:
        """공시검색(list.json)을 페이지 끝까지 순회하며 공시 1건씩 yield.

        corp_cls: Y=유가(KOSPI), K=코스닥, N=코넥스, E=기타
        pblntf_ty: A=정기공시 … I=거래소공시
        각 항목의 rcept_dt가 '실제 공개일' — 이 연구의 심장(제안서 4절).
        """
        page = 1
        while True:
            params = {
                "bgn_de": bgn_de,
                "end_de": end_de,
                "page_no": page,
                "page_count": 100,
                "last_reprt_at": last_reprt_at,
            }
            if corp_cls:
                params["corp_cls"] = corp_cls
            if pblntf_ty:
                params["pblntf_ty"] = pblntf_ty
            if corp_code:
                params["corp_code"] = corp_code

            data = self.get_json("list.json", **params)
            rows = data.get("list", []) or []
            yield from rows
            total_page = int(data.get("total_page", 1) or 1)
            if page >= total_page or not rows:
                return
            page += 1

    def corp_codes(self) -> list[dict]:
        """전체 고유번호(corp_code) ↔ 종목코드 매핑. zip(xml)이라 별도 처리."""
        cache = CACHE_DIR / "corpCode.json"
        if self.use_cache and cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))

        self._throttle()
        resp = self.session.get(
            f"{BASE}/corpCode.xml", params={"crtfc_key": self.key}, timeout=60
        )
        resp.raise_for_status()
        if resp.content[:2] != b"PK":  # zip이 아니면 오류 XML
            raise DartError("900", resp.text[:200])

        import xml.etree.ElementTree as ET

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xml = zf.read(zf.namelist()[0])
        root = ET.fromstring(xml)
        out = [
            {
                "corp_code": (e.findtext("corp_code") or "").strip(),
                "corp_name": (e.findtext("corp_name") or "").strip(),
                "stock_code": (e.findtext("stock_code") or "").strip(),
                "modify_date": (e.findtext("modify_date") or "").strip(),
            }
            for e in root.iter("list")
        ]
        if self.use_cache:
            cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out

    def financials(self, corp_code: str, year: int, reprt_code: str = "11011",
                   fs_div: str = "CFS") -> list[dict]:
        """단일회사 전체 재무제표(fnlttSinglAcntAll). reprt_code 11011=사업보고서."""
        data = self.get_json(
            "fnlttSinglAcntAll.json",
            corp_code=corp_code,
            bsns_year=str(year),
            reprt_code=reprt_code,
            fs_div=fs_div,
        )
        return data.get("list", []) or []
