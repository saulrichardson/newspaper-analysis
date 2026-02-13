#!/usr/bin/env python3
"""
Build place/county covariate panel aligned to selected city panels.

Outputs:
  - covariates/city_place_crosswalk.csv
  - covariates/city_county_crosswalk.csv
  - covariates/city_year_external_covariates.csv
  - covariates/covariate_missingness_report.csv
  - covariates/provenance.json

Notes:
  - Place-level covariates: ACS 5-year API.
  - County-level covariates: ACS profile API.
  - Building permits: Census BPS annual place files (where available).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import re
import time
import zipfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests


_STATE_ABBR_TO_FIPS: dict[str, str] = {
    "al": "01",
    "ak": "02",
    "az": "04",
    "ar": "05",
    "ca": "06",
    "co": "08",
    "ct": "09",
    "de": "10",
    "dc": "11",
    "fl": "12",
    "ga": "13",
    "hi": "15",
    "id": "16",
    "il": "17",
    "in": "18",
    "ia": "19",
    "ks": "20",
    "ky": "21",
    "la": "22",
    "me": "23",
    "md": "24",
    "ma": "25",
    "mi": "26",
    "mn": "27",
    "ms": "28",
    "mo": "29",
    "mt": "30",
    "ne": "31",
    "nv": "32",
    "nh": "33",
    "nj": "34",
    "nm": "35",
    "ny": "36",
    "nc": "37",
    "nd": "38",
    "oh": "39",
    "ok": "40",
    "or": "41",
    "pa": "42",
    "ri": "44",
    "sc": "45",
    "sd": "46",
    "tn": "47",
    "tx": "48",
    "ut": "49",
    "vt": "50",
    "va": "51",
    "wa": "53",
    "wv": "54",
    "wi": "55",
    "wy": "56",
}

_PLACE_SUFFIXES = (
    "census designated place",
    "unified government balance",
    "metropolitan government balance",
    "municipio",
    "municipality",
    "city and borough",
    "borough",
    "village",
    "town",
    "city",
    "cdp",
)

_ACS_PLACE_VARS = [
    "B01003_001E",  # population
    "B25001_001E",  # housing units
    "B19013_001E",  # median household income
    "B25077_001E",  # median home value
    "B25002_001E",  # occupancy status total
    "B25002_003E",  # vacancy count
]

_ACS_COUNTY_PROFILE_VARS = [
    "DP03_0009PE",  # unemployment rate
    "DP03_0088E",  # per-capita income
    "DP03_0033PE",  # agriculture share
    "DP03_0034PE",  # construction share
    "DP03_0035PE",  # manufacturing share
    "DP03_0037PE",  # retail share
    "DP03_0041PE",  # professional services share
]

_BPS_REGION_SPEC = {
    "Midwest": ("Midwest%20Region", "mw"),
    "Northeast": ("Northeast%20Region", "ne"),
    "South": ("South%20Region", "so"),
    "West": ("West%20Region", "we"),
}


def _to_float(v: Any) -> float:
    try:
        x = float(str(v).strip())
    except Exception:
        return math.nan
    if not math.isfinite(x):
        return math.nan
    return x


def _to_int(v: Any) -> int | None:
    x = _to_float(v)
    if not math.isfinite(x):
        return None
    try:
        return int(round(x))
    except Exception:
        return None


def _canonical_place_name(name: str) -> str:
    s = str(name or "").strip().lower()
    if not s:
        return ""
    if "," in s:
        s = s.split(",", 1)[0].strip()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for suf in sorted(_PLACE_SUFFIXES, key=len, reverse=True):
        if s.endswith(" " + suf):
            s = s[: -len(suf) - 1].strip()
    s = s.replace("saint ", "st ")
    return s


def _request_json_with_retry(url: str, *, timeout: float = 90.0, max_retries: int = 3) -> Any:
    last_err: Exception | None = None
    for i in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if i + 1 < max_retries:
                time.sleep(1.25 * (i + 1))
                continue
    raise RuntimeError(f"Request failed after retries: {url} err={last_err}")


def _request_text_with_retry(url: str, *, timeout: float = 90.0, max_retries: int = 3) -> str:
    last_err: Exception | None = None
    for i in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if i + 1 < max_retries:
                time.sleep(1.25 * (i + 1))
                continue
    raise RuntimeError(f"Request failed after retries: {url} err={last_err}")


@dataclass(frozen=True)
class CityRef:
    city_key: str
    city_name: str
    state_abbr: str
    region: str
    urbanicity_proxy: str
    min_issue_year: int
    max_issue_year: int


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build external covariate panel for selected city panels.")
    ap.add_argument(
        "--run-dir",
        default="/Users/saulrichardson/projects/newspapers/newspaper-analysis/reports/runs/prototype_zoning_panel_analysis/prototype_zoning_panel_analysis_iter8_scale30",
        help="Prototype run directory with panels/selected_panel_issues.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Output directory for covariate artifacts (default: <run-dir>/covariates).",
    )
    ap.add_argument(
        "--acs-start-year",
        type=int,
        default=2009,
        help="Earliest ACS year to request.",
    )
    ap.add_argument(
        "--acs-end-year",
        type=int,
        default=0,
        help="Latest ACS year to request (0 => current_year - 1).",
    )
    ap.add_argument(
        "--permits-start-year",
        type=int,
        default=1980,
        help="Earliest year for BPS annual place permit files.",
    )
    ap.add_argument(
        "--year-min",
        type=int,
        default=0,
        help="Optional global minimum year override (0 => infer from selected issues).",
    )
    ap.add_argument(
        "--year-max",
        type=int,
        default=0,
        help="Optional global maximum year override (0 => infer from selected issues).",
    )
    ap.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached API/file pulls and re-fetch.",
    )
    return ap.parse_args()


def _load_city_refs(run_dir: Path) -> list[CityRef]:
    p = run_dir / "panels" / "selected_panel_issues.csv"
    if p.is_file():
        df = pd.read_csv(p)
    else:
        rows: list[dict[str, Any]] = []
        panels_dir = run_dir / "panels"
        if not panels_dir.is_dir():
            raise SystemExit(f"Missing required directory: {panels_dir}")
        for city_dir in sorted(panels_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            issue_path = city_dir / "issue_texts.jsonl"
            if not issue_path.is_file():
                continue
            for raw in issue_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rows.append(
                    {
                        "city_key": str(obj.get("city_key") or city_dir.name),
                        "city_name": str(obj.get("city_name") or ""),
                        "state_abbr": str(obj.get("state_abbr") or "").lower(),
                        "region": str(obj.get("region") or ""),
                        "urbanicity_proxy": str(obj.get("urbanicity_proxy") or ""),
                        "issue_date": str(obj.get("issue_date") or ""),
                    }
                )
        if not rows:
            raise SystemExit(f"Missing required file: {p} and could not reconstruct from panels/issue_texts.jsonl")
        df = pd.DataFrame.from_records(rows)
    need = {"city_key", "city_name", "state_abbr", "region", "urbanicity_proxy", "issue_date"}
    miss = sorted(c for c in need if c not in df.columns)
    if miss:
        raise SystemExit(f"{p} missing columns: {miss}")

    df["state_abbr"] = df["state_abbr"].astype(str).str.lower()
    df["issue_year"] = pd.to_datetime(df["issue_date"], errors="coerce").dt.year
    df = df.dropna(subset=["issue_year"]).copy()
    g = (
        df.groupby(["city_key", "city_name", "state_abbr", "region", "urbanicity_proxy"], dropna=False)
        .agg(min_issue_year=("issue_year", "min"), max_issue_year=("issue_year", "max"))
        .reset_index()
        .sort_values(["state_abbr", "city_name"])
    )
    refs: list[CityRef] = []
    for r in g.itertuples(index=False):
        refs.append(
            CityRef(
                city_key=str(r.city_key),
                city_name=str(r.city_name),
                state_abbr=str(r.state_abbr),
                region=str(r.region),
                urbanicity_proxy=str(r.urbanicity_proxy),
                min_issue_year=int(r.min_issue_year),
                max_issue_year=int(r.max_issue_year),
            )
        )
    if not refs:
        raise SystemExit("No city refs discovered from selected_panel_issues.csv")
    return refs


def _load_gazetteer_places(cache_dir: Path, *, refresh: bool) -> pd.DataFrame:
    cache_json = cache_dir / "gazetteer_places_national.json"
    if cache_json.is_file() and not refresh:
        arr = json.loads(cache_json.read_text(encoding="utf-8"))
        return pd.DataFrame.from_records(arr)

    url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/Gaz_places_national.zip"
    raw = requests.get(url, timeout=90)
    raw.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(raw.content))
    names = zf.namelist()
    if not names:
        raise SystemExit("Unexpected empty gazetteer zip")
    with zf.open(names[0]) as f:
        gaz = pd.read_csv(f, sep="\t", dtype=str, encoding="latin1")
    gaz.columns = [str(c).strip() for c in gaz.columns]

    need = {"USPS", "GEOID", "NAME", "POP10", "INTPTLAT", "INTPTLONG"}
    miss = sorted(c for c in need if c not in gaz.columns)
    if miss:
        raise SystemExit(f"Gazetteer file missing columns: {miss}")

    gaz = gaz[list(need)].copy()
    gaz["state_abbr"] = gaz["USPS"].astype(str).str.lower()
    gaz["state_fips"] = gaz["GEOID"].astype(str).str.slice(0, 2)
    gaz["place_fips"] = gaz["GEOID"].astype(str).str.slice(2, 7)
    gaz["canonical_name"] = gaz["NAME"].astype(str).map(_canonical_place_name)
    gaz["pop10"] = pd.to_numeric(gaz["POP10"], errors="coerce")
    gaz["intptlat"] = pd.to_numeric(gaz["INTPTLAT"], errors="coerce")
    gaz["intptlong"] = pd.to_numeric(gaz["INTPTLONG"], errors="coerce")
    gaz = gaz[
        [
            "state_abbr",
            "state_fips",
            "place_fips",
            "GEOID",
            "NAME",
            "canonical_name",
            "pop10",
            "intptlat",
            "intptlong",
        ]
    ].rename(columns={"GEOID": "place_geoid", "NAME": "place_name_gazetteer"})
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_json.write_text(gaz.to_json(orient="records"), encoding="utf-8")
    return gaz


def _match_city_to_place(refs: list[CityRef], gaz: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in refs:
        st = r.state_abbr
        city_can = _canonical_place_name(r.city_name)
        g = gaz[gaz["state_abbr"] == st].copy()
        exact = g[g["canonical_name"] == city_can].copy()
        if not exact.empty:
            exact = exact.sort_values(["pop10", "place_geoid"], ascending=[False, True])
            hit = exact.iloc[0]
            rows.append(
                {
                    "city_key": r.city_key,
                    "city_name": r.city_name,
                    "state_abbr": st,
                    "region": r.region,
                    "urbanicity_proxy": r.urbanicity_proxy,
                    "min_issue_year": r.min_issue_year,
                    "max_issue_year": r.max_issue_year,
                    "place_geoid": str(hit["place_geoid"]),
                    "state_fips": str(hit["state_fips"]),
                    "place_fips": str(hit["place_fips"]),
                    "place_name_gazetteer": str(hit["place_name_gazetteer"]),
                    "place_pop10": _to_int(hit["pop10"]),
                    "intptlat": _to_float(hit["intptlat"]),
                    "intptlong": _to_float(hit["intptlong"]),
                    "match_method": "exact",
                    "match_score": 1.0,
                    "candidate_count": int(len(exact)),
                }
            )
            continue

        best: dict[str, Any] | None = None
        for gr in g.itertuples(index=False):
            cand = str(gr.canonical_name or "")
            if not cand:
                continue
            score = SequenceMatcher(None, city_can, cand).ratio()
            if best is None or score > float(best["score"]):
                best = {"score": score, "row": gr}
        if best is not None and float(best["score"]) >= 0.88:
            hit = best["row"]
            rows.append(
                {
                    "city_key": r.city_key,
                    "city_name": r.city_name,
                    "state_abbr": st,
                    "region": r.region,
                    "urbanicity_proxy": r.urbanicity_proxy,
                    "min_issue_year": r.min_issue_year,
                    "max_issue_year": r.max_issue_year,
                    "place_geoid": str(hit.place_geoid),
                    "state_fips": str(hit.state_fips),
                    "place_fips": str(hit.place_fips),
                    "place_name_gazetteer": str(hit.place_name_gazetteer),
                    "place_pop10": _to_int(hit.pop10),
                    "intptlat": _to_float(hit.intptlat),
                    "intptlong": _to_float(hit.intptlong),
                    "match_method": "fuzzy",
                    "match_score": float(best["score"]),
                    "candidate_count": int(len(g)),
                }
            )
            continue

        rows.append(
            {
                "city_key": r.city_key,
                "city_name": r.city_name,
                "state_abbr": st,
                "region": r.region,
                "urbanicity_proxy": r.urbanicity_proxy,
                "min_issue_year": r.min_issue_year,
                "max_issue_year": r.max_issue_year,
                "place_geoid": "",
                "state_fips": _STATE_ABBR_TO_FIPS.get(st, ""),
                "place_fips": "",
                "place_name_gazetteer": "",
                "place_pop10": math.nan,
                "intptlat": math.nan,
                "intptlong": math.nan,
                "match_method": "unmatched",
                "match_score": math.nan,
                "candidate_count": int(len(g)),
            }
        )
    out = pd.DataFrame.from_records(rows).sort_values(["state_abbr", "city_name"]).reset_index(drop=True)
    return out


def _load_fcc_county_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): (v if isinstance(v, dict) else {}) for k, v in raw.items()}


def _save_fcc_county_cache(path: Path, payload: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attach_county_crosswalk(place_xw: pd.DataFrame, cache_dir: Path, *, refresh: bool) -> pd.DataFrame:
    cache_path = cache_dir / "fcc_place_to_county_cache.json"
    cache = {} if refresh else _load_fcc_county_cache(cache_path)
    rows: list[dict[str, Any]] = []
    for r in place_xw.itertuples(index=False):
        place_geoid = str(r.place_geoid or "").strip()
        lat = _to_float(r.intptlat)
        lon = _to_float(r.intptlong)
        payload: dict[str, Any] = {}
        if place_geoid and math.isfinite(lat) and math.isfinite(lon):
            if place_geoid in cache:
                payload = cache[place_geoid]
            else:
                url = (
                    "https://geo.fcc.gov/api/census/block/find"
                    f"?format=json&latitude={lat}&longitude={lon}&showall=false"
                )
                try:
                    raw = _request_json_with_retry(url, timeout=45.0, max_retries=3)
                    county = raw.get("County") if isinstance(raw, dict) else {}
                    payload = {
                        "county_fips": str((county or {}).get("FIPS") or ""),
                        "county_name": str((county or {}).get("name") or ""),
                        "status": str((raw or {}).get("status") or ""),
                    }
                except Exception as e:
                    payload = {"county_fips": "", "county_name": "", "status": f"error:{e}"}
                cache[place_geoid] = payload

        rows.append(
            {
                "city_key": str(r.city_key),
                "city_name": str(r.city_name),
                "state_abbr": str(r.state_abbr),
                "state_fips": str(r.state_fips or ""),
                "place_geoid": place_geoid,
                "county_fips": str(payload.get("county_fips") or ""),
                "county_name": str(payload.get("county_name") or ""),
                "county_match_status": str(payload.get("status") or ""),
            }
        )

    _save_fcc_county_cache(cache_path, cache)
    out = pd.DataFrame.from_records(rows).sort_values(["state_abbr", "city_name"]).reset_index(drop=True)
    return out


def _load_acs_place_state_year(
    *,
    state_fips: str,
    year: int,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    cache_file = cache_dir / f"acs_place_{year}_{state_fips}.json"
    if cache_file.is_file() and not refresh:
        arr = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        vars_csv = ",".join(_ACS_PLACE_VARS)
        url = (
            f"https://api.census.gov/data/{int(year)}/acs/acs5"
            f"?get=NAME,{vars_csv}&for=place:*&in=state:{state_fips}"
        )
        arr = _request_json_with_retry(url, timeout=90.0, max_retries=3)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(arr), encoding="utf-8")

    if not isinstance(arr, list) or len(arr) < 2:
        return pd.DataFrame()
    hdr = arr[0]
    body = arr[1:]
    df = pd.DataFrame(body, columns=hdr)
    df["year"] = int(year)
    return df


def _build_place_covariates(
    place_xw: pd.DataFrame,
    *,
    years: list[int],
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    mapped = place_xw[place_xw["place_fips"].astype(str).str.len() > 0].copy()
    states = sorted(set(mapped["state_fips"].astype(str).tolist()))
    pulls: list[pd.DataFrame] = []
    for year in years:
        for state_fips in states:
            try:
                df = _load_acs_place_state_year(
                    state_fips=state_fips,
                    year=year,
                    cache_dir=cache_dir,
                    refresh=refresh,
                )
            except Exception:
                continue
            if df.empty:
                continue
            pulls.append(df)
    if not pulls:
        return pd.DataFrame()
    acs = pd.concat(pulls, ignore_index=True)
    acs["state_fips"] = acs["state"].astype(str).str.zfill(2)
    acs["place_fips"] = acs["place"].astype(str).str.zfill(5)
    keep = mapped[["state_fips", "place_fips"]].drop_duplicates()
    acs = acs.merge(keep, on=["state_fips", "place_fips"], how="inner")
    acs["population_place"] = pd.to_numeric(acs["B01003_001E"], errors="coerce")
    acs["housing_units_place"] = pd.to_numeric(acs["B25001_001E"], errors="coerce")
    acs["median_household_income_place"] = pd.to_numeric(acs["B19013_001E"], errors="coerce")
    acs["median_home_value_place"] = pd.to_numeric(acs["B25077_001E"], errors="coerce")
    occ_total = pd.to_numeric(acs["B25002_001E"], errors="coerce")
    vacant = pd.to_numeric(acs["B25002_003E"], errors="coerce")
    acs["vacancy_rate_place"] = (vacant / occ_total).replace([math.inf, -math.inf], math.nan)
    acs = acs[
        [
            "year",
            "state_fips",
            "place_fips",
            "population_place",
            "housing_units_place",
            "median_household_income_place",
            "median_home_value_place",
            "vacancy_rate_place",
        ]
    ].copy()
    return acs.drop_duplicates(subset=["year", "state_fips", "place_fips"], keep="last")


def _load_acs_county_state_year(
    *,
    state_fips: str,
    year: int,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    cache_file = cache_dir / f"acs_county_profile_{year}_{state_fips}.json"
    if cache_file.is_file() and not refresh:
        arr = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        vars_csv = ",".join(_ACS_COUNTY_PROFILE_VARS)
        url = (
            f"https://api.census.gov/data/{int(year)}/acs/acs5/profile"
            f"?get=NAME,{vars_csv}&for=county:*&in=state:{state_fips}"
        )
        arr = _request_json_with_retry(url, timeout=90.0, max_retries=3)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(arr), encoding="utf-8")
    if not isinstance(arr, list) or len(arr) < 2:
        return pd.DataFrame()
    hdr = arr[0]
    body = arr[1:]
    df = pd.DataFrame(body, columns=hdr)
    df["year"] = int(year)
    return df


def _build_county_covariates(
    county_xw: pd.DataFrame,
    *,
    years: list[int],
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    xw = county_xw[county_xw["county_fips"].astype(str).str.len() == 5].copy()
    states = sorted(set(xw["state_fips"].astype(str).tolist()))
    pulls: list[pd.DataFrame] = []
    for year in years:
        for state_fips in states:
            try:
                df = _load_acs_county_state_year(
                    state_fips=state_fips,
                    year=year,
                    cache_dir=cache_dir,
                    refresh=refresh,
                )
            except Exception:
                continue
            if df.empty:
                continue
            pulls.append(df)
    if not pulls:
        return pd.DataFrame()
    acs = pd.concat(pulls, ignore_index=True)
    acs["state_fips"] = acs["state"].astype(str).str.zfill(2)
    acs["county_3"] = acs["county"].astype(str).str.zfill(3)
    acs["county_fips"] = acs["state_fips"] + acs["county_3"]
    keep = xw[["county_fips"]].drop_duplicates()
    acs = acs.merge(keep, on="county_fips", how="inner")

    acs["unemployment_rate_county_pct"] = pd.to_numeric(acs["DP03_0009PE"], errors="coerce")
    acs["per_capita_income_county"] = pd.to_numeric(acs["DP03_0088E"], errors="coerce")
    acs["industry_agriculture_share_county_pct"] = pd.to_numeric(acs["DP03_0033PE"], errors="coerce")
    acs["industry_construction_share_county_pct"] = pd.to_numeric(acs["DP03_0034PE"], errors="coerce")
    acs["industry_manufacturing_share_county_pct"] = pd.to_numeric(acs["DP03_0035PE"], errors="coerce")
    acs["industry_retail_share_county_pct"] = pd.to_numeric(acs["DP03_0037PE"], errors="coerce")
    acs["industry_prof_services_share_county_pct"] = pd.to_numeric(acs["DP03_0041PE"], errors="coerce")

    acs = acs[
        [
            "year",
            "county_fips",
            "unemployment_rate_county_pct",
            "per_capita_income_county",
            "industry_agriculture_share_county_pct",
            "industry_construction_share_county_pct",
            "industry_manufacturing_share_county_pct",
            "industry_retail_share_county_pct",
            "industry_prof_services_share_county_pct",
        ]
    ].copy()
    return acs.drop_duplicates(subset=["year", "county_fips"], keep="last")


def _parse_bps_annual_file(text: str, *, year: int) -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return pd.DataFrame()
    data_rows = rows[2:]
    out_rows: list[dict[str, Any]] = []
    for r in data_rows:
        if len(r) < 41:
            continue
        state_fips = str(r[1]).strip().zfill(2)
        place_fips = str(r[5]).strip().zfill(5)
        if not state_fips or not place_fips:
            continue
        # 99990 appears to be county/other aggregate records.
        if place_fips == "99990":
            continue
        units_1 = _to_float(r[18])
        units_2 = _to_float(r[21])
        units_3_4 = _to_float(r[24])
        units_5_plus = _to_float(r[27])
        vals = [x for x in (units_1, units_2, units_3_4, units_5_plus) if math.isfinite(x)]
        permits_total = float(sum(vals)) if vals else math.nan
        out_rows.append(
            {
                "year": int(year),
                "state_fips": state_fips,
                "place_fips": place_fips,
                "permits_units_1": units_1,
                "permits_units_2": units_2,
                "permits_units_3_4": units_3_4,
                "permits_units_5_plus": units_5_plus,
                "permits_units_total": permits_total,
            }
        )
    out = pd.DataFrame.from_records(out_rows)
    if out.empty:
        return out
    out = (
        out.groupby(["year", "state_fips", "place_fips"], dropna=False)
        .agg(
            permits_units_1=("permits_units_1", "sum"),
            permits_units_2=("permits_units_2", "sum"),
            permits_units_3_4=("permits_units_3_4", "sum"),
            permits_units_5_plus=("permits_units_5_plus", "sum"),
            permits_units_total=("permits_units_total", "sum"),
        )
        .reset_index()
    )
    return out


def _build_permits_covariates(
    place_xw: pd.DataFrame,
    *,
    years: list[int],
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    mapped = place_xw[
        (place_xw["place_fips"].astype(str).str.len() == 5)
        & (place_xw["region"].astype(str).isin(_BPS_REGION_SPEC.keys()))
    ].copy()
    if mapped.empty:
        return pd.DataFrame()
    keep = mapped[["state_fips", "place_fips"]].drop_duplicates()
    regions = sorted(set(mapped["region"].astype(str).tolist()))
    out_rows: list[pd.DataFrame] = []
    for year in years:
        for region in regions:
            reg_path, reg_prefix = _BPS_REGION_SPEC[region]
            filename = f"{reg_prefix}{int(year)}a.txt"
            cache_file = cache_dir / f"bps_place_{region}_{year}.txt"
            text: str | None = None
            if cache_file.is_file() and not refresh:
                text = cache_file.read_text(encoding="utf-8", errors="ignore")
            else:
                url = f"https://www2.census.gov/econ/bps/Place/{reg_path}/{filename}"
                try:
                    text = _request_text_with_retry(url, timeout=90.0, max_retries=2)
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(text, encoding="utf-8")
                except Exception:
                    continue
            if not text:
                continue
            p = _parse_bps_annual_file(text, year=int(year))
            if p.empty:
                continue
            out_rows.append(p)
    if not out_rows:
        return pd.DataFrame()
    permits = pd.concat(out_rows, ignore_index=True)
    permits = permits.merge(keep, on=["state_fips", "place_fips"], how="inner")
    permits = permits.drop_duplicates(subset=["year", "state_fips", "place_fips"], keep="last")
    return permits


def _make_city_year_grid(city_xw: pd.DataFrame, *, year_min: int, year_max: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in city_xw.itertuples(index=False):
        lo = max(int(year_min), int(r.min_issue_year))
        hi = min(int(year_max), int(r.max_issue_year))
        if lo > hi:
            continue
        for y in range(lo, hi + 1):
            rows.append(
                {
                    "city_key": str(r.city_key),
                    "city_name": str(r.city_name),
                    "state_abbr": str(r.state_abbr),
                    "region": str(r.region),
                    "urbanicity_proxy": str(r.urbanicity_proxy),
                    "state_fips": str(r.state_fips or ""),
                    "place_fips": str(r.place_fips or ""),
                    "place_geoid": str(r.place_geoid or ""),
                    "county_fips": str(getattr(r, "county_fips", "") or ""),
                    "county_name": str(getattr(r, "county_name", "") or ""),
                    "year": int(y),
                }
            )
    return pd.DataFrame.from_records(rows)


def _build_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dimension", "key", "variable", "n_rows", "non_missing_n", "coverage_share"])
    vars_cov = [
        "population_place",
        "housing_units_place",
        "median_household_income_place",
        "median_home_value_place",
        "vacancy_rate_place",
        "permits_units_total",
        "permits_per_1000_pop",
        "unemployment_rate_county_pct",
        "per_capita_income_county",
        "industry_agriculture_share_county_pct",
        "industry_construction_share_county_pct",
        "industry_manufacturing_share_county_pct",
        "industry_retail_share_county_pct",
        "industry_prof_services_share_county_pct",
    ]
    rows: list[dict[str, Any]] = []
    n_all = len(df)
    for v in vars_cov:
        nn = int(df[v].notna().sum()) if v in df.columns else 0
        rows.append(
            {
                "dimension": "variable",
                "key": "all_years",
                "variable": v,
                "n_rows": int(n_all),
                "non_missing_n": int(nn),
                "coverage_share": float(nn / max(1, n_all)),
            }
        )
    for y, g in df.groupby("year", dropna=False):
        n = int(len(g))
        for v in vars_cov:
            nn = int(g[v].notna().sum()) if v in g.columns else 0
            rows.append(
                {
                    "dimension": "year",
                    "key": str(int(y) if pd.notna(y) else "NA"),
                    "variable": v,
                    "n_rows": n,
                    "non_missing_n": nn,
                    "coverage_share": float(nn / max(1, n)),
                }
            )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_dir / "covariates")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    refs = _load_city_refs(run_dir)
    inferred_year_min = min(r.min_issue_year for r in refs)
    inferred_year_max = max(r.max_issue_year for r in refs)
    year_min = int(args.year_min) if int(args.year_min) > 0 else int(inferred_year_min)
    year_max = int(args.year_max) if int(args.year_max) > 0 else int(inferred_year_max)
    if year_min > year_max:
        raise SystemExit(f"Invalid year bounds: min={year_min} max={year_max}")

    acs_end_default = dt.date.today().year - 1
    acs_end_year = int(args.acs_end_year) if int(args.acs_end_year) > 0 else int(acs_end_default)
    acs_years = [y for y in range(max(int(args.acs_start_year), year_min), min(acs_end_year, year_max) + 1)]
    permit_years = [y for y in range(max(int(args.permits_start_year), year_min), year_max + 1)]

    gaz = _load_gazetteer_places(cache_dir, refresh=bool(args.refresh_cache))
    place_xw = _match_city_to_place(refs, gaz)
    place_xw_path = out_dir / "city_place_crosswalk.csv"
    place_xw.to_csv(place_xw_path, index=False, quoting=csv.QUOTE_MINIMAL)

    county_xw = _attach_county_crosswalk(place_xw, cache_dir=cache_dir, refresh=bool(args.refresh_cache))
    county_xw_path = out_dir / "city_county_crosswalk.csv"
    county_xw.to_csv(county_xw_path, index=False, quoting=csv.QUOTE_MINIMAL)

    place_cov = _build_place_covariates(
        place_xw,
        years=acs_years,
        cache_dir=cache_dir / "acs_place",
        refresh=bool(args.refresh_cache),
    )
    county_cov = _build_county_covariates(
        county_xw,
        years=acs_years,
        cache_dir=cache_dir / "acs_county",
        refresh=bool(args.refresh_cache),
    )
    permits_cov = _build_permits_covariates(
        place_xw,
        years=permit_years,
        cache_dir=cache_dir / "bps_place",
        refresh=bool(args.refresh_cache),
    )

    city_meta = place_xw.merge(
        county_xw[["city_key", "county_fips", "county_name"]],
        on="city_key",
        how="left",
    )
    city_year = _make_city_year_grid(city_meta, year_min=year_min, year_max=year_max)
    if city_year.empty:
        raise SystemExit("City-year grid is empty; check selected_panel_issues.csv and year bounds.")

    if not place_cov.empty:
        city_year = city_year.merge(place_cov, on=["year", "state_fips", "place_fips"], how="left")
    else:
        for c in (
            "population_place",
            "housing_units_place",
            "median_household_income_place",
            "median_home_value_place",
            "vacancy_rate_place",
        ):
            city_year[c] = math.nan

    if not permits_cov.empty:
        city_year = city_year.merge(permits_cov, on=["year", "state_fips", "place_fips"], how="left")
    else:
        for c in ("permits_units_1", "permits_units_2", "permits_units_3_4", "permits_units_5_plus", "permits_units_total"):
            city_year[c] = math.nan

    if not county_cov.empty:
        city_year = city_year.merge(county_cov, on=["year", "county_fips"], how="left")
    else:
        for c in (
            "unemployment_rate_county_pct",
            "per_capita_income_county",
            "industry_agriculture_share_county_pct",
            "industry_construction_share_county_pct",
            "industry_manufacturing_share_county_pct",
            "industry_retail_share_county_pct",
            "industry_prof_services_share_county_pct",
        ):
            city_year[c] = math.nan

    city_year["permits_per_1000_pop"] = (
        pd.to_numeric(city_year["permits_units_total"], errors="coerce")
        / pd.to_numeric(city_year["population_place"], errors="coerce")
        * 1000.0
    ).replace([math.inf, -math.inf], math.nan)

    cov_path = out_dir / "city_year_external_covariates.csv"
    city_year = city_year.sort_values(["city_key", "year"]).reset_index(drop=True)
    city_year.to_csv(cov_path, index=False, quoting=csv.QUOTE_MINIMAL)

    miss = _build_missingness_report(city_year)
    miss_path = out_dir / "covariate_missingness_report.csv"
    miss.to_csv(miss_path, index=False, quoting=csv.QUOTE_MINIMAL)

    prov = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "n_city_refs": int(len(refs)),
        "year_min": int(year_min),
        "year_max": int(year_max),
        "acs_start_year": int(args.acs_start_year),
        "acs_end_year": int(acs_end_year),
        "permits_start_year": int(args.permits_start_year),
        "acs_years_requested": acs_years,
        "permit_years_requested": permit_years,
        "n_city_year_rows": int(len(city_year)),
        "place_match_rate": float((place_xw["match_method"] != "unmatched").mean()) if not place_xw.empty else math.nan,
        "county_match_rate": float((county_xw["county_fips"].astype(str).str.len() == 5).mean()) if not county_xw.empty else math.nan,
        "outputs": {
            "city_place_crosswalk": str(place_xw_path),
            "city_county_crosswalk": str(county_xw_path),
            "city_year_external_covariates": str(cov_path),
            "covariate_missingness_report": str(miss_path),
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Done. "
        f"city_refs={len(refs)} "
        f"city_year_rows={len(city_year)} "
        f"place_match_rate={prov['place_match_rate']:.3f} "
        f"county_match_rate={prov['county_match_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
