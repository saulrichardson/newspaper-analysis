from __future__ import annotations

"""
Lightweight, deterministic extraction of "what is this zoning text doing?" signals.

This is intentionally heuristic (regex-based). The goal is to provide:
  - consistent, machine-usable tags for time-series analysis
  - extra steerable signal for cluster topic labeling prompts

It is not intended to be a perfect legal parser.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MechanicsTags:
    action_tags: list[str]
    dimension_tags: list[str]
    instrument_tags: list[str]
    district_tokens: list[str]


_ACTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "adoption": [
        re.compile(r"\b(adopt|adoption|adopted)\b", re.IGNORECASE),
        re.compile(r"\b(be it ordained|ordained and enacted)\b", re.IGNORECASE),
    ],
    "amendment": [
        re.compile(r"\b(amend|amends|amended|amendment)\b", re.IGNORECASE),
        re.compile(r"\b(ordinance\s+no\.?\s*\\d+).*\\b(amend)", re.IGNORECASE),
    ],
    "rezoning_map_change": [
        re.compile(r"\b(rezone|rezoning|re-zoning)\b", re.IGNORECASE),
        re.compile(r"\b(reclassif(y|ication)|reclassification)\b", re.IGNORECASE),
        re.compile(r"\b(zoning\s+map)\b", re.IGNORECASE),
        re.compile(r"\b(change\s+of\s+zone|zone\s+change)\b", re.IGNORECASE),
    ],
    "public_hearing_notice": [
        re.compile(r"\b(notice\s+of\s+(public\s+)?hearing)\b", re.IGNORECASE),
        re.compile(r"\b(public\s+hearing)\b", re.IGNORECASE),
        re.compile(r"\b(planning\s+commission)\b", re.IGNORECASE),
    ],
    "variance_special_exception": [
        re.compile(r"\b(variance|variances)\b", re.IGNORECASE),
        re.compile(r"\b(special\s+exception|special\s+use)\b", re.IGNORECASE),
        re.compile(r"\b(board\s+of\s+appeals|zoning\s+board)\b", re.IGNORECASE),
    ],
    "conditional_use_permit": [
        re.compile(r"\b(conditional\s+use|special\s+permit)\b", re.IGNORECASE),
        re.compile(r"\b(use\s+permit)\b", re.IGNORECASE),
    ],
    "enforcement_penalties": [
        re.compile(r"\b(violation|violations|unlawful)\b", re.IGNORECASE),
        re.compile(r"\b(penalt(y|ies)|fine|imprisonment)\b", re.IGNORECASE),
        re.compile(r"\b(enforcement|enforce)\b", re.IGNORECASE),
    ],
    "administration_procedure": [
        re.compile(r"\b(permit\s+required|certificate\s+of\s+occupancy)\b", re.IGNORECASE),
        re.compile(r"\b(zoning\s+administrator|building\s+inspector)\b", re.IGNORECASE),
        re.compile(r"\b(application\s+shall\s+be\s+filed)\b", re.IGNORECASE),
    ],
}


_DIMENSION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "district_definitions": [
        re.compile(r"\b(definitions?)\b", re.IGNORECASE),
        re.compile(r"\b(districts?)\b", re.IGNORECASE),
        re.compile(r"\b(zoning\s+district)\b", re.IGNORECASE),
    ],
    "permitted_uses": [
        re.compile(r"\b(permitted\s+uses?)\b", re.IGNORECASE),
        re.compile(r"\b(uses?\s+permitted)\b", re.IGNORECASE),
        re.compile(r"\b(prohibited)\b", re.IGNORECASE),
    ],
    "setbacks_yards": [
        re.compile(r"\b(setback|setbacks)\b", re.IGNORECASE),
        re.compile(r"\b(front\s+yard|rear\s+yard|side\s+yard)\b", re.IGNORECASE),
        re.compile(r"\b(yard\s+requirements?)\b", re.IGNORECASE),
    ],
    "lot_size_area": [
        re.compile(r"\b(minimum\s+lot\s+(area|size))\b", re.IGNORECASE),
        re.compile(r"\b(lot\s+area|lot\s+size)\b", re.IGNORECASE),
        re.compile(r"\b(square\s+feet|sq\.?\s*ft\.?)\b", re.IGNORECASE),
    ],
    "height_bulk": [
        re.compile(r"\b(height|stories)\b", re.IGNORECASE),
        re.compile(r"\b(bulk\s+regulations?)\b", re.IGNORECASE),
        re.compile(r"\b(floor\s+area\s+ratio|far)\b", re.IGNORECASE),
    ],
    "density_dwelling_units": [
        re.compile(r"\b(density)\b", re.IGNORECASE),
        re.compile(r"\b(dwelling\s+units?)\b", re.IGNORECASE),
        re.compile(r"\b(units?\s+per\s+acre)\b", re.IGNORECASE),
    ],
    "parking_loading": [
        re.compile(r"\b(off-?street\s+parking|parking\s+spaces?)\b", re.IGNORECASE),
        re.compile(r"\b(loading\s+spaces?|off-?street\s+loading)\b", re.IGNORECASE),
    ],
    "signs": [
        re.compile(r"\b(signs?|billboards?)\b", re.IGNORECASE),
        re.compile(r"\b(sign\s+area|sign\s+height)\b", re.IGNORECASE),
    ],
    "mobile_homes": [
        re.compile(r"\b(mobile\s+home|manufactured\s+home|trailer\s+park)\b", re.IGNORECASE),
        re.compile(r"\b(mobile\s+home\s+park)\b", re.IGNORECASE),
    ],
    "nonconforming": [
        re.compile(r"\b(nonconforming|non-conforming)\b", re.IGNORECASE),
        re.compile(r"\b(grandfather(ed)?|lawful\s+nonconforming)\b", re.IGNORECASE),
    ],
    "subdivision_platting": [
        re.compile(r"\b(subdivision|platting|plat)\b", re.IGNORECASE),
        re.compile(r"\b(preliminary\s+plat|final\s+plat)\b", re.IGNORECASE),
    ],
    "pud_site_plan": [
        re.compile(r"\b(planned\s+unit\s+development|pud)\b", re.IGNORECASE),
        re.compile(r"\b(site\s+plan)\b", re.IGNORECASE),
    ],
}


_INSTRUMENT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "full_ordinance_text": [
        re.compile(r"\b(zoning\s+ordinance)\b", re.IGNORECASE),
        re.compile(r"\b(article\s+\\d+|section\s+\\d+)\b", re.IGNORECASE),
    ],
    "notice_legal_ad": [
        re.compile(r"\b(legal\s+notice|notice\s+is\s+hereby\s+given)\b", re.IGNORECASE),
        re.compile(r"\b(published\s+in\s+the)\b", re.IGNORECASE),
    ],
    "hearing_agenda_minutes": [
        re.compile(r"\b(agenda|minutes)\b", re.IGNORECASE),
        re.compile(r"\b(council\s+meeting)\b", re.IGNORECASE),
    ],
}

# Guardrail: only emit zoning-mechanics tags when the text actually looks zoning/land-use related.
# This prevents generic government/policy/board language from being mislabeled as zoning mechanics.
_ZONING_CONTEXT_RE = re.compile(
    r"\b("
    r"zoning|rezon|re-?zoning|ordinance|land\s+use|zoning\s+map|district\s+map|"
    r"planning\s+commission|board\s+of\s+appeals|zoning\s+board|conditional\s+use|variance|"
    r"nonconforming|setback|yard\s+requirements?|mobile\s+home|trailer\s+park|"
    r"parking\s+spaces?|off-?street\s+parking|signs?|billboards?"
    r")\b",
    re.IGNORECASE,
)


_DISTRICT_TOKEN_RE = re.compile(r"\b([rcim]|ag|mf)-\d+[a-z]?\b", re.IGNORECASE)
_DISTRICT_CODE_RE = re.compile(r"\b([rcim])\s*-?\s*(\d+)\b", re.IGNORECASE)


def _any_match(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def extract_mechanics(text: str) -> MechanicsTags:
    """
    Extract coarse tags indicating what a chunk is doing or regulating.
    """
    t = (text or "").strip()
    if not t:
        return MechanicsTags(action_tags=[], dimension_tags=[], instrument_tags=[], district_tokens=[])

    if not _ZONING_CONTEXT_RE.search(t):
        return MechanicsTags(action_tags=[], dimension_tags=[], instrument_tags=[], district_tokens=[])

    action = sorted([k for k, pats in _ACTION_PATTERNS.items() if _any_match(pats, t)])
    dims = sorted([k for k, pats in _DIMENSION_PATTERNS.items() if _any_match(pats, t)])
    instr = sorted([k for k, pats in _INSTRUMENT_PATTERNS.items() if _any_match(pats, t)])

    # District tokens: R-1, C-2, M1, etc.
    districts: set[str] = set()
    for m in _DISTRICT_TOKEN_RE.finditer(t):
        districts.add(m.group(0).upper().replace(" ", ""))
    for m in _DISTRICT_CODE_RE.finditer(t):
        districts.add(f"{m.group(1).upper()}-{m.group(2)}")

    # Limit output size deterministically.
    district_tokens = sorted(districts)[:20]

    return MechanicsTags(
        action_tags=action,
        dimension_tags=dims,
        instrument_tags=instr,
        district_tokens=district_tokens,
    )


def summarize_mechanics(texts: Iterable[str], *, top_k: int = 6) -> dict[str, list[str]]:
    """
    Aggregate mechanics tags over many texts (e.g., all chunks in a cluster).
    Returns the most common tags in each category.
    """
    action_c = Counter()
    dim_c = Counter()
    instr_c = Counter()
    district_c = Counter()

    for t in texts:
        tags = extract_mechanics(t)
        action_c.update(tags.action_tags)
        dim_c.update(tags.dimension_tags)
        instr_c.update(tags.instrument_tags)
        district_c.update(tags.district_tokens)

    k = max(0, int(top_k))
    return {
        "action_tags": [x for x, _ in action_c.most_common(k)],
        "dimension_tags": [x for x, _ in dim_c.most_common(k)],
        "instrument_tags": [x for x, _ in instr_c.most_common(k)],
        "district_tokens": [x for x, _ in district_c.most_common(k)],
    }

