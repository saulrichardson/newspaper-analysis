from __future__ import annotations

from typing import Any

# Schema versions for cluster topic labeling.
#
# v1: topic_name/topic_description/keywords (legacy)
# v2: adds an LLM-produced "mechanics" structure for zoning/land-use mechanisms
SCHEMA_VERSIONS = ("v1", "v2")

# v2: controlled vocabularies for structured mechanics tags.
MECHANICS_ZONING_RELEVANCE = ("zoning", "mixed", "non_zoning")

MECHANICS_ACTION_TAGS = (
    "adopt_ordinance",
    "amend_text",
    "amend_map_rezone",
    "annexation",
    "create_district",
    "create_overlay",
    "definitions",
    "nonconforming",
    "administration",
    "public_hearing_notice",
    "appeals",
    "variance",
    "conditional_use",
    "special_exception",
    "site_plan",
    "subdivision_plat",
    "pud",
    "enforcement_penalties",
    "permit_licensing",
    "fees",
)

MECHANICS_DIMENSION_TAGS = (
    # Use regulation
    "permitted_uses",
    "conditional_uses",
    "prohibited_uses",
    # Dimensional / bulk standards
    "setbacks",
    "yards",
    "lot_area",
    "lot_width",
    "building_height",
    "floor_area_ratio",
    "lot_coverage",
    "density",
    "dwelling_units",
    # Development standards
    "parking",
    "loading",
    "signs",
    "billboards",
    "landscaping_buffers",
    "fences_walls",
    "accessory_structures",
    "home_occupations",
    # Housing types / districts
    "mobile_homes",
    "trailer_parks",
    "manufactured_housing",
    "districts",
    "definitions_terms",
    "nonconforming_uses",
    # Processes often present in zoning corpora
    "subdivision_standards",
    "site_plan_requirements",
    "pud_standards",
    # Common overlays / special topics
    "floodplain",
    "airport",
    "adult_business",
    "alcohol",
    "historic_district",
    "utilities_sewer_water",
)

MECHANICS_DECISION_BODY_TAGS = (
    "city_council",
    "planning_commission",
    "zoning_board",
    "board_of_adjustment",
    "board_of_appeals",
    "county_commission",
    "town_board",
    "zoning_administrator",
    "unknown",
)

MECHANICS_INSTRUMENT_TAGS = (
    "ordinance",
    "resolution",
    "zoning_code",
    "zoning_map",
    "petition",
    "application",
    "permit",
    "notice",
    "hearing",
    "minutes",
    "report_recommendation",
    "affidavit_publication",
)


def openai_text_config(fmt: str, *, schema_version: str) -> dict[str, Any] | None:
    """
    Build OpenAI Responses API `text.format` config.

    fmt:
      - json_object: weaker enforcement (model still must follow prompt)
      - json_schema: strict JSON Schema validation (preferred)
    """

    if fmt == "json_object":
        return {"format": {"type": "json_object"}}

    if fmt != "json_schema":
        raise ValueError(f"Unknown openai_text_format: {fmt}")

    if schema_version not in SCHEMA_VERSIONS:
        raise ValueError(f"Unknown schema_version: {schema_version} (expected one of {SCHEMA_VERSIONS})")

    if schema_version == "v1":
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cluster_id": {"type": "integer"},
                "topic_name": {"type": "string"},
                "topic_description": {"type": "string"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 4,
                    "maxItems": 20,
                },
            },
            "required": ["cluster_id", "topic_name", "topic_description", "keywords"],
        }
        return {
            "format": {
                "type": "json_schema",
                "name": "cluster_topic_label_v1",
                "schema": schema,
                "strict": True,
            }
        }

    # v2: include structured mechanics tags for zoning/land-use mechanisms.
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cluster_id": {"type": "integer"},
            "topic_name": {"type": "string"},
            "topic_description": {"type": "string"},
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 4,
                "maxItems": 20,
            },
            "zoning_relevance": {"type": "string", "enum": list(MECHANICS_ZONING_RELEVANCE)},
            "mechanics": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action_tags": {"type": "array", "items": {"type": "string", "enum": list(MECHANICS_ACTION_TAGS)}},
                    "dimension_tags": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(MECHANICS_DIMENSION_TAGS)},
                    },
                    "decision_body_tags": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(MECHANICS_DECISION_BODY_TAGS)},
                    },
                    "instrument_tags": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(MECHANICS_INSTRUMENT_TAGS)},
                    },
                    "district_tokens": {"type": "array", "items": {"type": "string"}, "maxItems": 40},
                    "mechanism_phrases": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "tag": {"type": "string"},
                                "example_numbers": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                            },
                            "required": ["tag", "example_numbers"],
                        },
                        "maxItems": 80,
                    },
                },
                "required": [
                    "action_tags",
                    "dimension_tags",
                    "decision_body_tags",
                    "instrument_tags",
                    "district_tokens",
                    "mechanism_phrases",
                    "evidence",
                ],
            },
        },
        "required": ["cluster_id", "topic_name", "topic_description", "keywords", "zoning_relevance", "mechanics"],
    }
    return {
        "format": {
            "type": "json_schema",
            "name": "cluster_topic_label_v2_mechanics",
            "schema": schema,
            "strict": True,
        }
    }

