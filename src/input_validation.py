"""Request-level input contract validation for Lab 05."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "input_contract.json"
with CONTRACT_PATH.open(encoding="utf-8") as f:
    CONTRACT = json.load(f)

PREPROCESSING_ID = "tomato-rgb-resize64-bilinear-hist16x3-v1"
SEVERITY = {"pass": 0, "warning": 1, "error": 2}


def _check(rule, observed, status, action, explanation):
    return {
        "rule": rule,
        "observed": observed,
        "status": status,
        "action": action,
        "explanation": explanation,
    }


def validate_request(request):
    """Validate declared request metadata; does not open/decode input_ref."""
    checks = []

    input_ref = request.get("input_ref")
    fmt = str(request.get("format", "")).upper()
    width = request.get("width")
    height = request.get("height")

    structure_problems = []
    if not isinstance(input_ref, str) or not input_ref.strip():
        structure_problems.append("input_ref is missing or empty")
    if fmt not in CONTRACT["supported_formats"]:
        structure_problems.append(f"unsupported declared format: {fmt or 'missing'}")
    if not isinstance(width, int) or isinstance(width, bool) or width < CONTRACT["minimum_width"]:
        structure_problems.append(f"width must be an integer >= {CONTRACT['minimum_width']}")
    if not isinstance(height, int) or isinstance(height, bool) or height < CONTRACT["minimum_height"]:
        structure_problems.append(f"height must be an integer >= {CONTRACT['minimum_height']}")

    checks.append(_check(
        "input_structure",
        {"input_ref_present": bool(isinstance(input_ref, str) and input_ref.strip()),
         "format": fmt, "width": width, "height": height},
        "error" if structure_problems else "pass",
        "reject_input" if structure_problems else "continue_validation",
        "; ".join(structure_problems) if structure_problems else
        "Required reference and declared image metadata meet the request-level contract."
    ))

    source = request.get("source")
    source_ok = source in CONTRACT["approved_sources"]
    checks.append(_check(
        "supported_source", source, "pass" if source_ok else "warning",
        "continue_validation" if source_ok else "route_for_review",
        "Source is within the controlled prototype scope." if source_ok else
        "Source is not in the controlled PlantVillage-style evidence scope; phone/grower images are not validated."
    ))

    review_requested = bool(request.get("requires_review", False))
    checks.append(_check(
        "review_condition", review_requested,
        "warning" if review_requested else "pass",
        "route_for_review" if review_requested else "continue_validation",
        "Request is explicitly marked for human review." if review_requested else
        "No additional review flag was supplied."
    ))

    overall = max((c["status"] for c in checks), key=lambda s: SEVERITY[s])
    action = {"pass": "run_prediction", "warning": "route_for_review", "error": "reject_input"}[overall]
    return {
        "checks": checks,
        "overall_status": overall,
        "action": action,
        "prediction_allowed": overall == "pass",
        "preprocessing_id": PREPROCESSING_ID,
        "contract_version": CONTRACT["contract_version"],
    }
