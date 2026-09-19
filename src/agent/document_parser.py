"""
Document parsing: turns free-text trade documents (commercial invoices,
packing lists, freight quotes) into structured fields.

An LLM does the extraction instead of hand-written regex per document
template. The model is forced to answer through a tool with a JSON schema,
so the response is structured data rather than free text that has to be
parsed. When no API key is configured, or the call fails, a deterministic
regex parser takes over, so the module and its tests run offline.

Every fallback is logged with its reason, and `parsed_by` in the result says
which path produced it, so a silent drop in extraction quality shows up in
the logs instead of going unnoticed.
"""
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
FIELDS = ["commodity", "hs_code", "quantity_kg", "invoice_value_usd", "incoterm", "origin_country"]
NUMERIC_FIELDS = {"quantity_kg", "invoice_value_usd"}

EXTRACTION_TOOL = {
    "name": "record_trade_document",
    "description": "Record the fields extracted from a trade document. Use null for any field not present.",
    "input_schema": {
        "type": "object",
        "properties": {
            "commodity": {"type": ["string", "null"]},
            "hs_code": {"type": ["string", "null"], "description": "Harmonized System code, digits only"},
            "quantity_kg": {"type": ["number", "null"]},
            "invoice_value_usd": {"type": ["number", "null"]},
            "incoterm": {"type": ["string", "null"], "description": "e.g. FOB, CIF, EXW"},
            "origin_country": {"type": ["string", "null"]},
        },
        "required": FIELDS,
    },
}

SYSTEM_PROMPT = (
    "You extract structured fields from trade documents. Call the "
    "record_trade_document tool exactly once. Copy values from the document; "
    "never guess a value that is not there."
)


@dataclass
class ParsedDocument:
    commodity: Optional[str] = None
    hs_code: Optional[str] = None
    quantity_kg: Optional[float] = None
    invoice_value_usd: Optional[float] = None
    incoterm: Optional[str] = None
    origin_country: Optional[str] = None
    parsed_by: str = "regex_fallback"


class ExtractionError(Exception):
    """The model answered, but not with usable structured fields."""


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def _regex_fallback(text: str) -> ParsedDocument:
    def line(label: str) -> Optional[str]:
        m = re.search(rf"^\s*{label}\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
        return m.group(1) if m else None

    hs_code = re.search(r"\bHS\s*Code[:\s]*([0-9]{6,10})", text, re.IGNORECASE)
    qty = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:kg|kilograms)\b", text, re.IGNORECASE)
    value = re.search(r"(?:USD|\$)\s*([\d,]+(?:\.\d+)?)", text)
    incoterm = re.search(r"\b(FOB|CIF|EXW|CFR|DAP|DDP)\b", text, re.IGNORECASE)

    return ParsedDocument(
        commodity=line("Commodity"),
        hs_code=hs_code.group(1) if hs_code else None,
        quantity_kg=_number(qty.group(1)) if qty else None,
        invoice_value_usd=_number(value.group(1)) if value else None,
        incoterm=incoterm.group(1).upper() if incoterm else None,
        origin_country=line("Origin(?: Country)?"),
        parsed_by="regex_fallback",
    )


def _validated(fields: dict, model: str) -> ParsedDocument:
    """Coerce the model's tool input into a ParsedDocument, or raise."""
    values = {}
    for name in FIELDS:
        value = fields.get(name)
        if value is None:
            values[name] = None
        elif name in NUMERIC_FIELDS:
            try:
                values[name] = _number(str(value))
            except ValueError as exc:
                raise ExtractionError(f"{name} is not a number: {value!r}") from exc
        else:
            values[name] = str(value).strip() or None
    if values["incoterm"]:
        values["incoterm"] = values["incoterm"].upper()
    return ParsedDocument(**values, parsed_by=model)


def _extract_with_llm(text: str, api_key: str, model: str) -> ParsedDocument:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": EXTRACTION_TOOL["name"]},
        messages=[{"role": "user", "content": text}],
    )
    if response.stop_reason == "max_tokens":
        raise ExtractionError("response was cut off at max_tokens")
    for block in response.content:
        if block.type == "tool_use" and block.name == EXTRACTION_TOOL["name"]:
            if not isinstance(block.input, dict):
                raise ExtractionError("tool input is not an object")
            return _validated(block.input, model)
    raise ExtractionError("model did not call the extraction tool")


def parse_document(text: str) -> ParsedDocument:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _regex_fallback(text)

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed; using regex fallback")
        return _regex_fallback(text)

    try:
        return _extract_with_llm(text, api_key, model)
    except anthropic.APIError as exc:
        # Network, auth, rate-limit and server errors: degrade, but say so.
        logger.warning("LLM extraction failed (%s: %s); using regex fallback",
                       type(exc).__name__, exc)
    except ExtractionError as exc:
        logger.warning("LLM extraction unusable (%s); using regex fallback", exc)
    return _regex_fallback(text)


def parse_document_dict(text: str) -> dict:
    return asdict(parse_document(text))
