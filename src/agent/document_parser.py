"""
Agentic document parsing: turns free-text trade documents (commercial
invoices, packing lists, freight quotes) into structured JSON.

This is the "agentic coding / AI-assisted programming" piece of the stack --
an LLM call does the extraction instead of hand-written regex per document
template. Falls back to a deterministic regex parser when no API key is
configured, so the module (and its tests) run offline / in CI without
network access or secrets.
"""
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Optional

FIELDS = ["commodity", "hs_code", "quantity_kg", "invoice_value_usd", "incoterm", "origin_country"]


@dataclass
class ParsedDocument:
    commodity: Optional[str] = None
    hs_code: Optional[str] = None
    quantity_kg: Optional[float] = None
    invoice_value_usd: Optional[float] = None
    incoterm: Optional[str] = None
    origin_country: Optional[str] = None
    parsed_by: str = "regex_fallback"


SYSTEM_PROMPT = (
    "You extract structured trade-document fields from raw text. "
    "Respond ONLY with minified JSON matching this schema, no prose, no markdown fences: "
    '{"commodity": string|null, "hs_code": string|null, "quantity_kg": number|null, '
    '"invoice_value_usd": number|null, "incoterm": string|null, "origin_country": string|null}'
)


def _regex_fallback(text: str) -> ParsedDocument:
    hs_code = re.search(r"\bHS\s*Code[:\s]*([0-9]{6,10})", text, re.IGNORECASE)
    qty = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:kg|KG|kilograms)", text)
    value = re.search(r"(?:USD|\$)\s*([\d,]+(?:\.\d+)?)", text)
    incoterm = re.search(r"\b(FOB|CIF|EXW|CFR|DAP|DDP)\b", text, re.IGNORECASE)

    return ParsedDocument(
        hs_code=hs_code.group(1) if hs_code else None,
        quantity_kg=float(qty.group(1).replace(",", "")) if qty else None,
        invoice_value_usd=float(value.group(1).replace(",", "")) if value else None,
        incoterm=incoterm.group(1).upper() if incoterm else None,
        parsed_by="regex_fallback",
    )


def parse_document(text: str) -> ParsedDocument:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _regex_fallback(text)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        raw = "".join(block.text for block in response.content if block.type == "text")
        data = json.loads(raw)
        data["parsed_by"] = "claude-sonnet-4-6"
        return ParsedDocument(**{k: data.get(k) for k in FIELDS}, parsed_by="claude-sonnet-4-6")
    except Exception:
        # Any API/parsing failure degrades gracefully instead of 500ing the endpoint
        return _regex_fallback(text)


def parse_document_dict(text: str) -> dict:
    return asdict(parse_document(text))
