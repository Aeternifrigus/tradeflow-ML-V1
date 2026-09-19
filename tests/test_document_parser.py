import logging
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from src.agent import document_parser
from src.agent.document_parser import EXTRACTION_TOOL, _regex_fallback, parse_document

SAMPLE_INVOICE = """
COMMERCIAL INVOICE
Commodity: Organic Psyllium Husk
HS Code: 12129921
Quantity: 21,000 kg
Incoterm: FOB
Invoice Value: USD 26,400.00
Origin: India
"""

GOOD_FIELDS = {
    "commodity": "Organic Psyllium Husk",
    "hs_code": "12129921",
    "quantity_kg": 21000,
    "invoice_value_usd": "26,400.00",
    "incoterm": "fob",
    "origin_country": "India",
}


def test_regex_fallback_extracts_all_fields():
    result = _regex_fallback(SAMPLE_INVOICE)
    assert result.commodity == "Organic Psyllium Husk"
    assert result.hs_code == "12129921"
    assert result.quantity_kg == 21000.0
    assert result.invoice_value_usd == 26400.0
    assert result.incoterm == "FOB"
    assert result.origin_country == "India"


def test_parse_document_uses_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = parse_document(SAMPLE_INVOICE)
    assert result.parsed_by == "regex_fallback"
    assert result.hs_code == "12129921"


# --- LLM path, with the Anthropic client replaced by a fake -----------------

def fake_client(monkeypatch, *, response=None, error=None, calls=None):
    class FakeMessages:
        def create(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            if error is not None:
                raise error
            return response

    class FakeClient:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)


def tool_response(fields, stop_reason="tool_use", name=EXTRACTION_TOOL["name"]):
    block = SimpleNamespace(type="tool_use", name=name, input=fields)
    return SimpleNamespace(stop_reason=stop_reason, content=[block])


def test_llm_result_is_validated_and_normalised(monkeypatch):
    calls = []
    fake_client(monkeypatch, response=tool_response(GOOD_FIELDS), calls=calls)
    result = parse_document(SAMPLE_INVOICE)

    assert result.parsed_by == document_parser.DEFAULT_MODEL
    assert result.invoice_value_usd == 26400.0
    assert result.quantity_kg == 21000.0
    assert result.incoterm == "FOB"
    # The model is forced to answer through the extraction tool.
    assert calls[0]["tool_choice"] == {"type": "tool", "name": EXTRACTION_TOOL["name"]}


def test_model_name_comes_from_the_environment(monkeypatch):
    calls = []
    fake_client(monkeypatch, response=tool_response(GOOD_FIELDS), calls=calls)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test-model")
    result = parse_document(SAMPLE_INVOICE)
    assert calls[0]["model"] == "claude-test-model"
    assert result.parsed_by == "claude-test-model"


@pytest.mark.parametrize("case, response, message", [
    ("truncated", tool_response(GOOD_FIELDS, stop_reason="max_tokens"), "max_tokens"),
    ("no tool call", SimpleNamespace(stop_reason="end_turn",
                                     content=[SimpleNamespace(type="text", text="hi")]),
     "did not call"),
    ("bad number", tool_response({**GOOD_FIELDS, "quantity_kg": "lots"}), "not a number"),
])
def test_unusable_llm_answers_fall_back_and_are_logged(monkeypatch, caplog, case, response, message):
    fake_client(monkeypatch, response=response)
    with caplog.at_level(logging.WARNING, logger="src.agent.document_parser"):
        result = parse_document(SAMPLE_INVOICE)
    assert result.parsed_by == "regex_fallback"
    assert result.hs_code == "12129921"
    assert message in caplog.text


def test_api_errors_fall_back_and_are_logged(monkeypatch, caplog):
    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    fake_client(monkeypatch, error=error)
    with caplog.at_level(logging.WARNING, logger="src.agent.document_parser"):
        result = parse_document(SAMPLE_INVOICE)
    assert result.parsed_by == "regex_fallback"
    assert "APIConnectionError" in caplog.text


def test_programming_errors_are_not_hidden(monkeypatch):
    """Only API and extraction failures fall back; a bug must still surface."""
    fake_client(monkeypatch, error=RuntimeError("bug"))
    with pytest.raises(RuntimeError, match="bug"):
        parse_document(SAMPLE_INVOICE)
