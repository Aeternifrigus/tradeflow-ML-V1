from src.agent.document_parser import parse_document, _regex_fallback


SAMPLE_INVOICE = """
COMMERCIAL INVOICE
Commodity: Organic Psyllium Husk
HS Code: 12129921
Quantity: 21,000 kg
Incoterm: FOB
Invoice Value: USD 26,400.00
Origin: India
"""


def test_regex_fallback_extracts_hs_code():
    result = _regex_fallback(SAMPLE_INVOICE)
    assert result.hs_code == "12129921"


def test_regex_fallback_extracts_quantity():
    result = _regex_fallback(SAMPLE_INVOICE)
    assert result.quantity_kg == 21000.0


def test_regex_fallback_extracts_incoterm():
    result = _regex_fallback(SAMPLE_INVOICE)
    assert result.incoterm == "FOB"


def test_parse_document_uses_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = parse_document(SAMPLE_INVOICE)
    assert result.parsed_by == "regex_fallback"
    assert result.hs_code == "12129921"
