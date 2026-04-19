"""Unit tests for ISO optimizer JSON parsing."""
import pytest
from iso_harness.optimizer.parsing import (
    parse_json_from_response, parse_clusters_from_response,
    parse_prompts_from_response, parse_pairs_from_response,
    parse_insights_from_response, extract_reasoning,
)


class TestParseJsonFromResponse:
    def test_clean_json(self):
        assert parse_json_from_response('{"key": "value"}') == {"key": "value"}

    def test_fenced_json(self):
        text = 'Here:\n```json\n{"key": "value"}\n```\nDone.'
        assert parse_json_from_response(text) == {"key": "value"}

    def test_prose_wrapped(self):
        text = 'I think:\n{"key": "value"}\nThat should work.'
        assert parse_json_from_response(text) == {"key": "value"}

    def test_trailing_commas(self):
        text = '{"key": "value",}'
        assert parse_json_from_response(text) == {"key": "value"}

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_json_from_response(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_failure_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_json_from_response("no json here at all")


class TestParseClusters:
    def test_valid_clusters(self):
        response = '```json\n{"clusters": [{"label": "reasoning", "description": "Fails reasoning", "target_module": null, "example_failure_ids": ["ex_0"]}]}\n```'
        clusters = parse_clusters_from_response(response)
        assert len(clusters) == 1
        assert clusters[0]["label"] == "reasoning"

    def test_missing_label_raises(self):
        response = '{"clusters": [{"label": "", "description": "test"}]}'
        with pytest.raises(ValueError, match="missing label"):
            parse_clusters_from_response(response)


class TestParsePrompts:
    def test_valid_prompts(self):
        response = '{"prompts": {"qa": "Be helpful"}}'
        result = parse_prompts_from_response(response)
        assert result == {"qa": "Be helpful"}

    def test_non_string_value_raises(self):
        response = '{"prompts": {"qa": 123}}'
        with pytest.raises(ValueError, match="not a string"):
            parse_prompts_from_response(response)


class TestParsePairs:
    def test_valid_pairs(self):
        response = '{"pairs": [{"parent_a_id": "c1", "parent_b_id": "c2", "rationale": "complementary"}]}'
        pairs = parse_pairs_from_response(response)
        assert len(pairs) == 1
        assert pairs[0]["parent_a_id"] == "c1"


class TestParseInsights:
    def test_valid(self):
        response = '{"what_worked": "good", "what_failed": "bad", "recommended_changes": "fix it"}'
        result = parse_insights_from_response(response)
        assert result["what_worked"] == "good"
        assert result["what_failed"] == "bad"

    def test_missing_keys_default_empty(self):
        response = '{}'
        result = parse_insights_from_response(response)
        assert result["what_worked"] == ""


class TestExtractReasoning:
    def test_with_json(self):
        text = 'I think this because of X.\n{"key": "value"}'
        result = extract_reasoning(text)
        assert "I think" in result
        assert "{" not in result

    def test_no_json(self):
        text = "Just some text with no JSON"
        result = extract_reasoning(text)
        assert result == text
