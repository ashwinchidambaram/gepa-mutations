"""Unit tests for ISO feedback adapter."""
import pytest
from iso_harness.optimizer.feedback_adapter import adapt_evaluator_to_feedback_fn


class TestAdaptEvaluator:
    def test_wraps_correctly(self):
        def scorer(gold, pred):
            return 0.75, "Partially correct"

        fn = adapt_evaluator_to_feedback_fn(scorer)
        result = fn("gold_answer", "pred_answer")

        assert result["score"] == 0.75
        assert result["feedback"] == "Partially correct"
        assert result["metadata"] == {}

    def test_accepts_trace_and_pred_name(self):
        def scoring_fn(gold, pred):
            return 1.0, "Correct"

        fn = adapt_evaluator_to_feedback_fn(scoring_fn)
        result = fn("gold", "pred", trace="some_trace", pred_name="qa")
        assert result["score"] == 1.0

    def test_score_is_float(self):
        def int_scorer(gold, pred):
            return 1, "ok"  # Returns int, not float

        fn = adapt_evaluator_to_feedback_fn(int_scorer)
        result = fn("g", "p")
        assert isinstance(result["score"], float)

    def test_prompt_loader(self):
        from iso_harness.optimizer.prompts import load_prompt
        text = load_prompt("skill_discovery")
        assert "{n_failures}" in text
        assert "{modules_text}" in text
        assert "clusters" in text

    def test_all_prompts_loadable(self):
        from iso_harness.optimizer.prompts import load_prompt
        names = [
            "skill_discovery", "skill_instantiation", "initial_mutation",
            "per_candidate_reflection", "population_reflection",
            "pair_contrastive", "pair_apply", "elitist_cross",
            "reflector_guided_cross", "recombination_apply",
        ]
        for name in names:
            text = load_prompt(name)
            assert len(text) > 0, f"Prompt {name} is empty"

    def test_prompt_not_found_raises(self):
        from iso_harness.optimizer.prompts import load_prompt
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt")
