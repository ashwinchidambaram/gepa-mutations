"""Unit tests for ISO optimizer helpers."""
import random
import pytest
import dspy
from iso_harness.optimizer.helpers import (
    compute_top3_mean, top_k_examples, bottom_k_examples,
    find_candidate, get_all_example_ids, is_multi_module,
    apply_candidate_prompts, ensure_example_ids, sample_minibatches,
)
from iso_harness.optimizer.candidate import Candidate


class TestComputeTop3Mean:
    def test_normal(self):
        scores = {
            "c1": {"mean": 0.9},
            "c2": {"mean": 0.8},
            "c3": {"mean": 0.7},
            "c4": {"mean": 0.5},
        }
        result = compute_top3_mean(scores)
        assert abs(result - 0.8) < 1e-6  # mean(0.9, 0.8, 0.7) = 0.8

    def test_fewer_than_3(self):
        scores = {"c1": {"mean": 0.9}, "c2": {"mean": 0.7}}
        result = compute_top3_mean(scores)
        assert abs(result - 0.8) < 1e-6

    def test_empty(self):
        assert compute_top3_mean({}) == 0.0


class TestTopBottomK:
    def test_top_k(self):
        examples = {"a": 0.9, "b": 0.3, "c": 0.7}
        result = top_k_examples(examples, 2)
        assert result[0] == ("a", 0.9)
        assert result[1] == ("c", 0.7)

    def test_bottom_k(self):
        examples = {"a": 0.9, "b": 0.3, "c": 0.7}
        result = bottom_k_examples(examples, 2)
        assert result[0] == ("b", 0.3)
        assert result[1] == ("c", 0.7)


class TestFindCandidate:
    def test_found(self):
        pool = [Candidate(id="c1"), Candidate(id="c2"), Candidate(id="c3")]
        assert find_candidate(pool, "c2").id == "c2"

    def test_not_found(self):
        pool = [Candidate(id="c1")]
        assert find_candidate(pool, "missing") is None


class TestGetAllExampleIds:
    def test_collects_all(self):
        scores = {
            "c1": {"per_example": {"ex_0": 0.5, "ex_1": 0.6}},
            "c2": {"per_example": {"ex_1": 0.7, "ex_2": 0.8}},
        }
        ids = get_all_example_ids(scores)
        assert ids == {"ex_0", "ex_1", "ex_2"}


class TestIsMultiModule:
    def test_single(self):
        class SingleModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict("input -> output")
        assert is_multi_module(SingleModule()) is False

    def test_multi(self):
        class MultiModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.query = dspy.Predict("input -> query")
                self.answer = dspy.Predict("query -> answer")
        assert is_multi_module(MultiModule()) is True


class TestApplyCandidatePrompts:
    def test_patches_copy(self):
        class TestModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.qa = dspy.Predict("question -> answer")

        student = TestModule()
        original_instructions = student.qa.signature.instructions

        c = Candidate(prompts_by_module={"qa": "You are a helpful expert."})
        patched = apply_candidate_prompts(student, c)

        # Patched has new instructions
        for name, pred in patched.named_predictors():
            if name == "qa":
                assert pred.signature.instructions == "You are a helpful expert."

        # Original unchanged
        assert student.qa.signature.instructions == original_instructions

    def test_partial_update(self):
        """Only modules in prompts_by_module are updated."""
        class TwoModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.step1 = dspy.Predict("input -> intermediate")
                self.step2 = dspy.Predict("intermediate -> output")

        student = TwoModule()
        c = Candidate(prompts_by_module={"step1": "New step1 prompt"})
        patched = apply_candidate_prompts(student, c)

        predictors = dict(patched.named_predictors())
        assert predictors["step1"].signature.instructions == "New step1 prompt"
        # step2 should be unchanged (default DSPy instructions)
        assert len(predictors["step2"].signature.instructions) > 0


class TestEnsureExampleIds:
    def test_assigns_ids(self):
        class FakeExample:
            pass
        examples = [FakeExample(), FakeExample()]
        ensure_example_ids(examples)
        assert examples[0].id == "ex_0"
        assert examples[1].id == "ex_1"

    def test_preserves_existing(self):
        class FakeExample:
            def __init__(self, id):
                self.id = id
        examples = [FakeExample("custom_0"), FakeExample("custom_1")]
        ensure_example_ids(examples)
        assert examples[0].id == "custom_0"


class TestSampleMinibatches:
    def test_correct_shape(self):
        data = list(range(20))
        rng = random.Random(42)
        batches = sample_minibatches(data, n_batches=4, batch_size=5, rng=rng)
        assert len(batches) == 4
        for batch in batches:
            assert len(batch) == 5

    def test_disjoint(self):
        data = list(range(20))
        rng = random.Random(42)
        batches = sample_minibatches(data, n_batches=4, batch_size=5, rng=rng)
        all_items = []
        for batch in batches:
            all_items.extend(batch)
        assert len(all_items) == len(set(all_items))  # all unique

    def test_insufficient_data_raises(self):
        data = list(range(5))
        rng = random.Random(42)
        with pytest.raises(ValueError, match="need at least"):
            sample_minibatches(data, n_batches=3, batch_size=3, rng=rng)

    def test_reproducible(self):
        data = list(range(20))
        b1 = sample_minibatches(data, 2, 5, random.Random(42))
        b2 = sample_minibatches(data, 2, 5, random.Random(42))
        assert b1 == b2
