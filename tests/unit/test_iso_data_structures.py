"""Unit tests for ISO optimizer data structures."""
import json
import pytest
from iso_harness.optimizer.candidate import Candidate, SkillCluster, ModuleTrace, MutationProposal
from iso_harness.optimizer.config import ISOConfig, VariantHooks


class TestCandidate:
    def test_default_construction(self):
        c = Candidate()
        assert c.id  # UUID generated
        assert c.parent_ids == []
        assert c.birth_round == 0
        assert c.birth_mechanism == "seed"
        assert c.prompts_by_module == {}
        assert c.score_history == []
        assert c.death_round is None

    def test_custom_construction(self):
        c = Candidate(
            id="test-id",
            parent_ids=["p1", "p2"],
            birth_round=3,
            birth_mechanism="skill_discovery",
            skill_category="reasoning",
            prompts_by_module={"qa": "Be helpful"},
        )
        assert c.id == "test-id"
        assert c.parent_ids == ["p1", "p2"]
        assert c.birth_mechanism == "skill_discovery"

    def test_unique_ids(self):
        c1, c2 = Candidate(), Candidate()
        assert c1.id != c2.id

    def test_score_history_append(self):
        c = Candidate()
        c.score_history.append((1, 0.75))
        c.score_history.append((2, 0.82))
        assert len(c.score_history) == 2
        assert c.score_history[-1] == (2, 0.82)

    # Use valid UUID4 hex strings (36 chars, matching ^[0-9a-f-]{36}$)
    _CAND_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    _RUN_ID = "00000000-0000-0000-0000-000000000001"

    def test_to_record_seed(self):
        c = Candidate(id=self._CAND_ID, birth_mechanism="seed")
        rec = c.to_record(run_id=self._RUN_ID)
        assert rec.candidate_id == c.id
        assert rec.run_id == self._RUN_ID
        assert rec.birth_mechanism == "seed"

    def test_to_record_skill_discovery(self):
        c = Candidate(id=self._CAND_ID, birth_mechanism="skill_discovery")
        rec = c.to_record(run_id=self._RUN_ID)
        assert rec.birth_mechanism == "skill_discovery"

    def test_to_record_mutation_maps_to_reflection(self):
        c = Candidate(id=self._CAND_ID, birth_mechanism="mutation_per_candidate")
        rec = c.to_record(run_id=self._RUN_ID)
        assert rec.birth_mechanism == "reflection_mutation"

    def test_to_record_cross_mutation_maps(self):
        c = Candidate(id=self._CAND_ID, birth_mechanism="cross_mutation_elitist")
        rec = c.to_record(run_id=self._RUN_ID)
        assert rec.birth_mechanism == "cross_mutation"

    def test_to_record_initial_mutation_maps(self):
        c = Candidate(id=self._CAND_ID, birth_mechanism="initial_mutation")
        rec = c.to_record(run_id=self._RUN_ID)
        assert rec.birth_mechanism == "reflection_mutation"

    def test_to_record_merge_maps(self):
        c = Candidate(id=self._CAND_ID, birth_mechanism="merge")
        rec = c.to_record(run_id=self._RUN_ID)
        assert rec.birth_mechanism == "cross_mutation"


class TestSkillCluster:
    def test_construction(self):
        sc = SkillCluster(label="reasoning", description="Fails to chain reasoning")
        assert sc.label == "reasoning"
        assert sc.target_module is None
        assert sc.example_traces == []

    def test_with_traces(self):
        trace = ModuleTrace(example_id="ex_0", score=0.3, feedback="wrong")
        sc = SkillCluster(label="test", description="test", example_traces=[trace])
        assert len(sc.example_traces) == 1


class TestModuleTrace:
    def test_defaults(self):
        t = ModuleTrace(example_id="ex_0")
        assert t.score == 0.0
        assert t.feedback == ""
        assert t.metadata == {}
        assert t.module_outputs == {}


class TestMutationProposal:
    def test_construction(self):
        mp = MutationProposal(
            candidate_id="c1",
            new_prompts={"qa": "new prompt"},
            mechanism="population_level",
        )
        assert mp.candidate_id == "c1"
        assert mp.mechanism == "population_level"


class TestVariantHooks:
    def test_construction(self):
        hooks = VariantHooks(
            prune=lambda: None,
            reflect=lambda: None,
            cross_mutate=lambda: None,
        )
        assert callable(hooks.prune)
        assert hooks.prune_ratio is None
        assert hooks.pool_size_max is None
        assert hooks.cross_mutate_only_when_improving is False

    def test_with_params(self):
        hooks = VariantHooks(
            prune=lambda: None,
            reflect=lambda: None,
            cross_mutate=lambda: None,
            prune_ratio=0.5,
            pool_size_max=4,
            cross_mutate_only_when_improving=True,
        )
        assert hooks.prune_ratio == 0.5
        assert hooks.pool_size_max == 4


class TestISOConfig:
    def _make_hooks(self):
        def prune_fn(): pass
        def reflect_fn(): pass
        def cross_mutate_fn(): pass
        return VariantHooks(prune=prune_fn, reflect=reflect_fn, cross_mutate=cross_mutate_fn)

    def test_construction(self):
        config = ISOConfig(budget=1000, seed=42, hooks=self._make_hooks())
        assert config.budget == 1000
        assert config.pool_floor == 6
        assert config.max_rounds == 20

    def test_to_dict(self):
        config = ISOConfig(budget=1000, seed=42, hooks=self._make_hooks())
        d = config.to_dict()
        assert d["budget"] == 1000
        assert d["seed"] == 42
        assert d["hooks"]["prune"] == "prune_fn"
        assert d["hooks"]["reflect"] == "reflect_fn"

    def test_to_dict_is_json_serializable(self):
        config = ISOConfig(budget=500, seed=0, hooks=self._make_hooks())
        json.dumps(config.to_dict())  # Should not raise
