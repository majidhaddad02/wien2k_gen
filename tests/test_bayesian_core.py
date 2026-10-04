"""Tests for Bayesian search-space physics constraints in define_search_space."""

from forge.core.constants import HARD_ELEMENTS
from forge.optimizer.bayesian.core import define_search_space
from forge.types import Wien2kFlags


def _structure(*atomic_numbers: int) -> dict:
    return {"atoms": [{"z_num": z} for z in atomic_numbers]}


def _flags(*, is_soc: bool = False) -> Wien2kFlags:
    return Wien2kFlags(is_soc=is_soc)


class TestHardElementsCanonicalSet:
    def test_canonical_set_is_n_o_f_only(self):
        assert HARD_ELEMENTS == frozenset({7, 8, 9})
        assert 1 not in HARD_ELEMENTS
        assert 16 not in HARD_ELEMENTS
        assert 17 not in HARD_ELEMENTS


class TestDefineSearchSpacePhysics:
    def test_hard_element_floors_rkmax_bounds_without_soc_or_scf(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "forge.backends.wien2k.parsers.detect_wien2k_flags",
            lambda: _flags(is_soc=False),
        )
        monkeypatch.setattr(
            "forge.core.workflow_executor.detect_system_type",
            lambda case_dir=".": "unknown",
        )
        space = define_search_space(_structure(8), case_dir=str(tmp_path))
        assert space["rkmax"]["bounds"][0] == 7.0
        assert space["rkmax"]["default"] == 7.0

    def test_soc_alone_floors_rkmax_bounds(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "forge.backends.wien2k.parsers.detect_wien2k_flags",
            lambda: _flags(is_soc=True),
        )
        monkeypatch.setattr(
            "forge.core.workflow_executor.detect_system_type",
            lambda case_dir=".": "insulator",
        )
        space = define_search_space(_structure(14), case_dir=str(tmp_path))
        assert space["rkmax"]["bounds"][0] == 7.0

    def test_metal_constrains_mixing_and_kpoints(self, monkeypatch):
        monkeypatch.setattr(
            "forge.backends.wien2k.parsers.detect_wien2k_flags",
            lambda: _flags(is_soc=False),
        )
        monkeypatch.setattr(
            "forge.core.workflow_executor.detect_system_type",
            lambda case_dir=".": "metal",
        )
        space = define_search_space(_structure(14))
        assert space["mixing_beta"]["bounds"][1] == 0.30
        assert space["kpoint_density"]["bounds"][0] == 1000

    def test_unknown_treated_as_metal_for_bounds(self, monkeypatch):
        monkeypatch.setattr(
            "forge.backends.wien2k.parsers.detect_wien2k_flags",
            lambda: _flags(is_soc=False),
        )
        monkeypatch.setattr(
            "forge.core.workflow_executor.detect_system_type",
            lambda case_dir=".": "unknown",
        )
        space = define_search_space(_structure(14))
        assert space["mixing_beta"]["bounds"][1] == 0.30
        assert space["kpoint_density"]["bounds"][0] == 1000

    def test_insulator_keeps_unconstrained_mixing_and_kpoints(self, monkeypatch):
        monkeypatch.setattr(
            "forge.backends.wien2k.parsers.detect_wien2k_flags",
            lambda: _flags(is_soc=False),
        )
        monkeypatch.setattr(
            "forge.core.workflow_executor.detect_system_type",
            lambda case_dir=".": "insulator",
        )
        space = define_search_space(_structure(14))
        assert space["mixing_beta"]["bounds"] == (0.05, 1.0)
        assert space["kpoint_density"]["bounds"][0] == 100
        assert space["rkmax"]["bounds"][0] == 5.0

    def test_add_physics_priors_removed_from_public_api(self):
        import forge.optimizer.bayesian as bayesian_pkg

        assert not hasattr(bayesian_pkg, "add_physics_priors")
        assert "add_physics_priors" not in bayesian_pkg.__all__
