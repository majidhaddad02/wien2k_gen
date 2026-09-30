"""Tests for forge.backends.elpa_selector — solver routing and compile flags."""

from unittest.mock import patch

from forge.backends.elpa_selector import (
    get_recommended_wien2k_compile_flags,
    select_eigensolver,
)
from forge.core.topology import factorize_blacs_grid


class TestSelectEigensolverSocDoubling:
    @patch("forge.backends.elpa_selector.check_elpa_available", return_value=False)
    def test_soc_nmat_1500_uses_moderate_not_lapack(self, _elpa):
        soc = select_eigensolver(nmat=1500, nkpt=8, is_soc=True, gpu_available=False)
        non_soc = select_eigensolver(nmat=3000, nkpt=8, is_soc=False, gpu_available=False)
        assert soc.recommended_solver != "LAPACK"
        assert soc.recommended_solver == non_soc.recommended_solver
        assert non_soc.recommended_solver == "ScaLAPACK"

    @patch("forge.backends.elpa_selector.check_elpa_available", return_value=False)
    def test_soc_nmat_999_still_lapack(self, _elpa):
        result = select_eigensolver(nmat=999, nkpt=8, is_soc=True, gpu_available=False)
        assert result.recommended_solver == "LAPACK"

    @patch("forge.backends.elpa_selector.check_elpa_available", return_value=False)
    def test_non_soc_nmat_1500_still_lapack(self, _elpa):
        result = select_eigensolver(nmat=1500, nkpt=8, is_soc=False, gpu_available=False)
        assert result.recommended_solver == "LAPACK"

    @patch("forge.backends.elpa_selector.check_elpa_available", return_value=False)
    def test_recommended_grid_uses_balanced_factorization(self, _elpa):
        result = select_eigensolver(
            nmat=3000, nkpt=8, is_soc=False, gpu_available=False, total_ranks=100
        )
        assert result.recommended_grid == factorize_blacs_grid(100)
        assert result.recommended_grid == (10, 10)


class TestScaLAPACKCompileFlags:
    @patch("forge.backends.elpa_selector.check_mkl_available", return_value=False)
    def test_plain_scalapack_without_mkl_emits_base_flags(self, _mkl):
        flags = get_recommended_wien2k_compile_flags("ScaLAPACK", "xeon", "openmpi")
        assert "-DSCALAPACK" in flags["cflags"]
        assert "-DSCALAPACK" in flags["configure_opts"]
        assert "-DUSE_SCALAPACK_OPTIMIZED" not in flags["cflags"]
        assert "-DMKL_ILP64" not in flags["cflags"]
        assert "mkl_scalapack" not in flags["ldflags"]

    @patch("forge.backends.elpa_selector.check_mkl_available", return_value=False)
    def test_optimized_scalapack_without_mkl_adds_optimized_define(self, _mkl):
        flags = get_recommended_wien2k_compile_flags(
            "ScaLAPACK_OPTIMIZED", "xeon", "openmpi"
        )
        assert "-DSCALAPACK" in flags["cflags"]
        assert "-DUSE_SCALAPACK_OPTIMIZED" in flags["cflags"]
        assert "-DMKL_ILP64" not in flags["cflags"]

    @patch("forge.backends.elpa_selector.check_mkl_available", return_value=True)
    def test_scalapack_with_mkl_adds_ilp64(self, _mkl):
        flags = get_recommended_wien2k_compile_flags("ScaLAPACK", "xeon", "openmpi")
        assert "-DSCALAPACK" in flags["cflags"]
        assert "-DUSE_SCALAPACK_OPTIMIZED" in flags["cflags"]
        assert "-DMKL_ILP64" in flags["cflags"]
        assert "-lmkl_scalapack_ilp64" in flags["ldflags"]
