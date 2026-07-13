"""
Test Package Initialization for FORGE.
Provides a clean, side-effect-free entry point for pytest collection.
All test fixtures, mocks, and utilities are scoped within their respective modules
to ensure isolation, reproducibility, and CI/CD compatibility.
"""
# Intentionally left minimal to avoid import side-effects or namespace pollution.
# pytest discovers tests automatically via naming convention (test_*.py).