"""Tests for tools/codebase_map (stdlib-only AST analysis)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from codebase_map.extractor import (  # noqa: E402
    analyze_repository,
    classify_import,
    parse_python_file,
    path_to_module,
    resolve_relative,
)
from codebase_map.stdlib_modules import is_stdlib  # noqa: E402
from codebase_map.textutil import mermaid_id  # noqa: E402


class ExtractorTests(unittest.TestCase):
    def test_path_to_module(self) -> None:
        self.assertEqual(path_to_module("src/forge/cli.py"), "forge.cli")
        self.assertEqual(path_to_module("src/forge/backends/__init__.py"), "forge.backends")
        self.assertEqual(path_to_module("tests/test_cli.py"), "tests.test_cli")

    def test_resolve_relative(self) -> None:
        self.assertEqual(
            resolve_relative("forge.cli_commands.generate", 1, "base"),
            "forge.cli_commands.base",
        )
        self.assertEqual(
            resolve_relative("forge.cli_commands.generate", 2, "config"),
            "forge.config",
        )
        self.assertEqual(
            resolve_relative("forge.utils", 1, "atomic_write", is_package=True),
            "forge.utils.atomic_write",
        )

    def test_mermaid_id_strips_special_chars(self) -> None:
        self.assertEqual(mermaid_id("forge.cli:main"), "forge_cli_main")
        self.assertEqual(mermaid_id("forge.backends:<module>"), "forge_backends_module")
        self.assertEqual(mermaid_id("9cmd"), "n_9cmd")

    def test_stdlib_and_internal_classify(self) -> None:
        self.assertTrue(is_stdlib("json"))
        self.assertTrue(is_stdlib("os.path"))
        self.assertEqual(classify_import("forge.cli", {"forge.cli"}), "internal")
        self.assertEqual(classify_import("numpy", {"forge.cli"}), "third_party")
        self.assertEqual(classify_import("json", {"forge.cli"}), "stdlib")

    def test_parse_extracts_import_class_call(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            src = tmp_path / "src" / "forge"
            src.mkdir(parents=True)
            (src / "other.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
            target = src / "mod.py"
            target.write_text(
                dedent(
                    '''
                    """Sample."""
                    import json
                    from .other import helper

                    class Box:
                        def run(self):
                            helper()
                            json.dumps({})

                    if __name__ == "__main__":
                        Box().run()
                    '''
                ),
                encoding="utf-8",
            )
            rec = parse_python_file(target, tmp_path, {"forge.mod", "forge.other"})
            self.assertIsNone(rec.syntax_error)
            self.assertTrue(rec.has_main_guard)
            self.assertEqual(rec.module, "forge.mod")
            self.assertTrue(any(c.name == "Box" for c in rec.classes))
            cats = {i.category for i in rec.imports}
            self.assertIn("stdlib", cats)
            self.assertIn("internal", cats)
            self.assertTrue(any(c.callee_name == "helper" for c in rec.calls))

    def test_syntax_error_is_recorded(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "broken.py"
            p.write_text("def oops(\n", encoding="utf-8")
            rec = parse_python_file(p, Path(td), set())
            self.assertTrue(rec.syntax_error)

    def test_analyze_repository_does_not_crash(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            pkg = tmp_path / "src" / "forge"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "a.py").write_text("from . import b\n", encoding="utf-8")
            (pkg / "b.py").write_text("x = 1\n", encoding="utf-8")
            data = analyze_repository(tmp_path, {".git"})
            self.assertTrue(data["modules"])
            self.assertTrue(data["import_edges"])

    def test_subprocess_site_recorded(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "run.py"
            p.write_text("import subprocess\nsubprocess.run(['true'])\n", encoding="utf-8")
            rec = parse_python_file(p, Path(td), set())
            self.assertTrue(rec.external_exec)
            self.assertEqual(rec.external_exec[0].mechanism, "subprocess.run")

    def test_package_init_relative_import(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            pkg = tmp_path / "src" / "forge" / "utils"
            pkg.mkdir(parents=True)
            (pkg / "atomic_write.py").write_text("def atomic_write():\n    pass\n", encoding="utf-8")
            (pkg / "__init__.py").write_text("from .atomic_write import atomic_write\n", encoding="utf-8")
            rec = parse_python_file(
                pkg / "__init__.py",
                tmp_path,
                {"forge.utils", "forge.utils.atomic_write"},
            )
            self.assertTrue(
                any(i.resolved == "forge.utils.atomic_write" for i in rec.imports)
            )


if __name__ == "__main__":
    unittest.main()
