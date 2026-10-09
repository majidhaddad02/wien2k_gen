"""File tree and inventory markdown."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def write_file_tree(out_dir: Path, inventory: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        path = row["path"]
        parent = str(Path(path).parent) if "/" in path else "."
        by_dir[parent].append(row)

    lines = [
        "# Repository File Tree",
        "",
        f"Analyzed revision: `{meta['git'].get('commit', 'unknown')}` ({meta['git'].get('dirty')}).",
        f"Timestamp (UTC): {meta['timestamp_utc']}.",
        "",
        "Generated from a walk of the checkout. `.git/` and cache directories are excluded.",
        "`offline_packages/packaging_offline/` wheel files are summarized, not listed individually.",
        "",
        "## Category counts",
        "",
    ]
    counts: dict[str, int] = defaultdict(int)
    for row in inventory:
        counts[row["category"]] += 1
    lines.append("| Category | Files |")
    lines.append("|----------|------:|")
    for cat, n in sorted(counts.items()):
        lines.append(f"| {cat} | {n} |")
    lines.extend(["", "## Tree", "", "```", "wien2k_gen/"])

    def render(prefix: str, indent: str) -> None:
        children_dirs = sorted(
            d for d in by_dir if d != prefix and (
                d.startswith(prefix + "/") if prefix != "." else "/" not in d
            ) and d.count("/") == (0 if prefix == "." else prefix.count("/") + 1)
        )
        files = sorted(by_dir.get(prefix, []), key=lambda r: r["path"])
        for row in files:
            name = Path(row["path"]).name
            if name == "(summarized":
                continue
            extra = f"  [{row['category']}]"
            lines.append(f"{indent}{name}{extra}")
        for d in children_dirs:
            lines.append(f"{indent}{Path(d).name}/")
            render(d, indent + "  ")

    # Group by top-level
    top_files = [r for r in inventory if "/" not in r["path"]]
    top_dirs = sorted({r["path"].split("/", 1)[0] for r in inventory if "/" in r["path"]})
    for row in sorted(top_files, key=lambda r: r["path"]):
        lines.append(f"  {row['path']}  [{row['category']}]")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        if "/" in row["path"]:
            grouped[row["path"].split("/", 1)[0]].append(row)
    for d in top_dirs:
        lines.append(f"  {d}/")
        for row in sorted(grouped[d], key=lambda r: r["path"]):
            rel = row["path"][len(d) + 1:]
            lines.append(f"    {rel}  [{row['category']}]")
    lines.extend(["```", "", "Machine-readable copies: `file_inventory.json`, `file_inventory.csv`.", ""])
    (out_dir / "file_tree.md").write_text("\n".join(lines), encoding="utf-8")
