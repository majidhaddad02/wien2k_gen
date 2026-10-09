"""Shared text helpers for architecture reports."""

from __future__ import annotations


def mermaid_id(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    ident = "".join(out)
    while "__" in ident:
        ident = ident.replace("__", "_")
    ident = ident.strip("_") or "n"
    if ident[0].isdigit():
        ident = "n_" + ident
    return ident


def q(label: str) -> str:
    cleaned = label.replace('"', "'").replace("\n", " ")
    return f'"{cleaned}"'
