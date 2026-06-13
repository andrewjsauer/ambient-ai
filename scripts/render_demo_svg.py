#!/usr/bin/env python3
"""Render docs/assets/demo.svg from the synthetic `ambient insights` output.

Runs scripts/demo.py (no API key, synthetic data) and draws a terminal-window
SVG of the result. The SVG renders inline on GitHub, stays crisp at any zoom,
and is regenerable any time the summary format changes:

    python scripts/render_demo_svg.py
"""

import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "demo.svg"

# Theme (GitHub-dark-ish)
BG = "#0d1117"
CHROME = "#161b22"
BORDER = "#30363d"
DIM = "#8b949e"      # labels / punctuation
FG = "#c9d1d9"       # normal text
ACCENT = "#58a6ff"   # title, command
GREEN = "#3fb950"    # good numbers, prompt $
YELLOW = "#d29922"   # attention numbers
RED = "#f85149"      # the stuck finding

CHAR_W = 8.4
LINE_H = 22
PAD_X = 24
TOP = 56            # below the title bar
FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _t(x, y, text, fill, weight="normal"):
    w = f' font-weight="{weight}"' if weight != "normal" else ""
    return f'<text x="{x:.1f}" y="{y}" fill="{fill}"{w} xml:space="preserve">{escape(text)}</text>'


def render(lines: list[str]) -> str:
    # Prompt line + blank + output lines.
    display = ["$ ambient insights", ""] + lines
    width = int(max(len(l) for l in display) * CHAR_W) + PAD_X * 2
    height = TOP + len(display) * LINE_H + 16

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}" font-size="14">',
        f'<rect width="{width}" height="{height}" rx="10" fill="{BG}" stroke="{BORDER}"/>',
        f'<rect width="{width}" height="36" rx="10" fill="{CHROME}"/>',
        f'<rect y="26" width="{width}" height="10" fill="{CHROME}"/>',
        f'<circle cx="20" cy="18" r="6" fill="#ff5f56"/>',
        f'<circle cx="40" cy="18" r="6" fill="#ffbd2e"/>',
        f'<circle cx="60" cy="18" r="6" fill="#27c93f"/>',
        _t(width / 2 - 52, 22, "ambient — insights", DIM),
    ]

    y = TOP
    for i, line in enumerate(display):
        x = PAD_X
        if i == 0:  # prompt
            out.append(_t(x, y, "$ ", GREEN, "bold"))
            out.append(_t(x + 2 * CHAR_W, y, "ambient insights", ACCENT, "bold"))
        elif line == "":
            pass
        elif line.startswith("Ambient Insights"):
            out.append(_t(x, y, line, ACCENT, "bold"))
        elif line.startswith("Top finding"):
            out.append(_t(x, y, line, RED, "bold"))
        elif ":" in line and not line.startswith("  "):
            label, _, rest = line.partition(":")
            out.append(_t(x, y, label + ":", DIM))
            vx = x + (len(label) + 1) * CHAR_W
            # Colour the value: green for the headline velocity/resolved line,
            # yellow for gaps/stuck, normal otherwise.
            color = FG
            low = label.lower()
            if "velocity" in low:
                color = GREEN
            elif "gap" in low or "stuck" in low or "thrash" in low:
                color = YELLOW
            out.append(_t(vx, y, rest, color, "bold" if color != FG else "normal"))
        else:
            out.append(_t(x, y, line, FG))
        y += LINE_H

    out.append("</svg>")
    return "\n".join(out)


def main():
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)  # guarantee no network / no cost
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo.py")],
        capture_output=True, text=True, env=env,
    )
    if res.returncode != 0:
        sys.exit(f"demo.py failed:\n{res.stderr}")
    lines = [l.rstrip() for l in res.stdout.splitlines()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(lines))
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
