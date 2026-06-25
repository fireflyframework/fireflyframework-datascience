# Copyright 2026 Firefly Software Foundation.
"""Generate the Firefly DataScience diagram set as self-contained, WeasyPrint-safe SVGs.

House style (matches the agentic/pyfly/rust ecosystem): solid fills only, explicit ``<polygon>``
arrowheads (no ``<marker>``/``<filter>``), every viewBox set, no external fonts. Teal brand palette.

Run:  python assets/tools/gen_diagrams.py   ->   writes assets/diagrams/*.svg
"""

from __future__ import annotations

from pathlib import Path

# Teal brand palette (the DataScience member colour).
CYAN = "#06b6d4"
CYAN_D = "#0891b2"
CYAN_L = "#67e8f9"
INK = "#07131a"
CARD = "#f2fbfd"
CARD2 = "#e3f5fa"
CARD_S = "#bfe6ee"
ACCENT = "#fff7e6"
ACCENT_S = "#f6a821"
TITLE = "#0a2730"
SUB = "#5b7c84"
LINE = "#3aa6bd"
FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"


def _text(x: float, y: float, s: str, size: float = 13, fill: str = TITLE, weight: int = 600, anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{_esc(s)}</text>'
    )


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def card(x: float, y: float, w: float, h: float, title: str, sub: str = "", accent: bool = False) -> str:
    fill = ACCENT if accent else CARD
    stroke = ACCENT_S if accent else CARD_S
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
    if sub:
        out.append(_text(x + w / 2, y + h / 2 - 4, title, 13.5, TITLE, 700))
        out.append(_text(x + w / 2, y + h / 2 + 14, sub, 11, SUB, 500))
    else:
        out.append(_text(x + w / 2, y + h / 2 + 4, title, 13.5, TITLE, 700))
    return "".join(out)


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = LINE) -> str:
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    size = 7.0
    bx, by = x2 - size * math.cos(angle), y2 - size * math.sin(angle)
    left = (bx - size * 0.6 * math.cos(angle - math.pi / 2), by - size * 0.6 * math.sin(angle - math.pi / 2))
    right = (bx - size * 0.6 * math.cos(angle + math.pi / 2), by - size * 0.6 * math.sin(angle + math.pi / 2))
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{bx}" y2="{by}" stroke="{color}" stroke-width="2"/>'
        f'<polygon points="{x2},{y2} {left[0]},{left[1]} {right[0]},{right[1]}" fill="{color}"/>'
    )


def _svg(w: int, h: int, body: str, aria: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'role="img" aria-label="{_esc(aria)}">'
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>'
        f'<rect x="0" y="0" width="{w}" height="6" fill="{CYAN}"/>'
        f"{body}</svg>\n"
    )


def diagram_architecture() -> str:
    layers = [
        ("Core", "config · DI · banner · plugin discovery · types"),
        ("Agent  (reused: fireflyframework-agentic)", "FireflyAgent · memory · tools · embeddings · vectorstores"),
        ("DataScience", "datasets · features · models · evaluation · automl · serving"),
        ("Intelligence", "reasoning/search · validation · observability · security · cost-benefit gate"),
        ("Orchestration", "DAG pipelines · @workflow (HITL, budgets) · Airflow operators"),
    ]
    body = [_text(430, 34, "Firefly DataScience — Layered Architecture", 18, TITLE, 800)]
    y = 60
    for i, (t, s) in enumerate(layers):
        body.append(card(120, y, 620, 56, t, s, accent=(i == 2)))
        if i < len(layers) - 1:
            body.append(arrow(430, y + 56, 430, y + 70))
        y += 70
    return _svg(860, y + 12, "".join(body), "Firefly DataScience layered architecture")


def diagram_hexagonal() -> str:
    body = [_text(430, 34, "Hexagonal Ports & Adapters", 18, TITLE, 800)]
    body.append(card(330, 70, 200, 70, "AutoML Core", "library-agnostic", accent=True))
    ports = [
        ("DatasetLoaderPort", "sklearn · OpenML · HF"),
        ("TrainerPort", "RF · boosting · MLP"),
        ("SearchPolicyPort", "default · Optuna"),
        ("TrackerPort", "MLflow · noop"),
        ("ModelServerPort", "local · BentoML"),
        ("FeatureEngineerPort", "GenAI (CAAFE)"),
    ]
    coords = [(40, 70), (40, 170), (40, 270), (620, 70), (620, 170), (620, 270)]
    for (px, py), (t, s) in zip(coords, ports, strict=True):
        body.append(card(px, py, 200, 64, t, s))
        if px < 300:
            body.append(arrow(px + 200, py + 32, 330, 100 + (py - 70) * 0.1))
        else:
            body.append(arrow(px, py + 32, 530, 100 + (py - 70) * 0.1))
    return _svg(860, 360, "".join(body), "Hexagonal ports and adapters")


def diagram_automl_loop() -> str:
    steps = [
        ("Dataset", "load · validate"),
        ("Candidates", "trainers × params"),
        ("Cross-Validate", "classical engine"),
        ("Select", "best by metric"),
        ("Fit + Serve", "leaderboard · model"),
    ]
    body = [_text(470, 34, "Classical AutoML Pipeline", 18, TITLE, 800)]
    x = 30
    for i, (t, s) in enumerate(steps):
        body.append(card(x, 90, 160, 64, t, s, accent=(i == 2)))
        if i < len(steps) - 1:
            body.append(arrow(x + 160, 122, x + 182, 122))
        x += 182
    return _svg(940, 190, "".join(body), "Classical AutoML pipeline")


def diagram_genai_fusion() -> str:
    body = [_text(430, 34, "GenAI × Classical Fusion (governed)", 18, TITLE, 800)]
    body.append(card(60, 90, 200, 70, "LLM proposes", "feature / pipeline code", accent=True))
    body.append(card(330, 90, 200, 70, "Classical engine", "trains · cross-validates"))
    body.append(card(600, 90, 200, 70, "Cost/Benefit Gate", "keep only on lift"))
    body.append(arrow(260, 125, 330, 125))
    body.append(arrow(530, 125, 600, 125))
    body.append(arrow(700, 160, 700, 210))
    body.append(card(600, 210, 200, 56, "accept ✓ / reject ✗", "measured, not assumed"))
    body.append(_text(430, 300, "The LLM never decides — the measured score does.", 13, SUB, 600))
    return _svg(860, 330, "".join(body), "GenAI and classical fusion")


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "diagrams"
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "architecture.svg": diagram_architecture(),
        "hexagonal.svg": diagram_hexagonal(),
        "automl-loop.svg": diagram_automl_loop(),
        "genai-classical-fusion.svg": diagram_genai_fusion(),
    }
    for name, svg in figures.items():
        (out / name).write_text(svg)
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
