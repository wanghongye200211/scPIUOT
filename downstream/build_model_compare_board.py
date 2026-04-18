from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DOWNSTREAM_OUTPUT_ROOT
from _figure_utils import BG, LINE, MUTED, TEXT, default_board_layout, draw_panel, load_font


DEFAULT_TITLES = {
    "A": "Per-time W1",
    "B": "Per-time W2 squared",
    "C": "Per-time MMD",
    "D": "Shared manifold overlay",
}

DEFAULT_COPY_NAMES = {
    "A": "figure3_panel_A_w1.png",
    "B": "figure3_panel_B_w2sq.png",
    "C": "figure3_panel_C_mmd.png",
    "D": "figure3_panel_D_shared_overlay.png",
}


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def metric_line(row: dict[str, str]) -> str:
    model = row.get("model", "model")
    w1 = row.get("w1", row.get("mean_w1", "nan"))
    w2_sq = row.get("w2_sq", row.get("w2sq", row.get("mean_w2_sq", "nan")))
    mmd = row.get("mmd_rbf", row.get("mmd", row.get("mean_mmd", "nan")))
    return f"{model}: W1={float(w1):.3f}, W2^2={float(w2_sq):.3f}, MMD={float(mmd):.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a public Figure 3-style model comparison board.")
    parser.add_argument("--panel-a", type=Path, required=True)
    parser.add_argument("--panel-b", type=Path, required=True)
    parser.add_argument("--panel-c", type=Path, required=True)
    parser.add_argument("--panel-d", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--title", default="Figure 3. Multi-model comparison")
    parser.add_argument("--subtitle", default="Public benchmark board assembled from four comparison panels.")
    parser.add_argument("--output-dir", type=Path, default=DOWNSTREAM_OUTPUT_ROOT / "figure3_compare")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "A": args.panel_a.expanduser().resolve(),
        "B": args.panel_b.expanduser().resolve(),
        "C": args.panel_c.expanduser().resolve(),
        "D": args.panel_d.expanduser().resolve(),
    }
    local_sources = {}
    for label, source_path in sources.items():
        local_path = out_dir / DEFAULT_COPY_NAMES[label]
        shutil.copy2(source_path, local_path)
        local_sources[label] = str(local_path)

    layout = default_board_layout()
    title_font = load_font(24, bold=True)
    subtitle_font = load_font(17)
    panel_label_font = load_font(28, bold=True)
    panel_title_font = load_font(22, bold=True)
    footer_font = load_font(18)

    canvas = Image.new("RGB", (layout["canvas_w"], layout["canvas_h"]), BG)
    draw = ImageDraw.Draw(canvas)
    margin = layout["margin"]
    header_h = layout["header_h"]
    panel_w = layout["panel_w"]
    panel_h = layout["panel_h"]
    gap_x = layout["gap_x"]
    gap_y = layout["gap_y"]

    draw.text((margin, 36), args.title, font=title_font, fill=TEXT)
    draw.text((margin, 82), args.subtitle, font=subtitle_font, fill=MUTED)
    draw.line((margin, 130, layout["canvas_w"] - margin, 130), fill=LINE, width=2)

    positions = {
        "A": (margin, header_h),
        "B": (margin + panel_w + gap_x, header_h),
        "C": (margin, header_h + panel_h + gap_y),
        "D": (margin + panel_w + gap_x, header_h + panel_h + gap_y),
    }

    for label, source in sources.items():
        image = Image.open(source).convert("RGB")
        x, y = positions[label]
        draw_panel(
            canvas,
            image,
            x=x,
            y=y,
            panel_w=panel_w,
            panel_h=panel_h,
            label=label,
            title=DEFAULT_TITLES[label],
            label_font=panel_label_font,
            title_font=panel_title_font,
        )

    footer_lines: list[str] = []
    summary_rows: list[dict[str, str]] = []
    if args.summary_csv is not None and args.summary_csv.exists():
        summary_rows = read_summary_rows(args.summary_csv.expanduser().resolve())
        footer_lines = [metric_line(row) for row in summary_rows[:5]]

    footer_y = layout["canvas_h"] - layout["footer_h"] + 26
    if footer_lines:
        draw.line((margin, footer_y - 18, layout["canvas_w"] - margin, footer_y - 18), fill=LINE, width=2)
        for idx, line in enumerate(footer_lines):
            draw.text((margin, footer_y + 34 * idx), line, font=footer_font, fill=TEXT if idx == 0 else MUTED)

    board_png = out_dir / "figure3_compare_board.png"
    canvas.save(board_png, quality=95)

    manifest = {
        "sources": {k: str(v) for k, v in sources.items()},
        "local_sources": local_sources,
        "summary_csv": str(args.summary_csv.expanduser().resolve()) if args.summary_csv else None,
        "summary_json": str(args.summary_json.expanduser().resolve()) if args.summary_json else None,
        "board_png": str(board_png),
        "summary_rows": summary_rows,
        "title": args.title,
        "subtitle": args.subtitle,
    }
    (out_dir / "figure3_compare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
