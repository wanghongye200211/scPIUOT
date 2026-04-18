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
    "A": "Action curve",
    "B": "Potential-related curve",
    "C": "Additive candidates",
    "D": "Selected combined overlay",
}

DEFAULT_COPY_NAMES = {
    "A": "figure4_panel_A_action.png",
    "B": "figure4_panel_B_potential.png",
    "C": "figure4_panel_C_additive_candidates.png",
    "D": "figure4_panel_D_overlay.png",
}


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a public Figure 4-style additive criticality board.")
    parser.add_argument("--action-panel", type=Path, required=True)
    parser.add_argument("--potential-panel", type=Path, required=True)
    parser.add_argument("--candidates-panel", type=Path, required=True)
    parser.add_argument("--overlay-panel", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--selected-label", default=None)
    parser.add_argument("--title", default="Figure 4. Additive criticality")
    parser.add_argument(
        "--subtitle",
        default="Public additive-criticality board assembled from component and overlay panels.",
    )
    parser.add_argument("--output-dir", type=Path, default=DOWNSTREAM_OUTPUT_ROOT / "figure4_additive_criticality")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "A": args.action_panel.expanduser().resolve(),
        "B": args.potential_panel.expanduser().resolve(),
        "C": args.candidates_panel.expanduser().resolve(),
        "D": args.overlay_panel.expanduser().resolve(),
    }
    local_sources = {}
    for label, source_path in sources.items():
        local_path = out_dir / DEFAULT_COPY_NAMES[label]
        shutil.copy2(source_path, local_path)
        local_sources[label] = str(local_path)

    summary_rows: list[dict[str, str]] = []
    selected_row: dict[str, str] | None = None
    if args.summary_csv is not None and args.summary_csv.exists():
        summary_rows = read_summary_rows(args.summary_csv.expanduser().resolve())
        if args.selected_label:
            selected_row = next((row for row in summary_rows if row.get("label") == args.selected_label), None)
        if selected_row is None and summary_rows:
            selected_row = summary_rows[0]
        shutil.copy2(args.summary_csv.expanduser().resolve(), out_dir / args.summary_csv.name)
    if args.source_manifest is not None and args.source_manifest.exists():
        shutil.copy2(args.source_manifest.expanduser().resolve(), out_dir / args.source_manifest.name)

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

    footer_y = layout["canvas_h"] - layout["footer_h"] + 20
    if selected_row is not None:
        draw.line((margin, footer_y - 16, layout["canvas_w"] - margin, footer_y - 16), fill=LINE, width=2)
        label = selected_row.get("label", args.selected_label or "selected")
        draw.text((margin, footer_y), f"Selected weighting: {label}", font=footer_font, fill=TEXT)
        footer_lines = []
        if "peak_time" in selected_row and "peak_value" in selected_row:
            footer_lines.append(
                f"Peak time={float(selected_row['peak_time']):.3g}, peak value={float(selected_row['peak_value']):.3f}"
            )
        if "peak_action_component" in selected_row and "peak_potential_component" in selected_row:
            footer_lines.append(
                "Peak components: "
                f"action={float(selected_row['peak_action_component']):.3f}, "
                f"potential={float(selected_row['peak_potential_component']):.3f}"
            )
        if "action_to_potential_ratio_at_peak" in selected_row:
            footer_lines.append(
                f"Action/potential ratio at peak={float(selected_row['action_to_potential_ratio_at_peak']):.3f}"
            )
        for idx, line in enumerate(footer_lines, start=1):
            draw.text((margin, footer_y + 34 * idx), line, font=footer_font, fill=MUTED)

    board_png = out_dir / "figure4_additive_criticality_board.png"
    canvas.save(board_png, quality=95)

    manifest = {
        "sources": {k: str(v) for k, v in sources.items()},
        "local_sources": local_sources,
        "summary_csv": str(args.summary_csv.expanduser().resolve()) if args.summary_csv else None,
        "source_manifest": str(args.source_manifest.expanduser().resolve()) if args.source_manifest else None,
        "selected_label": args.selected_label,
        "selected_summary": selected_row,
        "board_png": str(board_png),
        "title": args.title,
        "subtitle": args.subtitle,
    }
    (out_dir / "figure4_additive_criticality_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
