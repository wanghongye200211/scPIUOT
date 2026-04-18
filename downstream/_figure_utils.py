from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


BG = "#f7f5f0"
PANEL_BG = "#ffffff"
TEXT = "#222222"
MUTED = "#666666"
LINE = "#d8d4ca"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[str] = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/SFNS.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/SFNS.ttf",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_into_box(image: Image.Image, box_w: int, box_h: int) -> Image.Image:
    return ImageOps.contain(image, (box_w, box_h), Image.Resampling.LANCZOS)


def draw_panel(
    canvas: Image.Image,
    panel_image: Image.Image,
    *,
    x: int,
    y: int,
    panel_w: int,
    panel_h: int,
    label: str,
    title: str,
    label_font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        [x, y, x + panel_w, y + panel_h],
        radius=24,
        fill=PANEL_BG,
        outline=LINE,
        width=2,
    )
    draw.text((x + 20, y + 16), label, font=label_font, fill=TEXT)
    draw.text((x + 72, y + 20), title, font=title_font, fill=TEXT)

    content_x = x + 16
    content_y = y + 60
    content_w = panel_w - 32
    content_h = panel_h - 76
    fitted = fit_into_box(panel_image, content_w, content_h)
    paste_x = content_x + (content_w - fitted.width) // 2
    paste_y = content_y + (content_h - fitted.height) // 2
    canvas.paste(fitted, (paste_x, paste_y))


def default_board_layout() -> dict[str, int]:
    canvas_w = 3400
    canvas_h = 2860
    margin = 72
    gap_x = 44
    gap_y = 44
    header_h = 170
    footer_h = 180
    panel_w = (canvas_w - 2 * margin - gap_x) // 2
    panel_h = (canvas_h - header_h - footer_h - 2 * margin - gap_y) // 2
    return {
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "margin": margin,
        "gap_x": gap_x,
        "gap_y": gap_y,
        "header_h": header_h,
        "footer_h": footer_h,
        "panel_w": panel_w,
        "panel_h": panel_h,
    }
