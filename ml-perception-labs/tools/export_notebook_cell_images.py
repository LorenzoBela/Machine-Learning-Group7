#!/usr/bin/env python3
"""Export notebook code and output snapshots as images.

For each code cell in a notebook, this script creates:
1) one image containing the full code cell source
2) one image containing the cell output

Outputs are organized as:
<output_dir>/
  cell_001/
    code.png
    output.png
  cell_002/
    code.png
    output.png
  ...

The script is reusable for any .ipynb file.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc


DEFAULT_CANVAS_WIDTH = 1600
DEFAULT_MIN_CANVAS_WIDTH = 760
DEFAULT_MAX_CANVAS_WIDTH = 1200
DEFAULT_TEXT_FONT_SIZE = 22
BG_COLOR = "#000000"
TEXT_COLOR = "#f8fafc"
PADDING = 24
LINE_SPACING = 4
SECTION_GAP = 14


def normalize_multiline(value: Any, separator: str = "\n") -> str:
    """Convert notebook source/output values into a single string.

    Notebook JSON may store source/output as a list of strings where each item
    is one logical line without a trailing newline. We preserve readability by
    inserting a separator between items only when needed.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        merged_parts: list[str] = []
        for idx, part in enumerate(value):
            piece = str(part)
            merged_parts.append(piece)
            if idx < len(value) - 1 and not piece.endswith(("\n", "\r")):
                merged_parts.append(separator)
        return "".join(merged_parts)
    return str(value)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _load_monospace_font(size: int) -> ImageFont.ImageFont:
    """Load a readable monospace font with fallback to Pillow default."""
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/Consolas.ttf",
        "DejaVuSansMono.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width_px(font: ImageFont.ImageFont, text: str) -> int:
    bbox = font.getbbox(text)
    return max(0, bbox[2] - bbox[0])


def _resolve_text_canvas_width(
    text: str,
    font: ImageFont.ImageFont,
    min_width: int,
    max_width: int,
) -> int:
    normalized = _normalize_newlines(text)
    longest = 0
    for line in normalized.split("\n"):
        longest = max(longest, _text_width_px(font, line.expandtabs(4)))

    desired = longest + (PADDING * 2)
    return max(min_width, min(max_width, desired))


def _resolve_image_canvas_width(
    image: Image.Image,
    min_width: int,
    max_width: int,
) -> int:
    desired = image.width + (PADDING * 2)
    return max(min_width, min(max_width, desired))


def _wrap_lines(lines: list[str], max_chars: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        expanded = line.expandtabs(4)
        chunks = textwrap.wrap(
            expanded,
            width=max_chars,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if chunks:
            wrapped.extend(chunks)
        else:
            wrapped.append("")
    return wrapped


def build_text_image(
    text: str,
    width: int | None = None,
    min_width: int = DEFAULT_MIN_CANVAS_WIDTH,
    max_width: int = DEFAULT_MAX_CANVAS_WIDTH,
    font_size: int = DEFAULT_TEXT_FONT_SIZE,
) -> Image.Image:
    """Render text into an image using readable defaults and auto-fitted width."""
    text = _normalize_newlines(text)
    font = _load_monospace_font(font_size)

    if width is None:
        width = _resolve_text_canvas_width(text, font, min_width, max_width)

    char_w = max(7, _text_width_px(font, "M"))
    ag_bbox = font.getbbox("Ag")
    ag_height = max(10, ag_bbox[3] - ag_bbox[1])
    line_h = max(font_size + LINE_SPACING, ag_height + LINE_SPACING)

    max_chars = max(20, (width - (PADDING * 2)) // char_w)
    body_lines = _wrap_lines(text.split("\n"), max_chars)

    body_h = len(body_lines) * line_h
    total_h = PADDING + body_h + PADDING

    img = Image.new("RGB", (width, total_h), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    y = PADDING
    for line in body_lines:
        draw.text((PADDING, y), line, fill=TEXT_COLOR, font=font)
        y += line_h

    return img


def decode_first_image_output(outputs: list[dict[str, Any]]) -> Image.Image | None:
    """Return the first image output found in notebook cell outputs."""
    for output in outputs:
        data = output.get("data")
        if not isinstance(data, dict):
            continue

        for mime in ("image/png", "image/jpeg"):
            raw = data.get(mime)
            if not raw:
                continue

            # Base64 chunks should be concatenated directly without inserted separators.
            b64 = normalize_multiline(raw, separator="")
            try:
                img_bytes = base64.b64decode(b64)
                with Image.open(io.BytesIO(img_bytes)) as parsed:
                    return parsed.convert("RGB")
            except Exception:
                continue

    return None


def extract_text_output(outputs: list[dict[str, Any]]) -> str:
    """Collect textual output from stream/result/error notebook output types."""
    blocks: list[str] = []

    for output in outputs:
        output_type = output.get("output_type", "")

        if output_type == "stream":
            content = normalize_multiline(output.get("text")).rstrip()
            if content:
                blocks.append(content)

        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            if isinstance(data, dict):
                text_plain = normalize_multiline(data.get("text/plain")).rstrip()
                if text_plain:
                    blocks.append(text_plain)

        elif output_type == "error":
            ename = str(output.get("ename", "Error"))
            evalue = str(output.get("evalue", "")).strip()
            traceback_text = normalize_multiline(output.get("traceback")).rstrip()
            message = f"{ename}: {evalue}".strip()
            if traceback_text:
                message = f"{message}\n{traceback_text}" if message else traceback_text
            if message:
                blocks.append(message)

    return "\n\n".join(blocks).strip()


def _fit_image_to_width(
    image: Image.Image,
    target_width: int,
    zoom_small_outputs: bool,
) -> Image.Image:
    if image.width == target_width:
        return image

    if image.width > target_width:
        ratio = target_width / float(image.width)
        new_size = (target_width, max(1, int(image.height * ratio)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    # Enlarge small outputs so charts/text inside images remain readable.
    if zoom_small_outputs and image.width < int(target_width * 0.9):
        ratio = target_width / float(image.width)
        new_size = (target_width, max(1, int(image.height * ratio)))
        return image.resize(new_size, Image.Resampling.BICUBIC)

    return image


def _build_image_panel(
    image: Image.Image,
    width: int | None,
    min_width: int,
    max_width: int,
    zoom_small_outputs: bool,
) -> Image.Image:
    resolved_width = width if width is not None else _resolve_image_canvas_width(image, min_width, max_width)
    fitted = _fit_image_to_width(image, resolved_width - (PADDING * 2), zoom_small_outputs)

    panel_h = PADDING + fitted.height + PADDING
    panel = Image.new("RGB", (resolved_width, panel_h), BG_COLOR)

    x = (resolved_width - fitted.width) // 2
    y = PADDING
    panel.paste(fitted, (x, y))
    return panel


def build_output_image(
    outputs: list[dict[str, Any]],
    width: int | None = None,
    min_width: int = DEFAULT_MIN_CANVAS_WIDTH,
    max_width: int = DEFAULT_MAX_CANVAS_WIDTH,
    font_size: int = DEFAULT_TEXT_FONT_SIZE,
    zoom_small_outputs: bool = True,
) -> Image.Image:
    """Create one output image from all outputs in a cell."""
    text_output = extract_text_output(outputs)
    image_output = decode_first_image_output(outputs)

    if not outputs:
        return build_text_image(
            "[No output]",
            width=width,
            min_width=min_width,
            max_width=max_width,
            font_size=font_size,
        )

    if image_output is None and not text_output:
        return build_text_image(
            "[Output format not captured as text/image]",
            width=width,
            min_width=min_width,
            max_width=max_width,
            font_size=font_size,
        )

    resolved_width = width
    if resolved_width is None:
        candidates: list[int] = []
        if text_output:
            candidates.append(
                _resolve_text_canvas_width(
                    text_output,
                    _load_monospace_font(font_size),
                    min_width,
                    max_width,
                )
            )
        if image_output is not None:
            candidates.append(_resolve_image_canvas_width(image_output, min_width, max_width))
        resolved_width = max(candidates) if candidates else min_width

    if image_output is None:
        return build_text_image(
            text_output,
            width=resolved_width,
            min_width=min_width,
            max_width=max_width,
            font_size=font_size,
        )

    if not text_output:
        return _build_image_panel(
            image_output,
            width=resolved_width,
            min_width=min_width,
            max_width=max_width,
            zoom_small_outputs=zoom_small_outputs,
        )

    text_panel = build_text_image(
        text_output,
        width=resolved_width,
        min_width=min_width,
        max_width=max_width,
        font_size=font_size,
    )
    image_panel = _build_image_panel(
        image_output,
        width=resolved_width,
        min_width=min_width,
        max_width=max_width,
        zoom_small_outputs=zoom_small_outputs,
    )

    combined_h = text_panel.height + SECTION_GAP + image_panel.height
    combined = Image.new("RGB", (resolved_width, combined_h), BG_COLOR)
    combined.paste(text_panel, (0, 0))
    combined.paste(image_panel, (0, text_panel.height + SECTION_GAP))
    return combined


def export_notebook_cell_images(
    notebook_path: Path,
    output_root: Path,
    width: int | None,
    min_width: int,
    max_width: int,
    font_size: int,
    zoom_small_outputs: bool,
) -> Path:
    """Export code/output images for each code cell in a notebook."""
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError("Invalid notebook format: 'cells' must be a list")

    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    code_cell_counter = 0

    for notebook_cell_index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue

        code_cell_counter += 1
        cell_dir = output_root / f"cell_{code_cell_counter:03d}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        code_path = cell_dir / "code.png"
        output_path = cell_dir / "output.png"

        source_text = normalize_multiline(cell.get("source")).strip()
        if not source_text:
            source_text = "[Empty code cell]"

        build_text_image(
            source_text,
            width=width,
            min_width=min_width,
            max_width=max_width,
            font_size=font_size,
        ).save(code_path)

        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            outputs = []
        build_output_image(
            outputs,
            width=width,
            min_width=min_width,
            max_width=max_width,
            font_size=font_size,
            zoom_small_outputs=zoom_small_outputs,
        ).save(output_path)

        manifest.append(
            {
                "code_cell_number": code_cell_counter,
                "notebook_cell_number": notebook_cell_index,
                "code_image": str(code_path),
                "output_image": str(output_path),
            }
        )

    manifest_payload = {
        "notebook": str(notebook_path.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_cells_exported": code_cell_counter,
        "items": manifest,
    }

    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    return output_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export one code image and one output image per code cell from a Jupyter notebook."
        )
    )
    parser.add_argument("notebook", type=Path, help="Path to the .ipynb file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output folder. Default: <notebook_folder>/cell_images/<notebook_stem>"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Fixed image width in pixels. If omitted, width is auto-fitted to the longest line "
            "(bounded by --min-width/--max-width)."
        ),
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=DEFAULT_MIN_CANVAS_WIDTH,
        help=f"Minimum auto width in pixels (default: {DEFAULT_MIN_CANVAS_WIDTH})",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_MAX_CANVAS_WIDTH,
        help=f"Maximum auto width in pixels (default: {DEFAULT_MAX_CANVAS_WIDTH})",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=DEFAULT_TEXT_FONT_SIZE,
        help=f"Text font size in pixels (default: {DEFAULT_TEXT_FONT_SIZE})",
    )
    parser.add_argument(
        "--no-zoom-small-output",
        action="store_true",
        help="Disable automatic enlargement of small output images.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.min_width > args.max_width:
        parser.error("--min-width must be less than or equal to --max-width")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be a positive integer")
    if args.font_size <= 0:
        parser.error("--font-size must be a positive integer")

    notebook_path = args.notebook.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else notebook_path.parent / "cell_images" / notebook_path.stem
    )

    exported_path = export_notebook_cell_images(
        notebook_path=notebook_path,
        output_root=output_dir,
        width=args.width,
        min_width=args.min_width,
        max_width=args.max_width,
        font_size=args.font_size,
        zoom_small_outputs=not args.no_zoom_small_output,
    )
    print(f"Export complete: {exported_path}")


if __name__ == "__main__":
    main()
