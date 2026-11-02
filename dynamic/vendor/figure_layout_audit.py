#!/usr/bin/env python3
"""Conservative Matplotlib layout checks for publication figures."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

from matplotlib.artist import Artist
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox


IGNORE_GID = "qa-ignore"


def _label(artist: Artist) -> str:
    text = getattr(artist, "get_text", lambda: "")()
    text = " ".join(str(text).split())
    if text:
        return repr(text[:48])
    label = getattr(artist, "get_label", lambda: "")()
    return repr(str(label)) if label and not str(label).startswith("_") else artist.__class__.__name__


def _ignored(artist: Artist) -> bool:
    return artist.get_gid() == IGNORE_GID or not artist.get_visible()


def _extent(artist: Artist, renderer) -> Bbox | None:
    if _ignored(artist):
        return None
    try:
        # Annotation.get_window_extent also encloses its arrow; text collision
        # checks need the glyph box alone.
        bbox = (Text.get_window_extent(artist, renderer)
                if isinstance(artist, Annotation)
                else artist.get_window_extent(renderer))
    except (AttributeError, RuntimeError, ValueError):
        return None
    if bbox is None or bbox.width <= 0 or bbox.height <= 0:
        return None
    return bbox


def _intersection_area(first: Bbox, second: Bbox) -> float:
    width = min(first.x1, second.x1) - max(first.x0, second.x0)
    height = min(first.y1, second.y1) - max(first.y0, second.y0)
    return max(0.0, width) * max(0.0, height)


def _close(first: Bbox, second: Bbox, clearance: float) -> bool:
    """Whether two text boxes overlap or have too little visible separation."""
    horizontal_gap = max(first.x0 - second.x1, second.x0 - first.x1, 0.0)
    vertical_gap = max(first.y0 - second.y1, second.y0 - first.y1, 0.0)
    return horizontal_gap < clearance and vertical_gap < clearance


def _figure_text(fig) -> list[Artist]:
    """Include titles, ticks and figure annotations, even across axes."""
    artists: list[Artist] = list(fig.texts)
    for ax in fig.axes:
        artists.extend(ax.texts)
        artists.extend((ax.title, ax._left_title, ax._right_title))
        artists.extend((ax.xaxis.label, ax.yaxis.label))
        artists.extend(ax.get_xticklabels())
        artists.extend(ax.get_yticklabels())
    # Matplotlib may also list the suptitle in fig.texts.
    if getattr(fig, "_suptitle", None) is not None:
        artists.append(fig._suptitle)
    return list(dict.fromkeys(artist for artist in artists if artist.get_text().strip()))


def _inside(inner: Bbox, outer: Bbox, tolerance: float) -> bool:
    return (
        inner.x0 >= outer.x0 - tolerance
        and inner.y0 >= outer.y0 - tolerance
        and inner.x1 <= outer.x1 + tolerance
        and inner.y1 <= outer.y1 + tolerance
    )


def _frame_strips(bbox: Bbox, width: float) -> Iterable[Bbox]:
    yield Bbox.from_extents(bbox.x0 - width, bbox.y0 - width, bbox.x1 + width, bbox.y0 + width)
    yield Bbox.from_extents(bbox.x0 - width, bbox.y1 - width, bbox.x1 + width, bbox.y1 + width)
    yield Bbox.from_extents(bbox.x0 - width, bbox.y0 - width, bbox.x0 + width, bbox.y1 + width)
    yield Bbox.from_extents(bbox.x1 - width, bbox.y0 - width, bbox.x1 + width, bbox.y1 + width)


def _line_hits_bbox(line, bbox: Bbox) -> bool:
    if _ignored(line) or line.get_alpha() == 0:
        return False
    try:
        path = line.get_path().transformed(line.get_transform())
        return len(path.vertices) > 1 and path.intersects_bbox(bbox, filled=False)
    except (AttributeError, RuntimeError, ValueError):
        return False


def _arrow_hits_bbox(arrow: FancyArrowPatch, bbox: Bbox) -> bool:
    if _ignored(arrow) or arrow.get_alpha() == 0:
        return False
    try:
        path = arrow.get_path().transformed(arrow.get_transform())
        return path.intersects_bbox(bbox, filled=False)
    except (AttributeError, RuntimeError, ValueError):
        return False


def audit_figure(
    fig, name: str = "figure", tolerance_px: float = 1.5, clearance_px: float = 3.0
) -> list[str]:
    """Return likely layout-collision messages for a rendered Matplotlib figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues: list[str] = []

    figure_bbox = fig.bbox
    text_boxes = [(artist, _extent(artist, renderer)) for artist in _figure_text(fig)]
    text_boxes = [(artist, bbox) for artist, bbox in text_boxes if bbox is not None]
    for (first, first_box), (second, second_box) in combinations(text_boxes, 2):
        if _close(first_box, second_box, clearance_px):
            issues.append(
                f"{name}: figure text {_label(first)} overlaps or crowds {_label(second)}"
            )

    arrows = [
        patch for ax in fig.axes for patch in ax.patches
        if isinstance(patch, FancyArrowPatch)
    ]
    arrows.extend(
        text.arrow_patch for ax in fig.axes for text in ax.texts
        if getattr(text, "arrow_patch", None) is not None
    )
    for artist, bbox in text_boxes:
        if any(_arrow_hits_bbox(arrow, bbox) for arrow in arrows):
            issues.append(f"{name}: figure text {_label(artist)} crosses an arrow")

    for artist in fig.texts:
        bbox = _extent(artist, renderer)
        if bbox is not None and not _inside(bbox, figure_bbox, tolerance_px):
            issues.append(f"{name}: figure text {_label(artist)} is clipped by the canvas")

    for index, ax in enumerate(fig.axes, start=1):
        custom_text: list[Artist] = [text for text in ax.texts if text.get_text().strip()]
        if ax.title.get_text().strip():
            custom_text.append(ax.title)

        custom_boxes = [(artist, _extent(artist, renderer)) for artist in custom_text]
        custom_boxes = [(artist, bbox) for artist, bbox in custom_boxes if bbox is not None]

        for artist, bbox in custom_boxes:
            if not _inside(bbox, figure_bbox, tolerance_px):
                issues.append(f"{name}: axes {index} text {_label(artist)} is clipped by the canvas")
            if artist is ax.title and bbox.y0 < ax.bbox.y1 + clearance_px:
                issues.append(f"{name}: axes {index} title {_label(artist)} crowds the plot frame")
            if artist is not ax.title and any(
                _intersection_area(bbox, strip) > tolerance_px**2
                for strip in _frame_strips(ax.bbox, tolerance_px)
            ):
                issues.append(f"{name}: axes {index} text {_label(artist)} touches an axes frame")
            for line in ax.lines:
                if _line_hits_bbox(line, bbox):
                    issues.append(f"{name}: axes {index} text {_label(artist)} overlaps a plotted line")
                    break

        legend = ax.get_legend()
        legend_box = _extent(legend, renderer) if legend is not None else None
        if legend_box is not None:
            if not _inside(legend_box, figure_bbox, tolerance_px):
                issues.append(f"{name}: axes {index} legend is clipped by the canvas")
            for line in ax.lines:
                if _line_hits_bbox(line, legend_box):
                    issues.append(f"{name}: axes {index} legend overlaps plotted data")
                    break

        for label in (ax.xaxis.label, ax.yaxis.label):
            bbox = _extent(label, renderer)
            if bbox is None:
                continue
            if not _inside(bbox, figure_bbox, tolerance_px):
                issues.append(f"{name}: axes {index} label {_label(label)} is clipped by the canvas")
            for other in fig.axes:
                if other is not ax and _intersection_area(bbox, other.bbox) > tolerance_px**2:
                    issues.append(
                        f"{name}: axes {index} label {_label(label)} intrudes into a neighboring panel"
                    )
                    break

    return list(dict.fromkeys(issues))


def assert_figure_layout(
    fig, name: str = "figure", tolerance_px: float = 1.5, clearance_px: float = 3.0
) -> None:
    """Raise RuntimeError when the conservative layout audit finds issues."""
    issues = audit_figure(
        fig, name=name, tolerance_px=tolerance_px, clearance_px=clearance_px
    )
    if issues:
        raise RuntimeError("Publication-figure layout audit failed:\n- " + "\n- ".join(issues))


if __name__ == "__main__":
    raise SystemExit(
        "Import audit_figure or assert_figure_layout from a Matplotlib plotting script."
    )
