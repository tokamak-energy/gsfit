"""How GSFit's settings, Python database reader and Rust IDSs flow into a reconstructed equilibrium.

Each input IDS (`magnetics`, `pf_active`, `tf` and `wall`, the minimum GSFit needs) has its own XML settings file, which says where its data is
read from. GSFit makes an empty IDS, and the database reader fills it with the sensors, coils, vacuum toroidal field or limiter.
Optional IDSs, which GSFit can use but does not need, are read in the same way.
The `equilibrium` IDS has its own XML settings file too: which magnetic sensors to use, the numerical grid and which time-slices to reconstruct.
The input IDSs, and any optional IDSs, are added to the `equilibrium` IDS, which is then solved.

The database reader is written in Python, so that a reader for any database can be written, and the IDSs are Rust. Each database reader tile has
a ".py" badge, in Python's blue and yellow, and each IDS tile a ".rs" badge, in Rust's red.

This is how GSFit will work once the rewrite onto IMAS IDSs is complete, not how it works today.

Each step is an icon on a tile, labelled underneath. There is a lane for each input IDS, then one lane which stands for any optional IDSs, and under
them a lane for the `equilibrium` IDS, which the input and optional IDSs are copied into. An IDS is drawn as a tree, whose leaves are white when it
is empty, and half filled with its colour when it holds data.
An IDS which is not filled has a white tile with a dashed edge, whose background is hatched when it is partially filled, as the `equilibrium` IDS is
before it is solved. The optional IDSs' steps are faded, and their arrows are dotted. A faded step is drawn as it would be otherwise, and then all of
it but its labels is blended towards white, so that its labels can still be read.
Everything is laid out by hand in millimetres, on an axes which covers the whole figure.
The text is set in the paper's caption font, Computer Modern, with the IDS and file names in its typewriter face, so the figure reads with the
paper's other figures. The figure is the 153 millimetre \\textwidth set by the IOP class.

Usage: python code_flow.py
Needs only numpy and matplotlib.
Writes `code_flow.svg` into this script's directory.
"""

import typing
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Patch
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MatplotlibPath
from matplotlib.text import Text
from matplotlib.textpath import TextToPath

# --- The page ---
# The IOP class's \textwidth, which the figure fills
TEXT_WIDTH = 153.0  # [millimetre]
# The whitespace around the drawing, inside the figure, so that the tile edges on the outside are not cut in half
CANVAS_MARGIN = 0.5  # [millimetre]
MILLIMETRE_PER_INCH = 25.4  # [millimetre / inch]
MILLIMETRE_PER_POINT = 25.4 / 72.0  # [millimetre / point]

# --- The fonts ---
# The font of the paper's captions, which the IOP class sets in the LaTeX default roman, Computer Modern.
# `cmr10` is that roman, `cmb10` its bold, `cmti10` its italic and `cmtt10` its typewriter, which sets the IDS and file names as code.
# Matplotlib ships all four, so the figure can be set in them without going through LaTeX itself
CAPTION_FONT = "cmr10"
CAPTION_FONT_BOLD = "cmb10"
CAPTION_FONT_ITALIC = "cmti10"
TYPEWRITER_FONT = "cmtt10"
BODY_FONT_SIZE = 5.8  # [point]
SMALL_FONT_SIZE = 5.3  # [point]
# The distance between the baselines of consecutive lines, as a multiple of the font size
LINE_SPACING = 1.25  # [dimensionless]
# A line of text is centred on the middle of its capital letters, which is this far above its baseline, as a multiple of the font size
CAP_CENTRE_HEIGHT = 0.34  # [dimensionless]
# Lays out text the way matplotlib does, to measure it
TEXT_TO_PATH = TextToPath()

# --- The tiles ---
# Each step is an icon on a rounded square tile
TILE_SIZE = 9.0  # [millimetre]
ICON_SIZE = 6.0  # [millimetre]
TILE_CORNER_RADIUS = 2.0  # [millimetre]
TILE_LINE_WIDTH = 0.6  # [point]
ICON_LINE_WIDTH = 0.5  # [point]
# An empty IDS is drawn with a dashed edge, given as (dash, gap) in multiples of the line width
EMPTY_DASH_PATTERN: tuple[float, float] = (4.0, 2.5)  # [dimensionless]
# A partially filled IDS has its tile's background hatched, and its tree's leaves half filled. Both are the IDS's colour at this opacity over white,
# drawn as the opaque colour which that makes, so that the hatch does not show through the leaves
PARTIAL_HATCH = "//"
PARTIAL_HATCH_LINE_WIDTH = 0.5  # [point]
PARTIAL_ALPHA = 0.5  # [dimensionless]
# A filled IDS has its tree's leaves filled with the IDS's colour at this opacity, over its tile's tinted background
FILLED_LEAF_ALPHA = 0.5  # [dimensionless]
# An optional step is faded: all of it but its labels is blended this far towards white
OPTIONAL_FADE_FRACTION = 0.6  # [dimensionless]
# The labels under a tile start this far below it
LABEL_GAP = 0.8  # [millimetre]
# A badge, such as ".py", which says what a step is written in, sits over the top right corner of its tile, like a notification badge,
# with its centre this far in from both edges. The top right corner is where it covers the least of the icons
BADGE_WIDTH = 5.0  # [millimetre]
BADGE_HEIGHT = 2.4  # [millimetre]
BADGE_INSET = 1.0  # [millimetre]

# --- The arrows ---
ARROW_LINE_WIDTH = 0.8  # [point]
# The arrow head's size, which matplotlib scales its head length and width by
ARROW_HEAD_SCALE = 5.0  # [point]
ARROW_HEAD_LENGTH = 0.8  # [dimensionless]
ARROW_HEAD_WIDTH = 0.45  # [dimensionless]
# The corners of an arrow which turns are rounded, so a merge of arrows reads as one flow
ARROW_CORNER_RADIUS = 1.8  # [millimetre]
# An optional step's arrow has its line drawn as round dots, about this far apart, and a solid head
ARROW_DOT_RADIUS = 0.2  # [millimetre]
ARROW_DOT_SPACING = 1.0  # [millimetre]
# The dots are placed along each straight line and rounded corner of an arrow by sampling it at this many points
ARROW_SEGMENT_N_SAMPLE = 20  # [count]
# Arrows stop this far short of a tile's edge, and so this far from the centre of a tile
TILE_ARROW_GAP = 1.2  # [millimetre]
ARROW_CLEARANCE = 0.5 * TILE_SIZE + TILE_ARROW_GAP  # [millimetre]
# A label sits this far above the arrow it labels
ARROW_LABEL_OFFSET = 1.6  # [millimetre]
# The label down the arrow which the input IDSs are copied along has its baseline this far to the right of the arrow, which leaves a gap
# between the arrow and the descenders, which face it
MERGE_LABEL_GAP = 1.2  # [millimetre]
# The labels above the `equilibrium` lane's partially filled and filled IDSs' tiles, which say how full they are, have the middle of their last
# line's capitals this far above the tile, which clears the tile's badge
FILL_LABEL_ABOVE_TILE = 2.0  # [millimetre]
# The partially filled IDS's label starts this far to the right of the arrow which comes down into the middle of its tile, which clears its head
PARTIALLY_FILLED_LABEL_ARROW_GAP = 1.5  # [millimetre]

# --- The layout, in millimetres from the drawing's left edge ---
# The tile centres of each input lane: XML settings, empty IDS, database reader and filled IDS
INPUT_COLUMN_XS: list[float] = [11.0, 37.0, 63.0, 89.0]  # [millimetre]
HEADER_HEIGHT = 5.0  # [millimetre]
LANE_HEIGHT = 15.5  # [millimetre]
# A lane's tile sits at its top, with its labels underneath
TILE_CENTER_BELOW_LANE_TOP = 0.5 * TILE_SIZE + 0.5  # [millimetre]
# The `equilibrium` lane, under the input lanes. Its XML settings and empty IDS are in the input lanes' first two columns; these are the tile
# centres of the `equilibrium` IDS, which the input IDSs merge into from above, the solve and the solved IDS
EQUILIBRIUM_COLUMN_XS: list[float] = [112.0, 128.5, 145.0]  # [millimetre]
EQUILIBRIUM_LANE_GAP = 5.0  # [millimetre]
EQUILIBRIUM_LANE_HEIGHT = 18.5  # [millimetre]

# --- The IDSs ---
# The IDSs the database reader fills, which are the minimum GSFit needs, in the order they are drawn, top to bottom
INPUT_IDS_NAMES: list[str] = ["magnetics", "pf_active", "tf", "wall"]
# What each input IDS gives the `equilibrium` IDS
EQUILIBRIUM_INPUT_LABELS: dict[str, str] = {
    "magnetics": "sensors",
    "pf_active": "coil currents",
    "tf": "vacuum field",
    "wall": "limiter",
}
# The optional IDSs, which GSFit can use but does not need, share one lane under the input lanes. These label its XML settings and its IDSs,
# and what they give the `equilibrium` IDS
OPTIONAL_SETTINGS_LABEL = "***.xml"
OPTIONAL_IDS_LABEL = "optional IDSs"
OPTIONAL_EQUILIBRIUM_INPUT_LABEL = "optional"
# What the `equilibrium` IDS's XML settings hold
EQUILIBRIUM_SETTINGS_SUMMARY = "sensors to use, grid,\ntime-slices"


@dataclass(frozen=True)
class RoleStyle:
    """
    The colours of one kind of tile, and of the icon drawn on it.
    """

    edge_color: str
    face_color: str
    icon_edge_color: str
    icon_face_color: str


@dataclass(frozen=True)
class TextItem:
    """
    One or more lines of text, split on "\\n".
    """

    text: str
    font: str = CAPTION_FONT
    size: float = BODY_FONT_SIZE  # [point]


@dataclass(frozen=True)
class IconItem:
    """
    An icon, drawn by `draw(ax, center_x, center_y, size, style)`, `size` tall.
    """

    draw: Callable[[plt.Axes, float, float, float, RoleStyle], None]
    size: float  # [millimetre]


@dataclass(frozen=True)
class Badge:
    """
    A badge, such as ".py", which says what a step is written in.
    """

    text: str
    face_color: str
    text_color: str


# --- The colours ---
ARROW_COLOR = "#505050"
HEADER_COLOR = "#555555"
LABEL_COLOR = "#333333"
# Python's own blue and yellow
PYTHON_BLUE = "#3776ab"
PYTHON_YELLOW = "#ffd43b"
# Rust's red, which the Rust Foundation's brand guide calls "Rust Orange"
RUST_RED = "#d34516"
# The database reader is Python, and the IDSs are Rust
PYTHON_BADGE = Badge(text=".py", face_color=PYTHON_BLUE, text_color=PYTHON_YELLOW)
RUST_BADGE = Badge(text=".rs", face_color=RUST_RED, text_color="#ffffff")
# Each IDS is in its own colour from Paul Tol's colour-blind safe "bright" scheme, the XML settings in grey and the database reader in amber.
# The optional IDSs have no colour of their own, so they are in a darker grey, which still shows when faded
ROLE_STYLES: dict[str, RoleStyle] = {
    "equilibrium": RoleStyle(edge_color="#aa3377", face_color="#f5e1ec", icon_edge_color="#882255", icon_face_color="#d99cc2"),
    "magnetics": RoleStyle(edge_color="#4477aa", face_color="#e1eaf4", icon_edge_color="#2f5a86", icon_face_color="#8fb0d3"),
    "optional": RoleStyle(edge_color="#777777", face_color="#ececec", icon_edge_color="#555555", icon_face_color="#bbbbbb"),
    "pf_active": RoleStyle(edge_color="#c8465a", face_color="#fbe3e6", icon_edge_color="#a23447", icon_face_color="#ee8e9c"),
    "reader": RoleStyle(edge_color="#b58b00", face_color="#fff4d1", icon_edge_color="#8c6b00", icon_face_color="#f2c94c"),
    "settings": RoleStyle(edge_color="#7a7a7a", face_color="#f3f3f3", icon_edge_color="#5a5a5a", icon_face_color="#ffffff"),
    "solve": RoleStyle(edge_color="#555555", face_color="#e8e8e8", icon_edge_color="#3d3d3d", icon_face_color="#8c8c8c"),
    "tf": RoleStyle(edge_color="#228833", face_color="#ddf0e0", icon_edge_color="#1b6b29", icon_face_color="#7cc48a"),
    "wall": RoleStyle(edge_color="#3399bb", face_color="#dff3f9", icon_edge_color="#1f7699", icon_face_color="#9ad8ec"),
}


def make_white_tile_role_style(ids_name: str) -> RoleStyle:
    """
    An IDS which is not filled, empty or partially filled, keeps its IDS's colour on its edge and icon, but its tile is white.

    :param ids_name: The IDS's name, or "optional" for the optional IDSs
    :return: The style of the IDS's white tile
    """
    ids_style = ROLE_STYLES[ids_name]
    return RoleStyle(edge_color=ids_style.edge_color, face_color="#ffffff", icon_edge_color=ids_style.edge_color, icon_face_color="#ffffff")


def blend_with_white(color: str, alpha: float) -> tuple[float, float, float, float]:
    """
    :param color: A colour
    :param alpha: Its opacity [dimensionless]
    :return: The opaque colour which `color` makes at opacity `alpha` over white, as (red, green, blue, alpha) [dimensionless]
    """
    red, green, blue, _ = to_rgba(color)
    return (alpha * red + 1.0 - alpha, alpha * green + 1.0 - alpha, alpha * blue + 1.0 - alpha, 1.0)


def fade_color(color: str | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """
    :param color: A colour, which may be partly transparent
    :return: `color` blended `OPTIONAL_FADE_FRACTION` of the way towards white, with the same opacity, as (red, green, blue, alpha) [dimensionless]
    """
    red, green, blue, alpha = to_rgba(color)
    return (
        red + OPTIONAL_FADE_FRACTION * (1.0 - red),
        green + OPTIONAL_FADE_FRACTION * (1.0 - green),
        blue + OPTIONAL_FADE_FRACTION * (1.0 - blue),
        alpha,
    )


# ======================================================================================================================================
# Drawing primitives
# ======================================================================================================================================
def create_canvas(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    """
    Make a figure whose axes run in millimetres from (0, 0) at the bottom left of the drawing, with `CANVAS_MARGIN` around it.

    :param width: The drawing's width, without the margin [millimetre]
    :param height: The drawing's height, without the margin [millimetre]
    :return: The figure and its axes
    """
    figure_width = width + 2.0 * CANVAS_MARGIN  # [millimetre]
    figure_height = height + 2.0 * CANVAS_MARGIN  # [millimetre]
    fig = plt.figure(figsize=(figure_width / MILLIMETRE_PER_INCH, figure_height / MILLIMETRE_PER_INCH))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(-CANVAS_MARGIN, width + CANVAS_MARGIN)
    ax.set_ylim(-CANVAS_MARGIN, height + CANVAS_MARGIN)
    ax.set_axis_off()
    return fig, ax


@contextmanager
def fade_drawing(ax: plt.Axes) -> Iterator[None]:
    """
    Fade everything drawn on the axes inside this context, by blending its colours towards white, which marks an optional step.
    The fade is applied once the step is drawn, so that a step is drawn by the same functions whether it is faded or not.

    :param ax: The axes being drawn on
    """
    artists_before = set(ax.get_children())
    yield
    for artist in ax.get_children():
        if artist in artists_before:
            continue
        if isinstance(artist, Patch):
            artist.set_facecolor(fade_color(artist.get_facecolor()))
            artist.set_edgecolor(fade_color(artist.get_edgecolor()))
            if artist.get_hatch() is not None:
                artist.set_hatchcolor(fade_color(artist.get_hatchcolor()))
        elif isinstance(artist, Line2D | Text):
            artist.set_color(fade_color(artist.get_color()))
        else:
            raise TypeError(f"Cannot fade a {type(artist).__name__}")


def get_line_height(font_size: float) -> float:
    """
    :param font_size: The font size [point]
    :return: The distance between the baselines of consecutive lines [millimetre]
    """
    return LINE_SPACING * font_size * MILLIMETRE_PER_POINT


def get_text_width(text: str, font: str, size: float) -> float:
    """
    :param text: One line of text
    :param font: The font name
    :param size: The font size [point]
    :return: The text's width, measured as matplotlib lays it out [millimetre]
    """
    text_width, _, _ = TEXT_TO_PATH.get_text_width_height_descent(text, FontProperties(family=font, size=size), ismath=False)  # [point]
    return text_width * MILLIMETRE_PER_POINT


def draw_text_line(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    font: str,
    size: float,
    color: str,
    ha: str = "center",
) -> None:
    """
    Write one line of text, centred vertically on the middle of its capital letters, so that lines with and without descenders line up.

    :param ax: The axes to draw on
    :param x: Horizontal position, set by `ha` [millimetre]
    :param y: The middle of the capital letters [millimetre]
    :param text: The text, one line
    :param font: The font name
    :param size: The font size [point]
    :param color: The text colour
    :param ha: Horizontal alignment
    """
    baseline_y = y - CAP_CENTRE_HEIGHT * size * MILLIMETRE_PER_POINT  # [millimetre]
    ax.text(x, baseline_y, text, ha=ha, va="baseline", fontname=font, fontsize=size, color=color, clip_on=False)


def draw_text_lines(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    font: str,
    size: float,
    color: str,
    ha: str = "center",
) -> None:
    """
    Write text which may run over several lines, split on "\\n", centred vertically on `y`.

    :param ax: The axes to draw on
    :param x: Horizontal position of the lines, set by `ha` [millimetre]
    :param y: The vertical centre of the block of lines [millimetre]
    :param text: The text
    :param font: The font name
    :param size: The font size [point]
    :param color: The text colour
    :param ha: Horizontal alignment of every line
    """
    lines = text.split("\n")
    n_line = len(lines)
    line_height = get_line_height(size)  # [millimetre]
    for i_line in range(n_line):
        line_y = y + (0.5 * (n_line - 1) - i_line) * line_height  # [millimetre]
        draw_text_line(ax, x, line_y, lines[i_line], font, size, color, ha)


def draw_text_ink_centred(ax: plt.Axes, x: float, y: float, text: str, font: str, size: float, color: str) -> None:
    """
    Write one line of text centred on the middle of its ink, which centres text with no capital letters, such as ".py", whatever its descenders.

    :param ax: The axes to draw on
    :param x: The text's horizontal centre [millimetre]
    :param y: The vertical middle of the text's ink [millimetre]
    :param text: The text, one line
    :param font: The font name
    :param size: The font size [point]
    :param color: The text colour
    """
    _, ink_height, ink_descent = TEXT_TO_PATH.get_text_width_height_descent(text, FontProperties(family=font, size=size), ismath=False)  # [point]
    baseline_y = y - (0.5 * ink_height - ink_descent) * MILLIMETRE_PER_POINT  # [millimetre]
    ax.text(x, baseline_y, text, ha="center", va="baseline", fontname=font, fontsize=size, color=color, clip_on=False)


def draw_text_segments(ax: plt.Axes, center_x: float, center_y: float, segments: list[tuple[str, str]], font: str, size: float) -> None:
    """
    Write one line of text whose words are in different colours, centred on (`center_x`, `center_y`).
    Each segment is placed after the one before it by its measured width and a space, so that the line reads as one.

    :param ax: The axes to draw on
    :param center_x: The line's horizontal centre [millimetre]
    :param center_y: The middle of the line's capital letters [millimetre]
    :param segments: The segments, from left to right, each as (text, colour), with no space at either end
    :param font: The font name
    :param size: The font size [point]
    """
    space_width = get_text_width("x x", font, size) - get_text_width("xx", font, size)  # [millimetre]
    n_segment = len(segments)
    segment_widths: list[float] = []  # [millimetre]
    for i_segment in range(n_segment):
        segment_widths.append(get_text_width(segments[i_segment][0], font, size))
    line_width = sum(segment_widths) + (n_segment - 1) * space_width  # [millimetre]
    segment_x = center_x - 0.5 * line_width  # [millimetre]
    for i_segment in range(n_segment):
        segment_text, segment_color = segments[i_segment]
        draw_text_line(ax, segment_x, center_y, segment_text, font, size, segment_color, ha="left")
        segment_x += segment_widths[i_segment] + space_width


def draw_text_segments_down(
    ax: plt.Axes,
    baseline_x: float,
    center_y: float,
    segments: list[tuple[str, str]],
    size: float,
    color: str,
) -> None:
    """
    Write one line of text, turned clockwise to read downwards, whose words are in different fonts, centred vertically on `center_y`.
    Turned clockwise, the tops of the letters face right, so its baseline runs down `baseline_x` with any descenders to its left.
    Each segment is placed below the one before it by its measured width and a space, so that the line reads as one.

    :param ax: The axes to draw on
    :param baseline_x: The line's baseline [millimetre]
    :param center_y: The line's vertical centre [millimetre]
    :param segments: The segments, from first to last, each as (text, font), with no space at either end
    :param size: The font size [point]
    :param color: The text colour
    """
    space_width = get_text_width("x x", CAPTION_FONT, size) - get_text_width("xx", CAPTION_FONT, size)  # [millimetre]
    n_segment = len(segments)
    segment_widths: list[float] = []  # [millimetre]
    for i_segment in range(n_segment):
        segment_text, segment_font = segments[i_segment]
        segment_widths.append(get_text_width(segment_text, segment_font, size))
    line_length = sum(segment_widths) + (n_segment - 1) * space_width  # [millimetre]
    segment_top = center_y + 0.5 * line_length  # [millimetre]
    for i_segment in range(n_segment):
        segment_text, segment_font = segments[i_segment]
        # Anchored at the start of its baseline, which stays put as the text is turned
        ax.text(
            baseline_x,
            segment_top,
            segment_text,
            rotation=-90.0,
            rotation_mode="anchor",
            ha="left",
            va="baseline",
            fontname=segment_font,
            fontsize=size,
            color=color,
            clip_on=False,
        )
        segment_top -= segment_widths[i_segment] + space_width


def draw_tile_labels(ax: plt.Axes, tile_center_x: float, tile_center_y: float, labels: list[TextItem]) -> None:
    """
    Write labels one under the other, under a tile and centred on it.

    :param ax: The axes to draw on
    :param tile_center_x: The tile's centre [millimetre]
    :param tile_center_y: The tile's centre [millimetre]
    :param labels: The labels, from the top down
    """
    label_top = tile_center_y - 0.5 * TILE_SIZE - LABEL_GAP  # [millimetre]
    for label in labels:
        n_line = label.text.count("\n") + 1
        label_height = n_line * get_line_height(label.size)  # [millimetre]
        draw_text_lines(ax, tile_center_x, label_top - 0.5 * label_height, label.text, label.font, label.size, LABEL_COLOR)
        label_top -= label_height


def draw_fill_label(ax: plt.Axes, x: float, tile_center_y: float, text: str, ha: str) -> None:
    """
    Write how full an IDS is, above its tile, set like the column headers, which say how full the input lanes' IDSs are.
    Its last line is just above the tile, clear of the tile's badge, so a label over more lines reaches higher.

    :param ax: The axes to draw on
    :param x: Horizontal position of the label, set by `ha` [millimetre]
    :param tile_center_y: The tile's centre [millimetre]
    :param text: The label, which may run over several lines, split on "\\n"
    :param ha: Horizontal alignment
    """
    n_line = text.count("\n") + 1
    last_line_y = tile_center_y + 0.5 * TILE_SIZE + FILL_LABEL_ABOVE_TILE  # [millimetre]
    center_y = last_line_y + 0.5 * (n_line - 1) * get_line_height(BODY_FONT_SIZE)  # [millimetre]
    draw_text_lines(ax, x, center_y, text, CAPTION_FONT_BOLD, BODY_FONT_SIZE, HEADER_COLOR, ha)


def make_rounded_polyline(points: list[tuple[float, float]], corner_radius: float) -> MatplotlibPath:
    """
    Make a path through the points, with each corner rounded by a quadratic Bézier curve.

    :param points: The corners of the path, from its start to its end [millimetre]
    :param corner_radius: How far before and after each corner the rounding starts and ends; less on short segments [millimetre]
    :return: The path
    """
    vertices: list[tuple[float, float]] = [points[0]]
    codes: list[np.uint8] = [MatplotlibPath.MOVETO]
    n_point = len(points)
    for i_point in range(1, n_point - 1):
        corner = np.array(points[i_point])
        before = np.array(points[i_point - 1])
        after = np.array(points[i_point + 1])
        length_before = float(np.linalg.norm(corner - before))  # [millimetre]
        length_after = float(np.linalg.norm(after - corner))  # [millimetre]
        radius = min(corner_radius, 0.5 * length_before, 0.5 * length_after)  # [millimetre]
        start = corner + radius * (before - corner) / length_before
        end = corner + radius * (after - corner) / length_after
        vertices.append((float(start[0]), float(start[1])))
        codes.append(MatplotlibPath.LINETO)
        vertices.append((float(corner[0]), float(corner[1])))
        codes.append(MatplotlibPath.CURVE3)
        vertices.append((float(end[0]), float(end[1])))
        codes.append(MatplotlibPath.CURVE3)
    vertices.append(points[-1])
    codes.append(MatplotlibPath.LINETO)
    return MatplotlibPath(vertices, codes)


def draw_arrow(ax: plt.Axes, points: list[tuple[float, float]], has_head: bool = True) -> None:
    """
    Draw an arrow through the points, with rounded corners and its head at the last point.

    :param ax: The axes to draw on
    :param points: The arrow's corners, from its tail to its head [millimetre]
    :param has_head: Whether to draw the head; a line which joins another arrow has none
    """
    arrow_style = f"-|>,head_length={ARROW_HEAD_LENGTH},head_width={ARROW_HEAD_WIDTH}" if has_head else "-"
    arrow = FancyArrowPatch(
        path=make_rounded_polyline(points, ARROW_CORNER_RADIUS),
        arrowstyle=arrow_style,
        mutation_scale=ARROW_HEAD_SCALE,
        color=ARROW_COLOR,
        linewidth=ARROW_LINE_WIDTH,
        joinstyle="round",
        capstyle="butt",
        clip_on=False,
        # Matplotlib snaps straight lines to whole pixels when it draws a raster image, but not curves, which would misalign the lines
        # which join an arrow by half a pixel; the SVG is not snapped either way
        snap=False,
    )
    ax.add_patch(arrow)


def draw_dots(ax: plt.Axes, path: MatplotlibPath) -> None:
    """
    Draw round dots along a path, as close to `ARROW_DOT_SPACING` apart as a whole number of gaps allows, so that there is a dot at each end.

    :param ax: The axes to draw on
    :param path: The path, of straight lines and curves [millimetre]
    """
    sample_points: list[np.ndarray] = []  # [millimetre]
    for bezier_segment, code in path.iter_bezier():
        if code == MatplotlibPath.MOVETO:
            # The path's first point
            sample_points.append(bezier_segment.control_points)
            continue
        # Each segment starts where the one before it ends, so its first point is left out
        sample_points.append(bezier_segment(np.linspace(0.0, 1.0, ARROW_SEGMENT_N_SAMPLE)[1:]))
    path_points = np.concatenate(sample_points)  # [millimetre]
    # The distance along the path to each of its sample points
    path_distances = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path_points, axis=0), axis=1))])  # [millimetre]
    n_dot = max(2, round(path_distances[-1] / ARROW_DOT_SPACING) + 1)
    dot_distances = np.linspace(0.0, path_distances[-1], n_dot)  # [millimetre]
    dot_xs = np.interp(dot_distances, path_distances, path_points[:, 0])  # [millimetre]
    dot_ys = np.interp(dot_distances, path_distances, path_points[:, 1])  # [millimetre]
    for i_dot in range(n_dot):
        ax.add_patch(Circle((dot_xs[i_dot], dot_ys[i_dot]), ARROW_DOT_RADIUS, facecolor=ARROW_COLOR, edgecolor="none", clip_on=False))


def draw_dotted_arrow(ax: plt.Axes, points: list[tuple[float, float]]) -> None:
    """
    Draw an arrow through the points as `draw_arrow` does, but with its line drawn as dots, which marks an optional step. Its head is solid.

    :param ax: The axes to draw on
    :param points: The arrow's corners, from its tail to its head [millimetre]
    """
    tip_x, tip_y = points[-1]
    before_tip_x, before_tip_y = points[-2]
    last_length = np.hypot(tip_x - before_tip_x, tip_y - before_tip_y)  # [millimetre]
    head_length = ARROW_HEAD_LENGTH * ARROW_HEAD_SCALE * MILLIMETRE_PER_POINT  # [millimetre]
    # The dots stop at the base of the head, which is drawn as a solid arrow just as long as its head
    head_base = (
        float(tip_x - head_length * (tip_x - before_tip_x) / last_length),
        float(tip_y - head_length * (tip_y - before_tip_y) / last_length),
    )
    draw_dots(ax, make_rounded_polyline(points[:-1] + [head_base], ARROW_CORNER_RADIUS))
    draw_arrow(ax, [head_base, points[-1]])


def draw_join(ax: plt.Axes, points: list[tuple[float, float]], direction: tuple[float, float], is_dotted: bool = False) -> None:
    """
    Draw a line through the points which, at its last point, turns onto another arrow running in `direction`, and stops just after its
    rounded corner has joined it. Each of the arrows which merge is drawn once, so that no part of them is drawn on top of another,
    which would draw it heavier.
    A dotted line stops as its rounded corner joins the other arrow, as dots on top of the other arrow would stand out from it.

    :param ax: The axes to draw on
    :param points: The line's corners, the last of which is on the other arrow [millimetre]
    :param direction: The unit vector along which the other arrow runs, away from the last point [dimensionless]
    :param is_dotted: Whether to draw the line as dots, which marks an optional step
    """
    join_x, join_y = points[-1]
    # Two corner radii on, so that the corner is rounded by a full radius
    join_end = (join_x + 2.0 * ARROW_CORNER_RADIUS * direction[0], join_y + 2.0 * ARROW_CORNER_RADIUS * direction[1])
    if is_dotted:
        join_path = make_rounded_polyline(points + [join_end], ARROW_CORNER_RADIUS)
        # Without its last straight line, which runs along the other arrow from the end of the rounded corner
        draw_dots(ax, MatplotlibPath(join_path.vertices[:-1], join_path.codes[:-1]))
    else:
        draw_arrow(ax, points + [join_end], has_head=False)


def draw_rounded_box(
    ax: plt.Axes,
    left: float,
    bottom: float,
    width: float,
    height: float,
    corner_radius: float,
    face_color: str,
    edge_color: str,
    is_dashed: bool = False,
    hatch_color: tuple[float, float, float, float] | None = None,
) -> None:
    """
    Draw a rounded rectangle.

    :param ax: The axes to draw on
    :param left: The left edge [millimetre]
    :param bottom: The bottom edge [millimetre]
    :param width: The width [millimetre]
    :param height: The height [millimetre]
    :param corner_radius: The corners' radius [millimetre]
    :param face_color: The fill colour
    :param edge_color: The edge colour
    :param is_dashed: Whether to dash the edge, which marks an IDS which is not filled: empty, or partially filled
    :param hatch_color: The colour of a hatch over the fill, which marks a partially filled IDS; None for no hatch
    """
    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=TILE_LINE_WIDTH,
        clip_on=False,
    )
    if is_dashed:
        patch.set_linestyle((0.0, EMPTY_DASH_PATTERN))
    if hatch_color is not None:
        # The hatch is drawn over the fill colour, so it needs no transparency
        patch.set_hatch(PARTIAL_HATCH)
        patch.set_hatchcolor(hatch_color)
        patch.set_hatch_linewidth(PARTIAL_HATCH_LINE_WIDTH)
    ax.add_patch(patch)


def draw_badge(ax: plt.Axes, tile_center_x: float, tile_center_y: float, badge: Badge) -> None:
    """
    Draw a badge over the top right corner of a tile, with a white edge so that it stands clear of the tile's edge.

    :param ax: The axes to draw on
    :param tile_center_x: The tile's centre [millimetre]
    :param tile_center_y: The tile's centre [millimetre]
    :param badge: The badge
    """
    center_x = tile_center_x + 0.5 * TILE_SIZE - BADGE_INSET  # [millimetre]
    center_y = tile_center_y + 0.5 * TILE_SIZE - BADGE_INSET  # [millimetre]
    draw_rounded_box(
        ax,
        center_x - 0.5 * BADGE_WIDTH,
        center_y - 0.5 * BADGE_HEIGHT,
        BADGE_WIDTH,
        BADGE_HEIGHT,
        0.5 * BADGE_HEIGHT,
        badge.face_color,
        "#ffffff",
    )
    draw_text_ink_centred(ax, center_x, center_y, badge.text, TYPEWRITER_FONT, SMALL_FONT_SIZE, badge.text_color)


def draw_icon_tile(
    ax: plt.Axes,
    center_x: float,
    center_y: float,
    icon: IconItem,
    style: RoleStyle,
    is_dashed: bool = False,
    is_hatched: bool = False,
    badge: Badge | None = None,
) -> None:
    """
    Draw an icon on a rounded square tile, with an optional badge over its top right corner.

    :param ax: The axes to draw on
    :param center_x: The tile's centre [millimetre]
    :param center_y: The tile's centre [millimetre]
    :param icon: The icon, drawn in the middle of the tile
    :param style: The tile's and icon's colours
    :param is_dashed: Whether to dash the tile's edge, which marks an IDS which is not filled: empty, or partially filled
    :param is_hatched: Whether to hatch the tile's background, in its edge colour at `PARTIAL_ALPHA`, which marks a partially filled IDS
    :param badge: The badge, which says what the step is written in
    """
    draw_rounded_box(
        ax,
        center_x - 0.5 * TILE_SIZE,
        center_y - 0.5 * TILE_SIZE,
        TILE_SIZE,
        TILE_SIZE,
        TILE_CORNER_RADIUS,
        style.face_color,
        style.edge_color,
        is_dashed,
        hatch_color=blend_with_white(style.edge_color, PARTIAL_ALPHA) if is_hatched else None,
    )
    icon.draw(ax, center_x, center_y, icon.size, style)
    if badge is not None:
        draw_badge(ax, center_x, center_y, badge)


# ======================================================================================================================================
# Icons, each drawn `size` tall and centred on (center_x, center_y)
# ======================================================================================================================================
def draw_xml_icon(ax: plt.Axes, center_x: float, center_y: float, size: float, style: RoleStyle) -> None:
    """
    A page with a folded corner, marked "</>".
    """
    page_width = 0.78 * size  # [millimetre]
    fold = 0.28 * size  # [millimetre]
    left = center_x - 0.5 * page_width  # [millimetre]
    right = center_x + 0.5 * page_width  # [millimetre]
    bottom = center_y - 0.5 * size  # [millimetre]
    top = center_y + 0.5 * size  # [millimetre]
    page_corners = [(left, bottom), (right, bottom), (right, top - fold), (right - fold, top), (left, top)]
    fold_corners = [(right - fold, top), (right - fold, top - fold), (right, top - fold)]
    for corners in [page_corners, fold_corners]:
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor=style.icon_face_color,
                edgecolor=style.icon_edge_color,
                linewidth=ICON_LINE_WIDTH,
                joinstyle="round",
                clip_on=False,
            )
        )
    # "</>" is three typewriter characters, each 0.525 em wide, which should fill about three quarters of the page
    font_size = 0.36 * size / MILLIMETRE_PER_POINT  # [point]
    draw_text_line(ax, center_x, center_y - 0.1 * size, "</>", TYPEWRITER_FONT, font_size, style.icon_edge_color)


def draw_database_icon(ax: plt.Axes, center_x: float, center_y: float, size: float, style: RoleStyle) -> None:
    """
    A cylinder with two bands, the usual symbol for a database.
    """
    width = 0.8 * size  # [millimetre]
    ellipse_height = 0.3 * size  # [millimetre]
    top_y = center_y + 0.5 * size - 0.5 * ellipse_height  # [millimetre]
    bottom_y = center_y - 0.5 * size + 0.5 * ellipse_height  # [millimetre]
    ax.add_patch(
        Ellipse(
            (center_x, bottom_y),
            width,
            ellipse_height,
            facecolor=style.icon_face_color,
            edgecolor=style.icon_edge_color,
            linewidth=ICON_LINE_WIDTH,
            clip_on=False,
        )
    )
    # The body covers the top half of the bottom ellipse, so only its front edge shows
    ax.add_patch(Rectangle((center_x - 0.5 * width, bottom_y), width, top_y - bottom_y, facecolor=style.icon_face_color, edgecolor="none", clip_on=False))
    for side_x in [center_x - 0.5 * width, center_x + 0.5 * width]:
        ax.plot([side_x, side_x], [bottom_y, top_y], color=style.icon_edge_color, linewidth=ICON_LINE_WIDTH, solid_capstyle="butt", clip_on=False)
    for band_fraction in [1.0 / 3.0, 2.0 / 3.0]:
        band_y = bottom_y + band_fraction * (top_y - bottom_y)  # [millimetre]
        ax.add_patch(
            Arc(
                (center_x, band_y),
                width,
                ellipse_height,
                theta1=180.0,
                theta2=360.0,
                edgecolor=style.icon_edge_color,
                linewidth=ICON_LINE_WIDTH,
                clip_on=False,
            )
        )
    ax.add_patch(
        Ellipse(
            (center_x, top_y),
            width,
            ellipse_height,
            facecolor=style.icon_face_color,
            edgecolor=style.icon_edge_color,
            linewidth=ICON_LINE_WIDTH,
            clip_on=False,
        )
    )


def draw_gear_icon(ax: plt.Axes, center_x: float, center_y: float, size: float, style: RoleStyle) -> None:
    """
    A gear with eight teeth, for the solver.
    """
    n_teeth = 8
    outer_radius = 0.5 * size  # [millimetre]
    root_radius = 0.37 * size  # [millimetre]
    hole_radius = 0.15 * size  # [millimetre]
    pitch = 2.0 * np.pi / n_teeth  # [radian]
    # Each tooth runs from the root, up to its flat top, and back down, at these fractions of the pitch either side of its centre
    tooth_profile: list[tuple[float, float]] = [(-0.27, root_radius), (-0.15, outer_radius), (0.15, outer_radius), (0.27, root_radius)]
    vertices: list[tuple[float, float]] = []
    for i_tooth in range(n_teeth):
        for pitch_fraction, radius in tooth_profile:
            angle = (i_tooth + pitch_fraction) * pitch  # [radian]
            vertices.append((center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)))
    ax.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=style.icon_face_color,
            edgecolor=style.icon_edge_color,
            linewidth=ICON_LINE_WIDTH,
            joinstyle="round",
            clip_on=False,
        )
    )
    ax.add_patch(
        Circle((center_x, center_y), hole_radius, facecolor=style.face_color, edgecolor=style.icon_edge_color, linewidth=ICON_LINE_WIDTH, clip_on=False)
    )


def draw_ids_tree_icon(
    ax: plt.Axes,
    center_x: float,
    center_y: float,
    size: float,
    style: RoleStyle,
    leaf_fill: typing.Literal["empty", "partial", "filled"],
) -> None:
    """
    A tree of nodes, as an IDS is a tree of data: a root, and three leaves, which show how full the IDS is.
    The leaves are white when it is empty, and half filled with its colour when it is partially filled or filled. The edges of the leaves, and the
    root, are always the full colour.
    It is drawn in its tile's edge colour, which is the IDS's own colour, so that an IDS's tree is the same colour however full it is.
    """
    ids_color = style.edge_color
    root_x = center_x - 0.32 * size  # [millimetre]
    root_y = center_y + 0.36 * size  # [millimetre]
    root_size = 0.24 * size  # [millimetre]
    leaf_ys = [center_y + 0.1 * size, center_y - 0.15 * size, center_y - 0.4 * size]  # [millimetre]
    leaf_left = root_x + 0.17 * size  # [millimetre]
    leaf_width = center_x + 0.45 * size - leaf_left  # [millimetre]
    leaf_height = 0.17 * size  # [millimetre]
    leaf_box_style = f"round,pad=0,rounding_size={0.5 * leaf_height}"
    match leaf_fill:
        case "empty":
            leaf_face_color = to_rgba(style.face_color)
        case "partial":
            # Opaque, so that the hatched background of a partially filled IDS's tile does not show through the leaves
            leaf_face_color = blend_with_white(ids_color, PARTIAL_ALPHA)
        case "filled":
            leaf_face_color = to_rgba(ids_color, FILLED_LEAF_ALPHA)
        case _:
            raise ValueError(f"Unknown leaf fill: {leaf_fill}")
    ax.plot([root_x, root_x], [root_y, leaf_ys[-1]], color=ids_color, linewidth=ICON_LINE_WIDTH, solid_capstyle="butt", clip_on=False)
    for leaf_y in leaf_ys:
        ax.plot([root_x, leaf_left], [leaf_y, leaf_y], color=ids_color, linewidth=ICON_LINE_WIDTH, solid_capstyle="butt", clip_on=False)
        ax.add_patch(
            FancyBboxPatch(
                (leaf_left, leaf_y - 0.5 * leaf_height),
                leaf_width,
                leaf_height,
                boxstyle=leaf_box_style,
                facecolor=leaf_face_color,
                edgecolor=ids_color,
                linewidth=ICON_LINE_WIDTH,
                clip_on=False,
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (root_x - 0.5 * root_size, root_y - 0.5 * root_size),
            root_size,
            root_size,
            boxstyle=f"round,pad=0,rounding_size={0.2 * root_size}",
            facecolor=ids_color,
            edgecolor=ids_color,
            linewidth=ICON_LINE_WIDTH,
            clip_on=False,
        )
    )


def draw_empty_ids_icon(ax: plt.Axes, center_x: float, center_y: float, size: float, style: RoleStyle) -> None:
    """
    An IDS tree with nothing in its leaves.
    """
    draw_ids_tree_icon(ax, center_x, center_y, size, style, leaf_fill="empty")


def draw_partially_filled_ids_icon(ax: plt.Axes, center_x: float, center_y: float, size: float, style: RoleStyle) -> None:
    """
    An IDS tree with its leaves half filled, as some of the IDS is filled in.
    """
    draw_ids_tree_icon(ax, center_x, center_y, size, style, leaf_fill="partial")


def draw_filled_ids_icon(ax: plt.Axes, center_x: float, center_y: float, size: float, style: RoleStyle) -> None:
    """
    An IDS tree with its leaves filled in, as the IDS is filled in.
    """
    draw_ids_tree_icon(ax, center_x, center_y, size, style, leaf_fill="filled")


def draw_equilibrium_icon(ax: plt.Axes, center_x: float, center_y: float, size: float, style: RoleStyle) -> None:
    """
    Nested, elongated, triangular flux surfaces, the inner ones shifted outwards as by the Shafranov shift, and the magnetic axis.
    """
    elongation = 1.6  # [dimensionless]
    minor_radius = 0.5 * size / elongation  # [millimetre]
    shafranov_shift = 0.08 * size  # [millimetre]
    angles = np.linspace(0.0, 2.0 * np.pi, 121)  # [radian]
    rho_values = [1.0, 0.72, 0.44]  # [dimensionless]
    # The surfaces are drawn from the outside in, each a little more strongly coloured than the one around it
    face_alphas = [0.35, 0.6, 0.9]  # [dimensionless]
    n_surface = len(rho_values)
    for i_surface in range(n_surface):
        rho = rho_values[i_surface]
        triangularity = 0.35 * rho  # [dimensionless]
        surface_x = (
            center_x - 0.5 * shafranov_shift + shafranov_shift * (1.0 - rho**2) + rho * minor_radius * np.cos(angles + triangularity * np.sin(angles))
        )  # [millimetre]
        surface_y = center_y + elongation * rho * minor_radius * np.sin(angles)  # [millimetre]
        ax.fill(surface_x, surface_y, facecolor=style.icon_face_color, alpha=face_alphas[i_surface], edgecolor="none", clip_on=False)
        ax.plot(surface_x, surface_y, color=style.icon_edge_color, linewidth=ICON_LINE_WIDTH, clip_on=False)
    ax.add_patch(Circle((center_x + 0.5 * shafranov_shift, center_y), 0.04 * size, facecolor=style.icon_edge_color, edgecolor="none", clip_on=False))


# ======================================================================================================================================
# The figure
# ======================================================================================================================================
def draw_headers(ax: plt.Axes, header_y: float) -> None:
    """
    Write the input lanes' column headers, with the "Python" of the database reader's in Python's blue.

    :param ax: The axes to draw on
    :param header_y: The middle of the headers [millimetre]
    """
    column_headers: list[list[tuple[str, str]]] = [
        [("XML settings", HEADER_COLOR)],
        [("Empty IDS", HEADER_COLOR)],
        [("Database reader", HEADER_COLOR)],
        [("Filled IDS", HEADER_COLOR)],
    ]
    for i_column in range(len(column_headers)):
        draw_text_segments(ax, INPUT_COLUMN_XS[i_column], header_y, column_headers[i_column], CAPTION_FONT_BOLD, BODY_FONT_SIZE)


def draw_input_lane(ax: plt.Axes, lane_y: float, ids_name: str, is_dotted: bool = False) -> None:
    """
    Draw one input lane's tiles, XML settings -> empty IDS -> database reader -> filled IDS, and the arrows between them, but not their labels.

    :param ax: The axes to draw on
    :param lane_y: The tiles' vertical centre [millimetre]
    :param ids_name: The lane's IDS's name, or "optional" for the optional IDSs, which sets the colour its IDSs are drawn in
    :param is_dotted: Whether to draw the arrows as dots, which marks an optional step
    """
    settings_x, empty_x, reader_x, filled_x = INPUT_COLUMN_XS  # [millimetre]
    draw_icon_tile(ax, settings_x, lane_y, IconItem(draw_xml_icon, ICON_SIZE), ROLE_STYLES["settings"])
    draw_icon_tile(ax, empty_x, lane_y, IconItem(draw_empty_ids_icon, ICON_SIZE), make_white_tile_role_style(ids_name), is_dashed=True, badge=RUST_BADGE)
    draw_icon_tile(ax, reader_x, lane_y, IconItem(draw_database_icon, ICON_SIZE), ROLE_STYLES["reader"], badge=PYTHON_BADGE)
    draw_icon_tile(ax, filled_x, lane_y, IconItem(draw_filled_ids_icon, ICON_SIZE), ROLE_STYLES[ids_name], badge=RUST_BADGE)
    n_column = len(INPUT_COLUMN_XS)
    for i_column in range(n_column - 1):
        arrow_points = [(INPUT_COLUMN_XS[i_column] + ARROW_CLEARANCE, lane_y), (INPUT_COLUMN_XS[i_column + 1] - ARROW_CLEARANCE, lane_y)]  # [millimetre]
        if is_dotted:
            draw_dotted_arrow(ax, arrow_points)
        else:
            draw_arrow(ax, arrow_points)


def draw_input_lane_labels(ax: plt.Axes, lane_y: float, settings_label: TextItem, ids_label: TextItem) -> None:
    """
    Label one input lane's XML settings, and its empty and filled IDSs. The database reader is labelled by its column's header alone.

    :param ax: The axes to draw on
    :param lane_y: The tiles' vertical centre [millimetre]
    :param settings_label: The XML settings' label
    :param ids_label: The empty and filled IDSs' label
    """
    settings_x, empty_x, _, filled_x = INPUT_COLUMN_XS  # [millimetre]
    draw_tile_labels(ax, settings_x, lane_y, [settings_label])
    draw_tile_labels(ax, empty_x, lane_y, [ids_label])
    draw_tile_labels(ax, filled_x, lane_y, [ids_label])


def draw_input_lanes(ax: plt.Axes, lane_ys: list[float]) -> None:
    """
    Draw the input IDSs' lanes: XML settings -> empty IDS -> database reader -> filled IDS, each step a labelled tile.

    :param ax: The axes to draw on
    :param lane_ys: The tiles' vertical centre in each input lane, from the top down [millimetre]
    """
    n_lane = len(lane_ys)
    for i_lane in range(n_lane):
        ids_name = INPUT_IDS_NAMES[i_lane]
        draw_input_lane(ax, lane_ys[i_lane], ids_name)
        draw_input_lane_labels(ax, lane_ys[i_lane], TextItem(f"{ids_name}.xml", TYPEWRITER_FONT), TextItem(ids_name, TYPEWRITER_FONT))


def draw_optional_lane(ax: plt.Axes, lane_y: float) -> None:
    """
    Draw the optional IDSs' lane, under the input lanes. It is drawn as an input lane is, but faded and with dotted arrows, as its steps are optional.
    Its labels are not faded, so that they can still be read.

    :param ax: The axes to draw on
    :param lane_y: The tiles' vertical centre [millimetre]
    """
    with fade_drawing(ax):
        draw_input_lane(ax, lane_y, "optional", is_dotted=True)
    draw_input_lane_labels(ax, lane_y, TextItem(OPTIONAL_SETTINGS_LABEL, TYPEWRITER_FONT), TextItem(OPTIONAL_IDS_LABEL))


def draw_equilibrium_lane(ax: plt.Axes, lane_ys: list[float], optional_lane_y: float, equilibrium_y: float) -> None:
    """
    Draw the `equilibrium` lane under the input lanes: XML settings -> empty IDS -> partially filled `equilibrium` IDS -> solve -> solved IDS.
    The input IDSs are copied into the `equilibrium` IDS along one arrow from above, which they each join with a label of what they give it.
    The optional IDSs join it in the same way, but with a dotted and faded line, as their lane is drawn.

    :param ax: The axes to draw on
    :param lane_ys: The tiles' vertical centre in each input lane, from the top down [millimetre]
    :param optional_lane_y: The tiles' vertical centre in the optional IDSs' lane [millimetre]
    :param equilibrium_y: The tiles' vertical centre in the `equilibrium` lane [millimetre]
    """
    settings_x, empty_x = INPUT_COLUMN_XS[0], INPUT_COLUMN_XS[1]  # [millimetre]
    equilibrium_x, solve_x, solved_x = EQUILIBRIUM_COLUMN_XS  # [millimetre]
    draw_icon_tile(ax, settings_x, equilibrium_y, IconItem(draw_xml_icon, ICON_SIZE), ROLE_STYLES["settings"])
    draw_icon_tile(
        ax,
        empty_x,
        equilibrium_y,
        IconItem(draw_empty_ids_icon, ICON_SIZE),
        make_white_tile_role_style("equilibrium"),
        is_dashed=True,
        badge=RUST_BADGE,
    )
    # Partially filled: the input IDSs are copied into it, but it is not solved yet. Like the empty IDS before it, it is not filled,
    # so it has a white tile with a dashed edge, but its background is hatched
    draw_icon_tile(
        ax,
        equilibrium_x,
        equilibrium_y,
        IconItem(draw_partially_filled_ids_icon, ICON_SIZE),
        make_white_tile_role_style("equilibrium"),
        is_dashed=True,
        is_hatched=True,
        badge=RUST_BADGE,
    )
    draw_icon_tile(ax, solve_x, equilibrium_y, IconItem(draw_gear_icon, ICON_SIZE), ROLE_STYLES["solve"])
    draw_icon_tile(ax, solved_x, equilibrium_y, IconItem(draw_equilibrium_icon, 7.0), ROLE_STYLES["equilibrium"], badge=RUST_BADGE)
    draw_tile_labels(
        ax, settings_x, equilibrium_y, [TextItem("equilibrium.xml", TYPEWRITER_FONT), TextItem(EQUILIBRIUM_SETTINGS_SUMMARY, size=SMALL_FONT_SIZE)]
    )
    draw_tile_labels(ax, empty_x, equilibrium_y, [TextItem("equilibrium", TYPEWRITER_FONT)])
    draw_tile_labels(ax, equilibrium_x, equilibrium_y, [TextItem("equilibrium", TYPEWRITER_FONT)])
    draw_tile_labels(ax, solve_x, equilibrium_y, [TextItem("Solve", CAPTION_FONT_BOLD)])
    draw_tile_labels(ax, solved_x, equilibrium_y, [TextItem("equilibrium", TYPEWRITER_FONT)])
    # The partially filled and solved IDSs say how full they are above their tiles, as the column headers do for the input lanes' IDSs.
    # The partially filled IDS's label is to the right of the arrow which comes down into the middle of its tile, and the solved IDS's is centred
    draw_fill_label(ax, equilibrium_x + PARTIALLY_FILLED_LABEL_ARROW_GAP, equilibrium_y, "Partially\nfilled IDS", ha="left")
    draw_fill_label(ax, solved_x, equilibrium_y, "Filled IDS", ha="center")
    lane_xs = [settings_x, empty_x, equilibrium_x, solve_x, solved_x]  # [millimetre]
    for i_column in range(len(lane_xs) - 1):
        draw_arrow(ax, [(lane_xs[i_column] + ARROW_CLEARANCE, equilibrium_y), (lane_xs[i_column + 1] - ARROW_CLEARANCE, equilibrium_y)])

    # --- The input IDSs, each turning down onto one arrow into the `equilibrium` IDS, and labelled with what it gives it ---
    filled_right = INPUT_COLUMN_XS[3] + ARROW_CLEARANCE  # [millimetre]
    # The labels are centred on the line, less a little for the rounded corner at its end
    label_x = 0.5 * (filled_right + equilibrium_x) - 0.6  # [millimetre]
    for i_lane in range(len(lane_ys)):
        lane_y = lane_ys[i_lane]  # [millimetre]
        draw_join(ax, [(filled_right, lane_y), (equilibrium_x, lane_y)], (0.0, -1.0))
        label = EQUILIBRIUM_INPUT_LABELS[INPUT_IDS_NAMES[i_lane]]
        draw_text_line(ax, label_x, lane_y + ARROW_LABEL_OFFSET, label, CAPTION_FONT_ITALIC, SMALL_FONT_SIZE, LABEL_COLOR)
    with fade_drawing(ax):
        draw_join(ax, [(filled_right, optional_lane_y), (equilibrium_x, optional_lane_y)], (0.0, -1.0), is_dotted=True)
    draw_text_line(ax, label_x, optional_lane_y + ARROW_LABEL_OFFSET, OPTIONAL_EQUILIBRIUM_INPUT_LABEL, CAPTION_FONT_ITALIC, SMALL_FONT_SIZE, LABEL_COLOR)
    # The arrow starts where the top lane's line has turned onto it, and is labelled down its right-hand side, centred on it
    merge_top_y = lane_ys[0] - ARROW_CORNER_RADIUS  # [millimetre]
    merge_bottom_y = equilibrium_y + ARROW_CLEARANCE  # [millimetre]
    draw_arrow(ax, [(equilibrium_x, merge_top_y), (equilibrium_x, merge_bottom_y)])
    draw_text_segments_down(
        ax,
        equilibrium_x + MERGE_LABEL_GAP,
        0.5 * (lane_ys[0] + merge_bottom_y),
        [("Copy into", CAPTION_FONT), ("equilibrium", TYPEWRITER_FONT), ("IDS", CAPTION_FONT)],
        BODY_FONT_SIZE,
        LABEL_COLOR,
    )


def plot_code_flow() -> plt.Figure:
    """
    The input lanes, the optional IDSs' lane under them, and the `equilibrium` lane underneath.

    :return: The figure
    """
    width = TEXT_WIDTH - 2.0 * CANVAS_MARGIN  # [millimetre]
    n_lane = len(INPUT_IDS_NAMES)
    # The input lanes, and the optional IDSs' lane, which is as tall
    height = HEADER_HEIGHT + (n_lane + 1) * LANE_HEIGHT + EQUILIBRIUM_LANE_GAP + EQUILIBRIUM_LANE_HEIGHT  # [millimetre]
    fig, ax = create_canvas(width, height)
    lane_ys: list[float] = []  # [millimetre]
    for i_lane in range(n_lane):
        lane_ys.append(height - HEADER_HEIGHT - TILE_CENTER_BELOW_LANE_TOP - i_lane * LANE_HEIGHT)
    optional_lane_y = lane_ys[-1] - LANE_HEIGHT  # [millimetre]
    equilibrium_y = EQUILIBRIUM_LANE_HEIGHT - TILE_CENTER_BELOW_LANE_TOP  # [millimetre]
    draw_headers(ax, height - 0.5 * HEADER_HEIGHT)
    draw_input_lanes(ax, lane_ys)
    draw_optional_lane(ax, optional_lane_y)
    draw_equilibrium_lane(ax, lane_ys, optional_lane_y, equilibrium_y)
    return fig


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [CAPTION_FONT]
# The text is saved as paths, so the SVG looks the same on a machine without the Computer Modern fonts
plt.rcParams["svg.fonttype"] = "path"
# The SVG's element IDs are otherwise random, which would change the file every time it is made
plt.rcParams["svg.hashsalt"] = "gsfit_code_flow"


if __name__ == "__main__":
    fig = plot_code_flow()
    figure_path = Path(__file__).resolve().parent / "code_flow.svg"
    # No date, so the SVG only changes when the figure does
    fig.savefig(figure_path, metadata={"Date": None})
    print(f"saved {figure_path}")
    # Save to PNG
    figure_path = Path(__file__).resolve().parent / "code_flow.png"
    fig.savefig(figure_path, dpi=400)
