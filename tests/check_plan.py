"""Shared validators asserting a plan is physically and structurally sound."""
from __future__ import annotations

EPS = 1e-6


def iter_leaves(node: dict):
    if node["type"] in ("item", "waste"):
        yield node
    else:
        yield from iter_leaves(node["children"][0])
        yield from iter_leaves(node["children"][1])


def count_cut_nodes(node: dict) -> int:
    """Number of cut nodes in the subtree (one guillotine cut each)."""
    if node["type"] != "cut":
        return 0
    return 1 + sum(count_cut_nodes(child) for child in node["children"])


def check_cut_tree(node: dict, kerf: float) -> None:
    """Recursively verify the binary guillotine cut tree.

    Rules enforced:
    * a cut node has exactly two children and spans its whole rectangle;
    * the first child is flush with the node's origin;
    * the second child starts exactly ``kerf`` after the first child ends
      (the cut consumes the kerf);
    * children never exceed the node's bounds (edge trim is allowed, so the
      far edge of the second child may fall short of the node's far edge);
    * leaves have positive dimensions.
    """
    if node["type"] != "cut":
        assert node["width"] > 0 and node["height"] > 0
        return

    assert node["direction"] in ("vertical", "horizontal")
    assert len(node["children"]) == 2
    first, second = node["children"]

    if node["direction"] == "vertical":
        # Cut line is vertical and spans the full height of the node.
        assert abs(first["x"] - node["x"]) <= EPS
        assert abs(first["y"] - node["y"]) <= EPS
        assert abs(second["y"] - node["y"]) <= EPS
        assert abs(second["x"] - (first["x"] + first["width"] + kerf)) <= EPS
        assert first["height"] <= node["height"] + EPS
        assert second["height"] <= node["height"] + EPS
        assert second["x"] + second["width"] <= node["x"] + node["width"] + EPS
    else:
        # Cut line is horizontal and spans the full width of the node.
        assert abs(first["x"] - node["x"]) <= EPS
        assert abs(first["y"] - node["y"]) <= EPS
        assert abs(second["x"] - node["x"]) <= EPS
        assert abs(second["y"] - (first["y"] + first["height"] + kerf)) <= EPS
        assert first["width"] <= node["width"] + EPS
        assert second["width"] <= node["width"] + EPS
        assert second["y"] + second["height"] <= node["y"] + node["height"] + EPS

    check_cut_tree(first, kerf)
    check_cut_tree(second, kerf)


def check_sheet(sheet: dict, kerf: float) -> None:
    """Verify one sheet: tree structure, placements match tree leaves,
    pieces are inside the sheet and do not overlap, utilization matches.

    Placements report the finished (net) rectangle plus a
    ``production_rect`` (net size plus the four bleeds); the cut tree's
    item leaves span the production rectangle, so bounds and overlap are
    checked against it while utilization only counts the net area.
    """
    width, height = sheet["width"], sheet["height"]
    placements = sheet["placements"]

    check_cut_tree(sheet["cut_tree"], kerf)

    # The reported cut count must match the actual cut tree.
    assert sheet["cut_count"] == count_cut_nodes(sheet["cut_tree"])

    # Tree item leaves must be exactly the placements' production rects.
    tree_items = [
        (leaf["item"], leaf["x"], leaf["y"], leaf["width"], leaf["height"],
         leaf["rotated"])
        for leaf in iter_leaves(sheet["cut_tree"])
        if leaf["type"] == "item"
    ]
    placed = [
        (p["item"], p["production_rect"]["x"], p["production_rect"]["y"],
         p["production_rect"]["width"], p["production_rect"]["height"],
         p["rotated"])
        for p in placements
    ]
    assert sorted(map(repr, tree_items)) == sorted(map(repr, placed))

    # Every production rect contains its finished (net) rectangle.
    for p in placements:
        pr = p["production_rect"]
        assert pr["x"] <= p["x"] + EPS and pr["y"] <= p["y"] + EPS
        assert pr["x"] + pr["width"] >= p["x"] + p["width"] - EPS
        assert pr["y"] + pr["height"] >= p["y"] + p["height"] - EPS

    # Bounds and pairwise overlap of the production rectangles.
    rects = []
    for p in placements:
        pr = p["production_rect"]
        x2, y2 = pr["x"] + pr["width"], pr["y"] + pr["height"]
        assert pr["x"] >= -EPS and pr["y"] >= -EPS
        assert x2 <= width + EPS and y2 <= height + EPS
        rects.append((pr["x"], pr["y"], x2, y2))
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            overlap_x = min(a[2], b[2]) - max(a[0], b[0])
            overlap_y = min(a[3], b[3]) - max(a[1], b[1])
            assert overlap_x <= EPS or overlap_y <= EPS, (
                f"placements {i} and {j} overlap"
            )

    # Utilization matches the placed net (finished) area.
    used = sum(p["width"] * p["height"] for p in placements)
    assert abs(sheet["utilization"] - used / (width * height)) <= 1e-4


def check_plan(plan: dict) -> None:
    kerf = plan["kerf"]
    for sheet in plan["sheets"]:
        assert sheet["width"] == plan["stock"]["width"]
        assert sheet["height"] == plan["stock"]["height"]
        check_sheet(sheet, kerf)

    summary = plan["summary"]
    assert summary["sheet_count"] == len(plan["sheets"])
    assert summary["piece_count"] == sum(
        len(sheet["placements"]) for sheet in plan["sheets"]
    )


# ---------------------------------------------------------------------------
# duplex printing back-side validation
# ---------------------------------------------------------------------------

def _mirror_value(value: float, size: float, span: float) -> float:
    """Mirror the coordinate of a rectangle (``value``/``size``) along an
    axis of length ``size``."""
    return size - value - span


def _mirrored_rect(rect: dict, flip: str,
                   sheet_width: float, sheet_height: float) -> tuple:
    if flip == "left_right":
        return (_mirror_value(rect["x"], sheet_width, rect["width"]),
                rect["y"], rect["width"], rect["height"])
    return (rect["x"],
            _mirror_value(rect["y"], sheet_height, rect["height"]),
            rect["width"], rect["height"])


def _check_mirrored_tree(front: dict, back: dict, flip: str,
                         sheet_width: float, sheet_height: float) -> None:
    """Front and back trees are the same shape with mirrored rectangles;
    children are swapped exactly at cuts running along the mirror axis."""
    assert back["type"] == front["type"]
    assert back["width"] == front["width"]
    assert back["height"] == front["height"]
    bx, by, _, _ = _mirrored_rect(front, flip, sheet_width, sheet_height)
    assert abs(back["x"] - bx) <= EPS
    assert abs(back["y"] - by) <= EPS

    if front["type"] == "cut":
        assert back["direction"] == front["direction"]
        axis_cut = (
            (flip == "left_right" and front["direction"] == "vertical")
            or (flip == "top_bottom" and front["direction"] == "horizontal")
        )
        order = [1, 0] if axis_cut else [0, 1]
        for back_child, front_index in zip(back["children"], order):
            _check_mirrored_tree(front["children"][front_index], back_child,
                                 flip, sheet_width, sheet_height)
    elif front["type"] == "item":
        assert back["item"] == front["item"]
        assert back["rotated"] == front["rotated"]


def _check_back_cut_tree(node: dict, kerf: float) -> None:
    """Structural check for a duplex back tree.

    It is the exact geometric mirror of a validated front cut tree, so the
    physical invariants that survive mirroring are enforced:

    * the cut separates two children ordered along its axis (left before
      right for vertical cuts, top before bottom for horizontal cuts);
    * the children are exactly ``kerf`` apart along the cut axis;
    * every child stays inside the node's rectangle;
    * leaves have positive dimensions.

    The front-side encoding additionally pins both children to the cut's
    origin edge, but an implicit trim sliver (<= kerf) left on a far edge
    by the front engine moves to the opposite edge after mirroring (and a
    perpendicular-axis sliver moves the child's flush side).  Exact flush
    is therefore not asserted here; the recursive mirror correspondence
    checked separately (``_check_mirrored_tree``) proves this tree is the
    mirror of a front tree that passed the strict ``check_cut_tree``.
    """
    if node["type"] != "cut":
        assert node["width"] > 0 and node["height"] > 0
        return

    assert node["direction"] in ("vertical", "horizontal")
    assert len(node["children"]) == 2
    first, second = node["children"]
    nx, ny, nw, nh = node["x"], node["y"], node["width"], node["height"]

    for child in (first, second):
        # Every child stays within the node on both axes.
        assert child["x"] >= nx - EPS and child["y"] >= ny - EPS
        assert child["x"] + child["width"] <= nx + nw + EPS
        assert child["y"] + child["height"] <= ny + nh + EPS

    if node["direction"] == "vertical":
        # Ascending left-to-right order, separated by exactly the kerf.
        assert first["x"] <= second["x"] + EPS
        assert abs(second["x"] - (first["x"] + first["width"] + kerf)) <= EPS
        assert first["height"] <= nh + EPS and second["height"] <= nh + EPS
    else:
        # Ascending top-to-bottom order, separated by exactly the kerf.
        assert first["y"] <= second["y"] + EPS
        assert abs(second["y"] - (first["y"] + first["height"] + kerf)) <= EPS
        assert first["width"] <= nw + EPS and second["width"] <= nw + EPS

    _check_back_cut_tree(first, kerf)
    _check_back_cut_tree(second, kerf)


def _check_back_sheet_structure(sheet: dict, kerf: float) -> None:
    """Bounds/non-overlap/count checks shared with the front side, using
    the symmetric (mirror-aware) cut tree validator."""
    width, height = sheet["width"], sheet["height"]
    placements = sheet["placements"]

    _check_back_cut_tree(sheet["cut_tree"], kerf)
    assert sheet["cut_count"] == count_cut_nodes(sheet["cut_tree"])

    tree_items = [
        (leaf["item"], leaf["x"], leaf["y"], leaf["width"], leaf["height"],
         leaf["rotated"])
        for leaf in iter_leaves(sheet["cut_tree"])
        if leaf["type"] == "item"
    ]
    placed = [
        (p["item"], p["production_rect"]["x"], p["production_rect"]["y"],
         p["production_rect"]["width"], p["production_rect"]["height"],
         p["rotated"])
        for p in placements
    ]
    assert sorted(map(repr, tree_items)) == sorted(map(repr, placed))

    for p in placements:
        pr = p["production_rect"]
        assert pr["x"] <= p["x"] + EPS and pr["y"] <= p["y"] + EPS
        assert pr["x"] + pr["width"] >= p["x"] + p["width"] - EPS
        assert pr["y"] + pr["height"] >= p["y"] + p["height"] - EPS

    rects = []
    for p in placements:
        pr = p["production_rect"]
        x2, y2 = pr["x"] + pr["width"], pr["y"] + pr["height"]
        assert pr["x"] >= -EPS and pr["y"] >= -EPS
        assert x2 <= width + EPS and y2 <= height + EPS
        rects.append((pr["x"], pr["y"], x2, y2))
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            overlap_x = min(a[2], b[2]) - max(a[0], b[0])
            overlap_y = min(a[3], b[3]) - max(a[1], b[1])
            assert overlap_x <= EPS or overlap_y <= EPS

    used = sum(p["width"] * p["height"] for p in placements)
    assert abs(sheet["utilization"] - used / (width * height)) <= 1e-4


def check_back_sheet(sheet: dict, kerf: float, flip: str) -> None:
    """Validate one sheet's duplex back side.

    * the back side is a physically valid mirrored cut tree (cut lines
      span their node, children stay exactly ``kerf`` apart, implicit trim
      slivers are allowed on either edge after mirroring) and its item
      leaves match ``back_placements`` one-to-one;
    * every back placement is the front placement mirrored across the flip
      axis: net and production rectangles mirrored independently, while
      item/rotated/locked are preserved (asymmetric bleed offsets are not
      re-applied);
    * ``back_cut_tree`` is the recursively mirrored front tree, with
      children swapped exactly at cuts separating the exchanged sides so
      coordinate order stays ascending.
    """
    width, height = sheet["width"], sheet["height"]
    assert "back_placements" in sheet and "back_cut_tree" in sheet
    front, back_places = sheet["placements"], sheet["back_placements"]
    assert len(back_places) == len(front)

    back_view = {
        "placements": sheet["back_placements"],
        "cut_tree": sheet["back_cut_tree"],
        "cut_count": sheet["cut_count"],
        "width": width,
        "height": height,
        "utilization": sheet["utilization"],
    }
    _check_back_sheet_structure(back_view, kerf)

    for p, bp in zip(front, back_places):
        assert bp["item"] == p["item"]
        assert bp["rotated"] == p["rotated"]
        assert bp["locked"] == p["locked"]
        assert bp["width"] == p["width"] and bp["height"] == p["height"]
        nx, ny, nw, nh = _mirrored_rect(p, flip, width, height)
        assert abs(bp["x"] - nx) <= EPS
        assert abs(bp["y"] - ny) <= EPS
        assert bp["width"] == nw and bp["height"] == nh
        px, py, pw, ph = _mirrored_rect(p["production_rect"], flip,
                                        width, height)
        bpr = bp["production_rect"]
        assert abs(bpr["x"] - px) <= EPS
        assert abs(bpr["y"] - py) <= EPS
        assert bpr["width"] == pw and bpr["height"] == ph

    _check_mirrored_tree(sheet["cut_tree"], sheet["back_cut_tree"], flip,
                         width, height)


def check_duplex_plan(plan: dict, flip: str) -> None:
    """Validate the duplex back side of every sheet."""
    kerf = plan["kerf"]
    for sheet in plan["sheets"]:
        check_back_sheet(sheet, kerf, flip)
