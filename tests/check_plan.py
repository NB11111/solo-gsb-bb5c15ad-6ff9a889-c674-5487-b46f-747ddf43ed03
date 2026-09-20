"""Shared validators asserting a plan is physically and structurally sound."""
from __future__ import annotations

import pytest

EPS = 1e-6


def approx(value: float) -> float:
    return pytest.approx(value, abs=EPS)


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
# duplex printing (双面印刷背面定位)
# ---------------------------------------------------------------------------

def mirror_rect(rect: dict, duplex_flip: str,
                sheet_width: float, sheet_height: float) -> tuple:
    """Expected (x, y, width, height) of a rectangle after duplex mirroring."""
    if duplex_flip == "left_right":
        return (sheet_width - rect["x"] - rect["width"], rect["y"],
                rect["width"], rect["height"])
    return (rect["x"], sheet_height - rect["y"] - rect["height"],
            rect["width"], rect["height"])


def _rect_tuple(node: dict) -> tuple:
    return (node["x"], node["y"], node["width"], node["height"])


def check_mirrored_tree(front: dict, back: dict, duplex_flip: str,
                        sheet_width: float, sheet_height: float) -> None:
    """Assert ``back`` is the exact duplex mirror of ``front``.

    Every node's rectangle is mirrored geometrically; a cut parallel to the
    mirror axis must swap its two children (so the first child keeps the
    smaller coordinate along the cut's ordering axis); a perpendicular cut
    keeps the child order.  Item identity and orientation carry over.
    """
    assert back["type"] == front["type"]
    assert _rect_tuple(back) == mirror_rect(
        front, duplex_flip, sheet_width, sheet_height)

    if front["type"] == "item":
        assert back["item"] == front["item"]
        assert back["rotated"] == front["rotated"]
        return
    if front["type"] == "waste":
        return

    assert back["direction"] == front["direction"]
    f0, f1 = front["children"]
    b0, b1 = back["children"]
    parallel = (
        (duplex_flip == "left_right" and front["direction"] == "vertical")
        or (duplex_flip == "top_bottom" and front["direction"] == "horizontal")
    )
    if parallel:
        # First/second (left-right or top-bottom) children swap sides.
        check_mirrored_tree(f1, b0, duplex_flip, sheet_width, sheet_height)
        check_mirrored_tree(f0, b1, duplex_flip, sheet_width, sheet_height)
    else:
        check_mirrored_tree(f0, b0, duplex_flip, sheet_width, sheet_height)
        check_mirrored_tree(f1, b1, duplex_flip, sheet_width, sheet_height)


def check_back_cut_tree(node: dict, kerf: float, duplex_flip: str) -> None:
    """Verify a duplex back tree with back-side conventions.

    The back tree is a geometric mirror of the front.  Mirroring turns every
    near-edge (left/top) anchor into a far-edge (right/bottom) anchor:

    * along the cut direction the children keep ascending coordinate order
      (parallel cuts swap sides) and the blade still consumes exactly one
      ``kerf`` between them; without flipping, the first child is anchored
      at the node's near edge; with flipping, the second child is anchored
      at the far edge;
    * perpendicular to the cut both children share the node's span: without
      flipping they are anchored at the near edge, with flipping at the far
      edge;
    * all children stay inside the node bounds and leaves are positive.
    """
    if node["type"] != "cut":
        assert node["width"] > 0 and node["height"] > 0
        return

    assert node["direction"] in ("vertical", "horizontal")
    assert len(node["children"]) == 2
    first, second = node["children"]

    if node["direction"] == "vertical":
        span_axis_flipped = duplex_flip == "top_bottom"
        cut_axis_flipped = duplex_flip == "left_right"
        node_near_y, node_far_y = node["y"], node["y"] + node["height"]
        node_near_x, node_far_x = node["x"], node["x"] + node["width"]

        # Perpendicular axis (y): both children anchored to one edge.
        for child in (first, second):
            assert child["height"] <= node["height"] + EPS
            if span_axis_flipped:
                assert child["y"] + child["height"] == approx(node_far_y)
            else:
                assert child["y"] == approx(node_near_y)

        # Cut axis (x): ascending order with the exact kerf gap.
        assert first["x"] <= second["x"] + EPS
        assert second["x"] == approx(first["x"] + first["width"] + kerf)
        for child in (first, second):
            assert child["x"] >= node_near_x - EPS
            assert child["x"] + child["width"] <= node_far_x + EPS
        if cut_axis_flipped:
            assert second["x"] + second["width"] == approx(node_far_x)
        else:
            assert first["x"] == approx(node_near_x)
    else:
        span_axis_flipped = duplex_flip == "left_right"
        cut_axis_flipped = duplex_flip == "top_bottom"
        node_near_x, node_far_x = node["x"], node["x"] + node["width"]
        node_near_y, node_far_y = node["y"], node["y"] + node["height"]

        # Perpendicular axis (x): both children anchored to one edge.
        for child in (first, second):
            assert child["width"] <= node["width"] + EPS
            if span_axis_flipped:
                assert child["x"] + child["width"] == approx(node_far_x)
            else:
                assert child["x"] == approx(node_near_x)

        # Cut axis (y): ascending order with the exact kerf gap.
        assert first["y"] <= second["y"] + EPS
        assert second["y"] == approx(first["y"] + first["height"] + kerf)
        for child in (first, second):
            assert child["y"] >= node_near_y - EPS
            assert child["y"] + child["height"] <= node_far_y + EPS
        if cut_axis_flipped:
            assert second["y"] + second["height"] == approx(node_far_y)
        else:
            assert first["y"] == approx(node_near_y)

    check_back_cut_tree(first, kerf, duplex_flip)
    check_back_cut_tree(second, kerf, duplex_flip)


def check_duplex_sheet(sheet: dict, kerf: float, duplex_flip: str,
                       width: float, height: float) -> None:
    """Validate one sheet's back side: tree, leaves vs. back placements,
    bounds, non-overlap and utilization (net area is unchanged by mirror)."""
    placements = sheet["back_placements"]

    check_back_cut_tree(sheet["back_cut_tree"], kerf, duplex_flip)

    # Tree item leaves must be exactly the back placements' production rects.
    tree_items = [
        (leaf["item"], leaf["x"], leaf["y"], leaf["width"], leaf["height"],
         leaf["rotated"])
        for leaf in iter_leaves(sheet["back_cut_tree"])
        if leaf["type"] == "item"
    ]
    placed = [
        (p["item"], p["production_rect"]["x"], p["production_rect"]["y"],
         p["production_rect"]["width"], p["production_rect"]["height"],
         p["rotated"])
        for p in placements
    ]
    assert sorted(map(repr, tree_items)) == sorted(map(repr, placed))

    rects = []
    for p in placements:
        pr = p["production_rect"]
        x2, y2 = pr["x"] + pr["width"], pr["y"] + pr["height"]
        assert pr["x"] >= -EPS and pr["y"] >= -EPS
        assert x2 <= width + EPS and y2 <= height + EPS
        assert pr["x"] <= p["x"] + EPS and pr["y"] <= p["y"] + EPS
        assert pr["x"] + pr["width"] >= p["x"] + p["width"] - EPS
        assert pr["y"] + pr["height"] >= p["y"] + p["height"] - EPS
        rects.append((pr["x"], pr["y"], x2, y2))
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            overlap_x = min(a[2], b[2]) - max(a[0], b[0])
            overlap_y = min(a[3], b[3]) - max(a[1], b[1])
            assert overlap_x <= EPS or overlap_y <= EPS

    used = sum(p["width"] * p["height"] for p in placements)
    assert abs(sheet["utilization"] - used / (width * height)) <= 1e-4


def check_duplex_plan(plan: dict, duplex_flip: str) -> None:
    """Validate front and back sides of a duplex plan together."""
    width, height = plan["stock"]["width"], plan["stock"]["height"]
    for sheet in plan["sheets"]:
        assert "back_placements" in sheet and sheet["back_placements"] is not None
        assert "back_cut_tree" in sheet and sheet["back_cut_tree"] is not None

        check_duplex_sheet(sheet, plan["kerf"], duplex_flip, width, height)

        # Back placement order follows the front; identity flags carry over
        # and both the net and production rectangles are plain mirrors.
        assert len(sheet["back_placements"]) == len(sheet["placements"])
        for front, back in zip(sheet["placements"],
                               sheet["back_placements"]):
            assert back["item"] == front["item"]
            assert back["rotated"] == front["rotated"]
            assert back["locked"] == front["locked"]
            assert (back["x"], back["y"], back["width"], back["height"]) == \
                mirror_rect(front, duplex_flip, width, height)
            assert (back["production_rect"]["x"], back["production_rect"]["y"],
                    back["production_rect"]["width"],
                    back["production_rect"]["height"]) == \
                mirror_rect(front["production_rect"], duplex_flip,
                            width, height)

        check_mirrored_tree(sheet["cut_tree"], sheet["back_cut_tree"],
                            duplex_flip, width, height)
