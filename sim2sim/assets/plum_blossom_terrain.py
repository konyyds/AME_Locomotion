#!/usr/bin/env python3
"""Ground → One-sided stairs → Platform → Plum-blossom piles.

Robot spawns on ground (z=0), walks up staircase to platform (z=0.4m),
then onto plum-blossom piles with tops flush at z=0.4m.
All structures are adjacent — no gaps.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent
TEMPLATE_SCENE = ASSETS_DIR / "scene_flat.xml"
DEFAULT_OUTPUT = ASSETS_DIR / "scene_plum_blossom.xml"


def fmt(v):
    return f"{v:.6g}" if abs(v) >= 5e-8 else "0"


@dataclass
class TerrainCfg:
    total_height: float = 0.40

    # 台阶
    stairs_right_x: float = 4.0   # 最矮台阶右边缘（机器人从这边上）
    stair_depth: float = 0.30     # 每级台阶 x 方向深度
    stair_width_y: float = 2.0    # 台阶 y 方向宽度
    stair_layers: int = 5

    # 平台（紧接台阶左侧）
    platform_length_x: float = 2.5
    platform_width_y: float = 3.0

    # 梅花桩（紧接平台左侧）
    pile_rows: int = 10
    pile_cols: int = 15
    pile_size_x: float = 0.20
    pile_size_y: float = 0.20
    pile_spacing_x: float = 0.35
    pile_spacing_y: float = 0.35
    pile_stagger: bool = True


def build_scene(output_path: Path, cfg: TerrainCfg | None = None):
    cfg = cfg or TerrainCfg()
    root = ET.parse(TEMPLATE_SCENE).getroot()

    # --- Asset ---
    asset = root.find("asset")
    asset.clear()
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0",
        "width": "512", "height": "3072",
    })
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "groundplane", "builtin": "checker",
        "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3",
        "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300",
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane",
        "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2",
    })
    ET.SubElement(asset, "material", {"name": "stair_mat", "rgba": "0.55 0.58 0.52 1"})
    ET.SubElement(asset, "material", {"name": "platform_mat", "rgba": "0.46 0.46 0.58 1"})
    ET.SubElement(asset, "material", {"name": "pile_mat", "rgba": "0.7 0.45 0.3 1"})

    # --- Worldbody ---
    wb = root.find("worldbody")
    wb.clear()
    ET.SubElement(wb, "light", {"pos": "0 0 2.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {
        "name": "floor", "group": "2", "size": "0 0 0.05",
        "type": "plane", "material": "groundplane",
    })

    th = cfg.total_height

    # ====== 1. 台阶（单侧阶梯，从右往左逐级升高） ======
    step_h = th / cfg.stair_layers
    wb.append(ET.Comment(
        f" Stairs: {cfg.stair_layers} steps, rightmost at x={fmt(cfg.stairs_right_x)}"
    ))
    for i in range(cfg.stair_layers):
        h = step_h * (i + 1)
        cx = cfg.stairs_right_x - (i + 0.5) * cfg.stair_depth
        ET.SubElement(wb, "geom", {
            "name": f"stair_{i}", "group": "2",
            "pos": f"{fmt(cx)} 0 {fmt(h / 2.0)}",
            "type": "box",
            "size": f"{fmt(cfg.stair_depth / 2.0)} {fmt(cfg.stair_width_y / 2.0)} {fmt(h / 2.0)}",
            "material": "stair_mat",
        })

    # ====== 2. 平台（紧接台阶左侧） ======
    stairs_left_x = cfg.stairs_right_x - cfg.stair_layers * cfg.stair_depth
    plat_right_x = stairs_left_x        # 紧挨台阶
    plat_left_x = plat_right_x - cfg.platform_length_x
    plat_cx = (plat_right_x + plat_left_x) / 2.0

    wb.append(ET.Comment(
        f" Platform: x=[{fmt(plat_left_x)}, {fmt(plat_right_x)}], top z={fmt(th)}"
    ))
    ET.SubElement(wb, "geom", {
        "name": "platform", "group": "2",
        "pos": f"{fmt(plat_cx)} 0 {fmt(th / 2.0)}",
        "type": "box",
        "size": f"{fmt(cfg.platform_length_x / 2.0)} {fmt(cfg.platform_width_y / 2.0)} {fmt(th / 2.0)}",
        "material": "platform_mat",
    })

    # ====== 3. 梅花桩（紧接平台左侧） ======
    pile_right_edge = plat_left_x       # 紧挨平台
    half_cols = (cfg.pile_cols - 1) / 2.0
    pile_center_x = pile_right_edge - cfg.pile_size_x / 2.0 - half_cols * cfg.pile_spacing_x

    wb.append(ET.Comment(
        f" Plum-blossom piles: {cfg.pile_rows}x{cfg.pile_cols}, "
        f"center_x={fmt(pile_center_x)}, top z={fmt(th)}"
    ))
    sx = cfg.pile_size_x / 2.0
    sy = cfg.pile_size_y / 2.0

    for row in range(cfg.pile_rows):
        for col in range(cfg.pile_cols):
            x_off = (cfg.pile_spacing_x / 2.0) if (cfg.pile_stagger and row % 2 == 1) else 0.0
            local_x = (col - half_cols) * cfg.pile_spacing_x + x_off
            local_y = (row - (cfg.pile_rows - 1) / 2.0) * cfg.pile_spacing_y

            px = pile_center_x + local_x
            py = local_y

            ET.SubElement(wb, "geom", {
                "name": f"pile_r{row}_c{col}", "group": "2",
                "pos": f"{fmt(px)} {fmt(py)} {fmt(th / 2.0)}",
                "type": "box",
                "size": f"{fmt(sx)} {fmt(sy)} {fmt(th / 2.0)}",
                "material": "pile_mat",
            })

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=False)
    print(f"Generated terrain: {output_path}")
    print(f"  Layout (x):  piles[{fmt(pile_center_x - half_cols * cfg.pile_spacing_x)}, "
          f"{fmt(pile_right_edge)}] | platform[{fmt(plat_left_x)}, {fmt(plat_right_x)}] | "
          f"stairs[{fmt(stairs_left_x)}, {fmt(cfg.stairs_right_x)}]")
    print(f"  All tops at z={fmt(th)}")
    print(f"  Robot spawn: ({fmt(cfg.stairs_right_x + 1.5)}, 0, 0.8)")


if __name__ == "__main__":
    build_scene(DEFAULT_OUTPUT)
