#!/usr/bin/env python3
"""Ground → Stairs → Platform → Plum-blossom piles → Platform → Stairs → Ground."""

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
    total_height: float = 1.0

    # 台阶
    stair_depth: float = 0.30
    stair_width_y: float = 2.0
    stair_layers: int = 5

    # 平台
    platform_length_x: float = 2.0
    platform_width_y: float = 3.0

    # 梅花桩
    pile_center_x: float = -3.0
    pile_center_y: float = 0.0
    pile_rows: int = 10
    pile_cols: int = 15
    pile_size_x: float = 0.25
    pile_size_y: float = 0.25
    pile_spacing_x: float = 0.45
    pile_spacing_y: float = 0.45
    pile_stagger: bool = True


def build_scene(output_path: Path, cfg: TerrainCfg | None = None):
    cfg = cfg or TerrainCfg()
    root = ET.parse(TEMPLATE_SCENE).getroot()

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

    wb = root.find("worldbody")
    wb.clear()
    ET.SubElement(wb, "light", {"pos": "0 0 2.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {
        "name": "floor", "group": "2", "size": "0 0 0.05",
        "type": "plane", "material": "groundplane",
    })

    th = cfg.total_height
    step_h = th / cfg.stair_layers
    sx = cfg.pile_size_x / 2.0
    sy = cfg.pile_size_y / 2.0

    # ====== 上行台阶 ======
    up_stair_right = 5.0
    wb.append(ET.Comment(" Up stairs "))
    for i in range(cfg.stair_layers):
        h = step_h * (i + 1)
        cx = up_stair_right - (i + 0.5) * cfg.stair_depth
        ET.SubElement(wb, "geom", {
            "name": f"up_stair_{i}", "group": "2",
            "pos": f"{fmt(cx)} 0 {fmt(h / 2.0)}",
            "type": "box",
            "size": f"{fmt(cfg.stair_depth / 2.0)} {fmt(cfg.stair_width_y / 2.0)} {fmt(h / 2.0)}",
            "material": "stair_mat",
        })

    # ====== 平台1（上行台阶左侧） ======
    up_stair_left = up_stair_right - cfg.stair_layers * cfg.stair_depth
    plat1_right = up_stair_left
    plat1_left = plat1_right - cfg.platform_length_x
    plat1_cx = (plat1_right + plat1_left) / 2.0

    wb.append(ET.Comment(f" Platform 1: x=[{fmt(plat1_left)}, {fmt(plat1_right)}]"))
    ET.SubElement(wb, "geom", {
        "name": "platform_1", "group": "2",
        "pos": f"{fmt(plat1_cx)} 0 {fmt(th / 2.0)}",
        "type": "box",
        "size": f"{fmt(cfg.platform_length_x / 2.0)} {fmt(cfg.platform_width_y / 2.0)} {fmt(th / 2.0)}",
        "material": "platform_mat",
    })

    # ====== 梅花桩（紧接平台1左侧） ======
    half_cols = (cfg.pile_cols - 1) / 2.0
    half_rows = (cfg.pile_rows - 1) / 2.0
    pile_edge_right = plat1_left
    pile_center_x = pile_edge_right - sx - half_cols * cfg.pile_spacing_x

    wb.append(ET.Comment(
        f" Plum-blossom piles: {cfg.pile_rows}x{cfg.pile_cols}, "
        f"center_x={fmt(pile_center_x)}, top z={fmt(th)}"
    ))

    for row in range(cfg.pile_rows):
        for col in range(cfg.pile_cols):
            x_off = (cfg.pile_spacing_x / 2.0) if (cfg.pile_stagger and row % 2 == 1) else 0.0
            local_x = (col - half_cols) * cfg.pile_spacing_x + x_off
            local_y = (row - half_rows) * cfg.pile_spacing_y

            px = pile_center_x + local_x
            py = local_y

            ET.SubElement(wb, "geom", {
                "name": f"pile_r{row}_c{col}", "group": "2",
                "pos": f"{fmt(px)} {fmt(py)} {fmt(th / 2.0)}",
                "type": "box",
                "size": f"{fmt(sx)} {fmt(sy)} {fmt(th / 2.0)}",
                "material": "pile_mat",
            })

    # ====== 平台2（梅花桩左侧） ======
    pile_edge_left = pile_center_x - half_cols * cfg.pile_spacing_x - sx
    plat2_right = pile_edge_left
    plat2_left = plat2_right - cfg.platform_length_x
    plat2_cx = (plat2_right + plat2_left) / 2.0

    wb.append(ET.Comment(f" Platform 2: x=[{fmt(plat2_left)}, {fmt(plat2_right)}]"))
    ET.SubElement(wb, "geom", {
        "name": "platform_2", "group": "2",
        "pos": f"{fmt(plat2_cx)} 0 {fmt(th / 2.0)}",
        "type": "box",
        "size": f"{fmt(cfg.platform_length_x / 2.0)} {fmt(cfg.platform_width_y / 2.0)} {fmt(th / 2.0)}",
        "material": "platform_mat",
    })

    # ====== 下行台阶 ======
    down_stair_right = plat2_left
    wb.append(ET.Comment(f" Down stairs: right_edge={fmt(down_stair_right)}"))
    for i in range(cfg.stair_layers):
        h = th - step_h * i
        if h <= 0:
            break
        cx = down_stair_right - (i + 0.5) * cfg.stair_depth
        ET.SubElement(wb, "geom", {
            "name": f"down_stair_{i}", "group": "2",
            "pos": f"{fmt(cx)} 0 {fmt(h / 2.0)}",
            "type": "box",
            "size": f"{fmt(cfg.stair_depth / 2.0)} {fmt(cfg.stair_width_y / 2.0)} {fmt(h / 2.0)}",
            "material": "stair_mat",
        })

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=False)
    print(f"Generated terrain: {output_path}")
    print(f"  Layout (x): stairs↑[{fmt(up_stair_left)}, {fmt(up_stair_right)}] | "
          f"plat1[{fmt(plat1_left)}, {fmt(plat1_right)}] | "
          f"piles[{fmt(pile_edge_left)}, {fmt(pile_edge_right)}] | "
          f"plat2[{fmt(plat2_left)}, {fmt(plat2_right)}] | "
          f"stairs↓[{fmt(down_stair_right - cfg.stair_layers * cfg.stair_depth)}, {fmt(down_stair_right)}]")
    print(f"  Robot spawn: ({fmt(up_stair_right + 2.0)}, 0, 0.8)")


if __name__ == "__main__":
    build_scene(DEFAULT_OUTPUT)
