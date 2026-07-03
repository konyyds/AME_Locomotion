#!/usr/bin/env python3
"""Generate MuJoCo rough terrain scenes for sim2sim raycaster tests.

The default configuration recreates the four terrain patches currently used by
scene_rough.xml: pyramid stairs, box grid, slope pyramid, and random heightfield.
Edit DEFAULT_TERRAINS or pass a different output path on the command line.
"""

from __future__ import annotations

import argparse
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


ASSETS_DIR = Path(__file__).resolve().parent
TEMPLATE_SCENE = ASSETS_DIR / "scene_flat.xml"
DEFAULT_OUTPUT = ASSETS_DIR / "scene_rough.xml"


def fmt(value: float) -> str:
    if abs(value) < 5e-8:
        value = 0.0
    return f"{value:.6g}"


def vec(values: Iterable[float]) -> str:
    return " ".join(fmt(float(value)) for value in values)


def rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return x * cos_yaw - y * sin_yaw, x * sin_yaw + y * cos_yaw


def euler_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


@dataclass
class PyramidStairsCfg:
    name: str = "pyramid_stairs"
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    width: float = 5.0
    length: float = 5.0
    yaw: float = 0.0
    num_layers: int = 8
    layer_height: float = 0.08
    step_width: float = 0.25
    material: str = "pyramid_stairs_mat"
    group: int = 2


@dataclass
class BoxGridCfg:
    name: str = "boxes"
    center: tuple[float, float, float] = (6.0, 0.0, 0.0)
    width: float = 5.0
    length: float = 5.0
    yaw: float = 0.0
    nrow: int = 7
    ncol: int = 7
    spacing_x: float | None = None
    spacing_y: float | None = None
    overlap: float = 0.001
    min_height: float = 0.06
    max_height: float = 0.23
    height_values: Sequence[Sequence[float]] | None = None
    seed: int = 11
    material: str = "boxes_mat"
    group: int = 2


@dataclass
class SlopePyramidCfg:
    name: str = "slope_pyramid"
    center: tuple[float, float, float] = (12.0, 0.0, 0.0)
    width: float = 5.0
    length: float = 5.0
    top_width: float = 2.0
    top_length: float = 2.0
    height: float = 0.4
    yaw: float = 0.0
    material: str = "slope_mat"
    group: int = 2


@dataclass
class RandomRoughHFieldCfg:
    name: str = "random_rough"
    center: tuple[float, float, float] = (18.0, 0.0, 0.0)
    width: float = 5.0
    length: float = 5.0
    height_scale: float = 0.20
    collision_depth: float = 0.04
    nrow: int = 33
    ncol: int = 33
    seed: int = 7
    border_zero: bool = True
    smooth_passes: int = 0
    smooth_strength: float = 0.2
    elevation_values: Sequence[Sequence[float]] | None = None
    material: str = "random_rough_mat"
    group: int = 2


TerrainCfg = PyramidStairsCfg | BoxGridCfg | SlopePyramidCfg | RandomRoughHFieldCfg


DEFAULT_TERRAINS: tuple[TerrainCfg, ...] = (
    PyramidStairsCfg(
        center=(0.0, 0.0, 0.0),
        width=5.0,
        length=5.0,
        num_layers=8,
        layer_height=0.10,
        step_width=0.25,
    ),
    BoxGridCfg(
        center=(6.0, 0.0, 0.0),
        width=5.0,
        length=5.0,
        nrow=12,
        ncol=12,
        min_height=0.06,
        max_height=0.23,
    ),
    SlopePyramidCfg(
        center=(12.0, 0.0, 0.0),
        width=5.0,
        length=5.0,
        top_width=1.0,
        top_length=1.0,
        height=0.5,
    ),
    RandomRoughHFieldCfg(
        center=(18.0, 0.0, 0.0),
        width=5.0,
        length=5.0,
        nrow=24,
        ncol=24,
        height_scale=0.20,
        smooth_passes=1,
    ),
)


@dataclass
class SceneCfg:
    terrains: Sequence[TerrainCfg] = field(default_factory=lambda: DEFAULT_TERRAINS)


class RoughTerrainGenerator:
    def __init__(self, scene_cfg: SceneCfg | None = None) -> None:
        self.scene_cfg = scene_cfg or SceneCfg()
        self.root = ET.parse(TEMPLATE_SCENE).getroot()
        self.asset: ET.Element
        self.worldbody: ET.Element
        self._replace_terrain_scene()

    def _replace_terrain_scene(self) -> None:
        self.asset = self._required_child(self.root, "asset")
        self.asset.clear()
        ET.SubElement(
            self.asset,
            "texture",
            {
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.3 0.5 0.7",
                "rgb2": "0 0 0",
                "width": "512",
                "height": "3072",
            },
        )
        ET.SubElement(
            self.asset,
            "texture",
            {
                "type": "2d",
                "name": "groundplane",
                "builtin": "checker",
                "mark": "edge",
                "rgb1": "0.2 0.3 0.4",
                "rgb2": "0.1 0.2 0.3",
                "markrgb": "0.8 0.8 0.8",
                "width": "300",
                "height": "300",
            },
        )
        ET.SubElement(
            self.asset,
            "material",
            {
                "name": "groundplane",
                "texture": "groundplane",
                "texuniform": "true",
                "texrepeat": "5 5",
                "reflectance": "0.2",
            },
        )
        ET.SubElement(self.asset, "material", {"name": "pyramid_stairs_mat", "rgba": "0.55 0.58 0.52 1"})
        ET.SubElement(self.asset, "material", {"name": "boxes_mat", "rgba": "0.62 0.48 0.34 1"})
        ET.SubElement(self.asset, "material", {"name": "slope_mat", "rgba": "0.46 0.46 0.58 1"})
        ET.SubElement(self.asset, "material", {"name": "random_rough_mat", "rgba": "0.36 0.52 0.46 1"})

        self.worldbody = self._required_child(self.root, "worldbody")
        self.worldbody.clear()
        ET.SubElement(self.worldbody, "light", {"pos": "0 0 2.5", "dir": "0 0 -1", "directional": "true"})
        ET.SubElement(
            self.worldbody,
            "geom",
            {"name": "floor", "group": "2", "size": "0 0 0.05", "type": "plane", "material": "groundplane"},
        )

    def _required_child(self, parent: ET.Element, tag: str) -> ET.Element:
        child = parent.find(tag)
        if child is None:
            raise ValueError(f"Template scene is missing <{tag}>")
        return child

    def add_all(self, terrains: Sequence[TerrainCfg] | None = None) -> None:
        for terrain in terrains or self.scene_cfg.terrains:
            self.add_terrain(terrain)

    def add_terrain(self, terrain: TerrainCfg) -> None:
        if isinstance(terrain, PyramidStairsCfg):
            self.add_pyramid_stairs(terrain)
        elif isinstance(terrain, BoxGridCfg):
            self.add_box_grid(terrain)
        elif isinstance(terrain, SlopePyramidCfg):
            self.add_slope_pyramid(terrain)
        elif isinstance(terrain, RandomRoughHFieldCfg):
            self.add_random_rough_hfield(terrain)
        else:
            raise TypeError(f"Unsupported terrain config: {terrain!r}")

    def _add_box(
        self,
        name: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        material: str,
        group: int,
        yaw: float = 0.0,
    ) -> None:
        attrib = {
            "name": name,
            "group": str(group),
            "pos": vec(center),
            "type": "box",
            "size": vec((size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)),
            "material": material,
        }
        if abs(yaw) > 1e-8:
            attrib["quat"] = vec(euler_to_quat(0.0, 0.0, yaw))
        ET.SubElement(self.worldbody, "geom", attrib)

    def add_pyramid_stairs(self, cfg: PyramidStairsCfg) -> None:
        self.worldbody.append(ET.Comment(f" {cfg.name}: centered at {vec(cfg.center)}, width={fmt(cfg.width)}, length={fmt(cfg.length)}. "))
        for layer_idx in range(cfg.num_layers):
            layer_width = cfg.width - 2.0 * cfg.step_width * layer_idx
            layer_length = cfg.length - 2.0 * cfg.step_width * layer_idx
            if layer_width <= 0.0 or layer_length <= 0.0:
                break

            z = cfg.center[2] + cfg.layer_height * (layer_idx + 0.5)
            self._add_box(
                name=f"{cfg.name}_l{layer_idx}",
                center=(cfg.center[0], cfg.center[1], z),
                size=(layer_width, layer_length, cfg.layer_height),
                material=cfg.material,
                group=cfg.group,
                yaw=cfg.yaw,
            )

    def add_box_grid(self, cfg: BoxGridCfg) -> None:
        heights = self._box_heights(cfg)
        x_step = cfg.spacing_x if cfg.spacing_x is not None else cfg.width / cfg.ncol
        y_step = cfg.spacing_y if cfg.spacing_y is not None else cfg.length / cfg.nrow
        box_width = x_step + cfg.overlap
        box_length = y_step + cfg.overlap

        self.worldbody.append(ET.Comment(f" {cfg.name}: centered at {vec(cfg.center)}, width={fmt(cfg.width)}, length={fmt(cfg.length)}. "))
        for row in range(cfg.nrow):
            for col in range(cfg.ncol):
                local_x = (col - (cfg.ncol - 1) / 2.0) * x_step
                local_y = (row - (cfg.nrow - 1) / 2.0) * y_step
                rot_x, rot_y = rotate_xy(local_x, local_y, cfg.yaw)
                height = heights[row][col]
                self._add_box(
                    name=f"{cfg.name}_r{row}_c{col}",
                    center=(cfg.center[0] + rot_x, cfg.center[1] + rot_y, cfg.center[2] + height / 2.0),
                    size=(box_width, box_length, height),
                    material=cfg.material,
                    group=cfg.group,
                    yaw=cfg.yaw,
                )

    def _box_heights(self, cfg: BoxGridCfg) -> list[list[float]]:
        if cfg.height_values is not None:
            if len(cfg.height_values) != cfg.nrow or any(len(row) != cfg.ncol for row in cfg.height_values):
                raise ValueError(f"{cfg.name}.height_values must have shape nrow x ncol")
            return [[float(value) for value in row] for row in cfg.height_values]

        rng = random.Random(cfg.seed)
        return [
            [rng.uniform(cfg.min_height, cfg.max_height) for _ in range(cfg.ncol)]
            for _ in range(cfg.nrow)
        ]

    def add_slope_pyramid(self, cfg: SlopePyramidCfg) -> None:
        mesh_name = f"{cfg.name}_mesh"
        vertices = self._slope_vertices(cfg)
        faces = (
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        )
        ET.SubElement(self.asset, "mesh", {"name": mesh_name, "vertex": vec(vertices), "face": vec(faces)})
        self.worldbody.append(ET.Comment(f" {cfg.name}: centered at {vec(cfg.center)}, width={fmt(cfg.width)}, length={fmt(cfg.length)}. "))

        attrib = {
            "name": cfg.name,
            "group": str(cfg.group),
            "pos": vec(cfg.center),
            "type": "mesh",
            "mesh": mesh_name,
            "material": cfg.material,
        }
        if abs(cfg.yaw) > 1e-8:
            attrib["quat"] = vec(euler_to_quat(0.0, 0.0, cfg.yaw))
        ET.SubElement(self.worldbody, "geom", attrib)

    def _slope_vertices(self, cfg: SlopePyramidCfg) -> tuple[float, ...]:
        bottom_x = cfg.width / 2.0
        bottom_y = cfg.length / 2.0
        top_x = cfg.top_width / 2.0
        top_y = cfg.top_length / 2.0
        return (
            -bottom_x, -bottom_y, 0.0,
            bottom_x, -bottom_y, 0.0,
            bottom_x, bottom_y, 0.0,
            -bottom_x, bottom_y, 0.0,
            -top_x, -top_y, cfg.height,
            top_x, -top_y, cfg.height,
            top_x, top_y, cfg.height,
            -top_x, top_y, cfg.height,
        )

    def add_random_rough_hfield(self, cfg: RandomRoughHFieldCfg) -> None:
        elevation = self._hfield_elevation(cfg)
        hfield_name = f"{cfg.name}_hfield"
        ET.SubElement(
            self.asset,
            "hfield",
            {
                "name": hfield_name,
                "size": vec((cfg.width / 2.0, cfg.length / 2.0, cfg.height_scale, cfg.collision_depth)),
                "nrow": str(cfg.nrow),
                "ncol": str(cfg.ncol),
                "elevation": "\n" + self._format_elevation(elevation) + "\n      ",
            },
        )
        self.worldbody.append(ET.Comment(f" {cfg.name}: centered at {vec(cfg.center)}, width={fmt(cfg.width)}, length={fmt(cfg.length)}. "))
        ET.SubElement(
            self.worldbody,
            "geom",
            {
                "name": cfg.name,
                "group": str(cfg.group),
                "pos": vec(cfg.center),
                "type": "hfield",
                "hfield": hfield_name,
                "material": cfg.material,
            },
        )

    def _hfield_elevation(self, cfg: RandomRoughHFieldCfg) -> list[list[float]]:
        if cfg.elevation_values is not None:
            if len(cfg.elevation_values) != cfg.nrow or any(len(row) != cfg.ncol for row in cfg.elevation_values):
                raise ValueError(f"{cfg.name}.elevation_values must have shape nrow x ncol")
            return [[float(value) for value in row] for row in cfg.elevation_values]

        rng = random.Random(cfg.seed)
        values = [[rng.random() for _ in range(cfg.ncol)] for _ in range(cfg.nrow)]
        if cfg.border_zero:
            self._zero_hfield_border(values)
        for _ in range(cfg.smooth_passes):
            values = self._smooth_hfield(values, cfg.smooth_strength)
            if cfg.border_zero:
                self._zero_hfield_border(values)
        return values

    def _zero_hfield_border(self, values: list[list[float]]) -> None:
        if not values:
            return
        last_row = len(values) - 1
        last_col = len(values[0]) - 1
        for row_idx, row in enumerate(values):
            for col_idx in range(len(row)):
                if row_idx in (0, last_row) or col_idx in (0, last_col):
                    row[col_idx] = 0.0

    def _smooth_hfield(self, values: list[list[float]], strength: float) -> list[list[float]]:
        if strength < 0.0 or strength > 1.0:
            raise ValueError("smooth_strength must be between 0.0 and 1.0")

        nrow = len(values)
        ncol = len(values[0])
        smoothed: list[list[float]] = []
        for row_idx in range(nrow):
            row_values: list[float] = []
            for col_idx in range(ncol):
                total = 0.0
                count = 0
                for r in range(max(0, row_idx - 1), min(nrow, row_idx + 2)):
                    for c in range(max(0, col_idx - 1), min(ncol, col_idx + 2)):
                        total += values[r][c]
                        count += 1
                average = total / count
                row_values.append(values[row_idx][col_idx] * (1.0 - strength) + average * strength)
            smoothed.append(row_values)
        return smoothed

    def _format_elevation(self, values: Sequence[Sequence[float]]) -> str:
        return "\n".join("        " + " ".join(f"{value:.2f}" for value in row) for row in values)

    def save(self, output_path: Path) -> None:
        ET.indent(self.root, space="  ")
        tree = ET.ElementTree(self.root)
        tree.write(output_path, encoding="utf-8", xml_declaration=False)


def build_scene(output_path: Path) -> None:
    generator = RoughTerrainGenerator()
    generator.add_all()
    generator.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a MuJoCo scene with configurable rough terrain patches.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output XML path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = ASSETS_DIR / output_path
    build_scene(output_path)
    print(f"Generated rough terrain scene: {output_path}")


if __name__ == "__main__":
    main()
