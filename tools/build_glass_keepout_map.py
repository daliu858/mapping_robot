#!/usr/bin/env python3
"""Create a derived occupancy map with one verified-clear patch and glass wall."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def world_to_pixel(x, y, origin, resolution, height):
    column = int(round((x - origin[0]) / resolution))
    row = height - 1 - int(round((y - origin[1]) / resolution))
    return column, row


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-yaml", required=True)
    parser.add_argument("--input-image")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--yaml-image", required=True)
    parser.add_argument("--clear-x", type=float, required=True)
    parser.add_argument("--clear-y", type=float, required=True)
    parser.add_argument("--clear-radius", type=float, default=0.35)
    parser.add_argument("--wall-x", type=float, required=True)
    parser.add_argument("--wall-y", type=float, required=True)
    parser.add_argument("--wall-yaw", type=float, required=True)
    parser.add_argument("--wall-length", type=float, default=1.50)
    parser.add_argument("--wall-thickness", type=float, default=0.10)
    args = parser.parse_args()

    input_yaml = Path(args.input_yaml).resolve()
    config = yaml.safe_load(input_yaml.read_text(encoding="utf-8"))
    input_image = Path(args.input_image) if args.input_image else Path(config["image"])
    if not input_image.is_absolute():
        input_image = input_yaml.parent / input_image
    if not input_image.exists() and not args.input_image:
        local_peer = input_yaml.parent / Path(str(config["image"])).name
        if local_peer.exists():
            input_image = local_peer
    image = Image.open(input_image).convert("L")
    width, height = image.size
    resolution = float(config["resolution"])
    origin = [float(value) for value in config["origin"]]
    draw = ImageDraw.Draw(image)

    clear_center = world_to_pixel(
        args.clear_x, args.clear_y, origin, resolution, height)
    clear_pixels = int(math.ceil(args.clear_radius / resolution))
    draw.ellipse(
        (clear_center[0] - clear_pixels, clear_center[1] - clear_pixels,
         clear_center[0] + clear_pixels, clear_center[1] + clear_pixels),
        fill=254)

    half = args.wall_length / 2.0
    dx = half * math.cos(args.wall_yaw)
    dy = half * math.sin(args.wall_yaw)
    first = world_to_pixel(
        args.wall_x - dx, args.wall_y - dy, origin, resolution, height)
    second = world_to_pixel(
        args.wall_x + dx, args.wall_y + dy, origin, resolution, height)
    wall_pixels = max(1, int(math.ceil(args.wall_thickness / resolution)))
    draw.line((first, second), fill=0, width=wall_pixels)

    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_image = output_prefix.with_suffix(".pgm")
    output_yaml = output_prefix.with_suffix(".yaml")
    output_meta = output_prefix.with_suffix(".glass_keepout.json")
    image.save(output_image)

    output_config = dict(config)
    output_config["image"] = args.yaml_image
    output_yaml.write_text(
        yaml.safe_dump(output_config, sort_keys=False), encoding="utf-8")
    metadata = {
        "schema": 1,
        "source_yaml": str(input_yaml),
        "source_image_sha256": sha256(input_image),
        "output_image_sha256": sha256(output_image),
        "clear_patch": {
            "center_m": [args.clear_x, args.clear_y],
            "radius_m": args.clear_radius,
            "evidence": "live 360-degree LiDAR minimum clearance >= 0.518 m",
        },
        "glass_keepout": {
            "center_m": [args.wall_x, args.wall_y],
            "yaw_rad": args.wall_yaw,
            "length_m": args.wall_length,
            "thickness_m": args.wall_thickness,
        },
    }
    output_meta.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
