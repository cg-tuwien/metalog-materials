# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import mitsuba as mi
from dataclasses import dataclass
from typing import List


@dataclass
class RotationConfig:
    transform_key: str
    rot_angles: List[float]
    up_axis: List[float]
    mesh_center: mi.Vector3f