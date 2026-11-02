# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import mitsuba as mi
from dataclasses import dataclass
from typing import List


@dataclass
class RotationConfig:
    transform_key: str
    rot_angles: List[float]
    up_axis: List[float]
    mesh_center: mi.Vector3f