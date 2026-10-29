# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import torch
from typing import TypedDict

class SampleRecord(TypedDict):
    unif2: torch.Tensor # or mi.Point2f, mi.Float
    samples: torch.Tensor # or mi.Point2f, mi.Float
    pdf: torch.Tensor # or mi.Float