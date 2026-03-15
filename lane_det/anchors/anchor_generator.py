import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AnchorSet:
    anchor_xs: np.ndarray
    valid_mask: np.ndarray
    x_bottom: np.ndarray
    angles: np.ndarray
    y_samples: np.ndarray


class AnchorGenerator:
    """Generate (x_bottom, angle) anchors at fixed y samples."""

    def __init__(
        self,
        img_size,
        num_y,
        x_positions,
        angles,
        side_y_step: int = 20,
        side_y_start_ratio: float = 0.5,
        side_angle_min: float = 10.0,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        y_samples=None,
    ):
        self.img_w = int(img_size[0])
        self.img_h = int(img_size[1])
        self.y_samples = self._parse_y_samples(y_samples, num_y)
        self.num_y = int(self.y_samples.shape[0])
        self.x_positions = self._parse_x_positions(x_positions)
        self.angles = np.asarray(angles, dtype=np.float32)
        self.side_y_step = max(1, int(side_y_step))
        self.side_y_start_ratio = float(side_y_start_ratio)
        self.side_angle_min = float(side_angle_min)
        self.cache_dir = cache_dir
        self.use_cache = use_cache and cache_dir is not None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], cache_dir: Optional[str] = "outputs/cache/anchors"):
        acfg = cfg["anchor"]
        dcfg = cfg["dataset"]
        return cls(
            img_size=dcfg["img_size"],
            num_y=acfg.get("num_y", dcfg["y_samples"]),
            x_positions=acfg["x_positions"],
            angles=acfg["angles"],
            side_y_step=acfg.get("side_y_step", 20),
            side_y_start_ratio=acfg.get("side_y_start_ratio", 0.5),
            side_angle_min=acfg.get("side_angle_min", 10.0),
            cache_dir=cache_dir,
            use_cache=True,
            y_samples=None,
        )

    def _parse_y_samples(self, y_samples, num_y):
        if y_samples is not None:
            arr = np.asarray(y_samples, dtype=np.float32)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError("y_samples must be a non-empty 1D list when provided")
            return np.clip(arr, 0, self.img_h - 1)
        return np.linspace(0, self.img_h - 1, int(num_y), dtype=np.float32)

    def _parse_x_positions(self, x_positions):
        if isinstance(x_positions, dict):
            start = float(x_positions.get("start", 0))
            end = float(x_positions.get("end", self.img_w - 1))
            step = float(x_positions.get("step", 20))
            return np.arange(start, end + step, step, dtype=np.float32)

        if isinstance(x_positions, int):
            # If int, generate uniform points between 0 and W-1
            if x_positions <= 1:
                return np.array([0.5 * (self.img_w - 1)], dtype=np.float32)
            return np.linspace(0, self.img_w - 1, int(x_positions), dtype=np.float32)
        
        arr = np.asarray(x_positions, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("anchor.x_positions must be int, dict(start,end,step), or 1D non-empty list")
        # Do NOT clip here if we want to allow out-of-image anchors
        return arr

    def _cache_path(self):
        if not self.use_cache:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Convert x_positions to list for JSON serialization
        # If it was a dict in config, it's already converted to ndarray by _parse_x_positions
        x_pos_list = self.x_positions.tolist()
        
        key_obj = {
            "img_w": self.img_w,
            "img_h": self.img_h,
            "num_y": self.num_y,
            "x_positions": x_pos_list,
            "angles": self.angles.tolist(),
            "y_samples": self.y_samples.tolist(),
            "side_y_step": self.side_y_step,
            "side_y_start_ratio": self.side_y_start_ratio,
            "side_angle_min": self.side_angle_min,
        }
        key = json.dumps(key_obj, sort_keys=True)
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"anchors_{digest}.npz")

    def _build(self):
        y_samples = self.y_samples.astype(np.float32)
        y_bottom = float(self.img_h - 1)
        dy = y_bottom - y_samples

        # 1. Bottom Anchors (Original)
        x_grid, a_grid = np.meshgrid(self.x_positions, self.angles, indexing="xy")
        x_bottom = x_grid.reshape(-1).astype(np.float32)
        angles = a_grid.reshape(-1).astype(np.float32)
        slopes = np.tan(np.deg2rad(angles)).astype(np.float32)

        anchor_xs = x_bottom[:, None] + slopes[:, None] * dy[None, :]
        
        # 2. Side Anchors (New)
        # Generate anchors starting from left/right edges at various heights
        # We only need angles pointing INWARDS.
        # Left edge (x=0): angles should be positive (leaning right) -> [20, 80]
        # Right edge (x=W): angles should be negative (leaning left) -> [-80, -20]
        
        side_anchor_xs_list = []
        side_angles_list = []
        side_x_bottom_list = [] # Virtual x_bottom for compatibility
        
        # Define y_starts for side anchors (e.g., every 40 pixels from top to bottom)
        # We focus on the middle-to-bottom part where lanes usually enter
        side_y_start = int(self.img_h * self.side_y_start_ratio)
        side_y_start = min(max(0, side_y_start), self.img_h - 1)
        side_y_starts = np.arange(side_y_start, self.img_h, self.side_y_step, dtype=np.float32)
        
        # Filter angles for sides
        # Left side: angles > 15
        left_angles = self.angles[self.angles >= self.side_angle_min]
        # Right side: angles < -15
        right_angles = self.angles[self.angles <= -self.side_angle_min]
        
        for y_start in side_y_starts:
            dy_start = y_bottom - y_start # distance from bottom to start y
            
            # --- Left Edge Anchors (x=0 at y_start) ---
            if len(left_angles) > 0:
                # x = x_start + slope * (y_start - y)
                #   = 0 + slope * (y_start - y)
                #   = slope * (y_start - y_bottom + y_bottom - y)
                #   = slope * (-dy_start + dy)
                l_slopes = np.tan(np.deg2rad(left_angles)).astype(np.float32)
                # [N_angles, N_y]
                l_xs = l_slopes[:, None] * (dy[None, :] - dy_start)
                
                # Virtual x_bottom (x at y_bottom, i.e. dy=0) -> x = slope * (-dy_start)
                l_x_bottom = l_slopes * (-dy_start)
                
                side_anchor_xs_list.append(l_xs)
                side_angles_list.append(left_angles)
                side_x_bottom_list.append(l_x_bottom)

            # --- Right Edge Anchors (x=W-1 at y_start) ---
            if len(right_angles) > 0:
                # x = (W-1) + slope * (dy - dy_start)
                r_slopes = np.tan(np.deg2rad(right_angles)).astype(np.float32)
                r_xs = (self.img_w - 1) + r_slopes[:, None] * (dy[None, :] - dy_start)
                
                # Virtual x_bottom
                r_x_bottom = (self.img_w - 1) + r_slopes * (-dy_start)
                
                side_anchor_xs_list.append(r_xs)
                side_angles_list.append(right_angles)
                side_x_bottom_list.append(r_x_bottom)
        
        # Concatenate everything
        if len(side_anchor_xs_list) > 0:
            side_xs = np.concatenate(side_anchor_xs_list, axis=0)
            side_ang = np.concatenate(side_angles_list, axis=0)
            side_xb = np.concatenate(side_x_bottom_list, axis=0)
            
            anchor_xs = np.concatenate([anchor_xs, side_xs], axis=0)
            angles = np.concatenate([angles, side_ang], axis=0)
            x_bottom = np.concatenate([x_bottom, side_xb], axis=0)

        valid_mask = ((anchor_xs >= 0) & (anchor_xs <= (self.img_w - 1))).astype(np.uint8)

        return AnchorSet(
            anchor_xs=anchor_xs.astype(np.float32),
            valid_mask=valid_mask,
            x_bottom=x_bottom,
            angles=angles,
            y_samples=y_samples,
        )

    def generate(self, force_rebuild: bool = False) -> AnchorSet:
        cache_path = self._cache_path()
        if self.use_cache and (not force_rebuild) and cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            return AnchorSet(
                anchor_xs=data["anchor_xs"].astype(np.float32),
                valid_mask=data["valid_mask"].astype(np.uint8),
                x_bottom=data["x_bottom"].astype(np.float32),
                angles=data["angles"].astype(np.float32),
                y_samples=data["y_samples"].astype(np.float32),
            )

        anchors = self._build()
        if self.use_cache and cache_path:
            np.savez_compressed(
                cache_path,
                anchor_xs=anchors.anchor_xs,
                valid_mask=anchors.valid_mask,
                x_bottom=anchors.x_bottom,
                angles=anchors.angles,
                y_samples=anchors.y_samples,
            )
        return anchors
