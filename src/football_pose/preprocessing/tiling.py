from __future__ import annotations

import math

import numpy as np

from football_pose.domain import FramePacket
from football_pose.preprocessing.base import ProcessorContext


class TileProcessor:
    """Split each packet into a deterministic row-major grid of overlapping tiles."""

    name = "tile"

    def __init__(
        self,
        *,
        rows: int = 2,
        columns: int = 2,
        overlap_ratio: float = 0.0,
    ) -> None:
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError("tile rows must be a positive integer")
        if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
            raise ValueError("tile columns must be a positive integer")
        if not 0.0 <= overlap_ratio < 0.5:
            raise ValueError("tile overlap_ratio must be in [0.0, 0.5)")
        self.rows = rows
        self.columns = columns
        self.overlap_ratio = float(overlap_ratio)

    def process(
        self, packet: FramePacket, context: ProcessorContext
    ) -> list[FramePacket]:
        del context
        if self.rows > packet.height or self.columns > packet.width:
            raise ValueError("tile grid cannot have more rows/columns than image pixels")

        outputs: list[FramePacket] = []
        parent_id = packet.crop_id or f"{packet.frame_index:09d}"
        for row in range(self.rows):
            nominal_top = math.floor(row * packet.height / self.rows)
            nominal_bottom = math.floor((row + 1) * packet.height / self.rows)
            nominal_height = nominal_bottom - nominal_top
            # Split the requested overlap evenly across the shared boundary so
            # overlap_ratio describes the total overlap between neighboring cells.
            pad_y = round(nominal_height * self.overlap_ratio / 2)
            top = max(0, nominal_top - (pad_y if row > 0 else 0))
            bottom = min(
                packet.height,
                nominal_bottom + (pad_y if row < self.rows - 1 else 0),
            )

            for column in range(self.columns):
                nominal_left = math.floor(column * packet.width / self.columns)
                nominal_right = math.floor((column + 1) * packet.width / self.columns)
                nominal_width = nominal_right - nominal_left
                pad_x = round(nominal_width * self.overlap_ratio / 2)
                left = max(0, nominal_left - (pad_x if column > 0 else 0))
                right = min(
                    packet.width,
                    nominal_right + (pad_x if column < self.columns - 1 else 0),
                )

                tile_id = f"{parent_id}-tile-r{row:02d}-c{column:02d}"
                current_to_previous = np.array(
                    [[1, 0, left], [0, 1, top], [0, 0, 1]], dtype=np.float64
                )
                source_points = packet.points_to_source(
                    np.array([[left, top], [right, bottom]], dtype=np.float64)
                )
                source_bbox = tuple(float(value) for value in source_points.reshape(-1))
                metadata = {
                    **packet.metadata,
                    "tile": {
                        "row": row,
                        "column": column,
                        "rows": self.rows,
                        "columns": self.columns,
                        "overlap_ratio": self.overlap_ratio,
                        "input_bounds": [left, top, right, bottom],
                        "source_bounds": list(source_bbox),
                    },
                }
                outputs.append(
                    packet.derived(
                        packet.image[top:bottom, left:right].copy(),
                        current_to_previous,
                        source_id=tile_id,
                        crop_id=tile_id,
                        metadata=metadata,
                    )
                )
        return outputs
