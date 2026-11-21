"""Fixed numbers for the whole project. Do not change them here or override them in a config.

The trained models, the printed tiles and the web scanner all depend on
these values agreeing.
"""

from __future__ import annotations

# Number of bits the embedder writes and the reader returns.
# Layout of the bits: /docs/message-format.md.
MESSAGE_BITS = 64

# Side of one tile in pixels, as the embedder sees it and as the reader
# receives a flattened patch.
TILE_PX = 384

# Camera pixels covered by one printed tile at demo distance. Measured value,
# to be supplied. The simulation's loss-of-resolution step refuses to run
# while this is None.
CAMERA_PX_PER_TILE: int | None = None
