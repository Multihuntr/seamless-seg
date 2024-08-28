# Seamless segmentation (seamless_seg)

This is an algorithm that removes tiling artifacts created from stitching together adjacent tiles in large segmentation tasks. This is useful for any segmentation task with very large images. For example, medical and satellite images.

TODO: Real example images

* :white_check_mark: Optimal! No more tiling artifacts. Guaranteed seamless segmentation for any segmentation model.
* :floppy_disk: Efficient! Needs <1% of image in memory at once for large images (40000x40000 and above).
* :purple_circle: Decoupled! Minimal dependencies. Does not prescribe any IO libraries.
* :zap: Fast! Approx 0.25ms of overhead per tile.

## Installation

Copy `seamless_seg.py` into your project.

Dependencies: `shapely` and `scipy`.

TODO: pip?

## Getting started

Here is a minimum working example of how to use `seamless_seg` on dummy data. In a real example, you would use the logits of a segmentation algorithm.

```python
import numpy as np

import seamless_seg

def minimal_random_colour_grid(image_size, tile_size, overlap):
    plan, grid = seamless_seg.plan_run_grid(image_size, tile_size, overlap)

    out_img = np.zeros((*image_size, 3))
    def _get_tile(index, geom):
        # Dummy data; a tile consisting of a randomly selected colour.
        base = np.ones((*tile_size, 3), dtype=np.uint8)
        rand_rgb = np.random.randint(20, 255, (1, 1, 3), dtype=np.uint8)
        return base * rand_rgb

    def _write_tile(index, out_geom, out_tile):
        y_slc, x_slc = seamless_seg.shape_to_slices(out_geom)
        out_img[y_slc, x_slc] = out_tile

    batch_size = 8
    max_tiles = 1024
    seamless_seg.run_plan(plan, batch_size, max_tiles, _write_tile, get_tile=_get_tile)
```

## Fixing tiling artifacts

### Where do tiling artifacts come from?

Tiling artifacts are a result of hard boundaries between adajcent tiles. The most naive approach is select tiles with no overlap, and just let the model predict whatever it wills. At the boundary of those tiles, models will often make significantly different predictions. This results in sharp lines in your output segmentation.

TODO: Real example

This is not a model failure per se. The problem is just that the model is using a different context for pixels on one side of a boundary to the other side of that boundary. If it were given a full context around each object, it may still segment it correctly.

### Overlapping tiles

Typical solutions to this will always somehow use overlapping tiles. A slightly less naive approach commonly taken is to overlap tiles, and only keep a smaller window of the outputs. That solution *reduces* tiling artifacts, but does not remove them entirely. So long as there are hard boundaries between the tiles, tiling artifacts will appear.

TODO: Diagram explaining output crop margin.

TODO: Example of reducing tiling artifacts.

In the extreme case, we could evaluate a tile centered on every single pixel independently and only trust that central pixel. But this involves lots of redundant calculation. We need a better solution.

### Trusting model outputs

Pixels at the edge of each tile have a truncated context because of their position within the tile. This lack of context degrades model performance at the edges of each tile.

TODO: Diagram showing truncated context, and

In some sense, this means that we should inherently trust the pixels at the edges less than those at the centre. So, we define a map of trustworthiness of each pixel. This is simply a measure of the distance of that pixel to the centre of the tile.

TODO: image of circle of trust

We can use the trust values to determine how much we should use from each overlapping tile. This gives us a distinct weighted sum at each pixel. Using a weighted sum based on distance produces a smooth transition across tiles. Pixels at the centre of an output tile come entirely from the model output for that tile, and pixels halfway between two tiles come 50% from each, etc.

TODO: diagram showing spatial relationship between these tiles.

![Eight-way smoothing with 50% overlap](img/8_way_smoothing.png)
![Eight-way smoothing with approx 10% overlap](img/8_way_small_overlap.png)

These weights can be obtained by calling `seamless_seg.overlap_weights`, but you probably don't want to do that. It is recommended to use `seamless_seg.plan_run_grid` instead.

## Tiling plan - optimising read/writes

Utilising the overlap weights described in the previous section is not inherently linked to using a regular grid of overlapping tiles, but usually that's what we want. To make the typical use case easier, `seamless_seg` includes `seamless_seg.plan_run_grid` to create a tiling plan in the shape of a grid.

![Grid of colour blocks, smoothly transitioning between each](img/mid_grid.png)

All you need to do is provide an image size, tile size and overlap amount (in pixels). The plan is created entirely within geometric pixel space. That is, the plan is made before any real data is read from disk. This allows you to inspect the plan and optimise reading from disk (e.g. batching, threaded/async).

Finally, `seamless_seg.run_plan` is provided to actually run the plan. This automatically handles batching, threading of reading from disk and holding a minimum number of tiles in memory at once.

### Memory

Operating on large images often means that entire input images and output segmentation maps cannot be kept in RAM. With this in mind, the tiling plan includes explicit load/unload instructions. Following this plan ensures that tiles are never requested more than once **and** that the minimum number of tiles are kept in memory. For some perspective, given a (40000, 40000) image with a tile size of (256, 256) and an overlap of (64, 64), there will be at most 1% of the image held in memory at once.


