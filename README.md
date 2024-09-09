# `seamless_seg`: Seamless tiled segmentation postprocessing tools for large images

Typical strategies for segmenting large images involve tiling. Unfortunately this can cause visible, obviously incorrect seams between tiles. This repo provides postprocessing functions which gracefully remove *all* such tiling artifacts for any segmentation task.

<!-- TODO: Real example images -->

* :white_check_mark: Optimal! No more tiling artifacts. Guaranteed seamless segmentation for any segmentation model.
* :floppy_disk: Efficient! Needs <1% of image in memory at once for large images (40000x40000 and above).
* :purple_circle: Decoupled! Minimal dependencies. Does not prescribe any IO libraries.
* :zap: Fast! Approx 0.25ms of overhead per tile.

## Installation

Copy `seamless_seg.py` into your project.

Dependencies: `shapely` and `scipy`.

<!-- TODO: pip? -->

## Getting started

Here is a minimum working example of how to use `seamless_seg` on dummy data. In a real example, you would use the logits of a segmentation algorithm.

```python
import numpy as np

import seamless_seg

def minimal_random_colour_grid(image_size, tile_size, overlap):
    def _input_generator(plan):
        for index, geom in seamless_seg.get_plan_input_geoms(plan):
            # Creating fake data; in real use cases, should yield model logits 
            # obtained by running your model on image data within geom 
            # Note: geom is a shapely.Geometry
            tile = np.ones((*tile_size, 3), dtype=np.uint8)
            tile *= np.random.randint(20, 255, (3,), dtype=np.uint8)
            yield tile

    # Iterate over output tiles; in this case, write directly to a np array
    # But in real use cases, you can write the tile to disk (e.g. rasterio/tifffile)
    plan, grid = seamless_seg.plan_regular_grid(image_size, tile_size, overlap)
    in_tiles = _input_generator(plan)
    out_img = np.zeros((*image_size, 3))
    for index, out_geom, out_tile in seamless_seg.run_plan(plan, in_tiles):
        y_slc, x_slc = seamless_seg.shape_to_slices(out_geom)
        out_img[y_slc, x_slc] = out_tile
    # All done!
```

Here is an unbatched example using a pytorch model on a geotiff with rasterio

```python
import rasterio
import torch

import seamless_seg

model = # get model from somewhere; assuming pytorch model
in_fpath = # get geotiff image from somewhere
out_fpath = # where to save segmentation
profile = # geotiff output profile

in_tif = rasterio.open(in_fpath)
out_tif = rasterio.open(out_fpath, 'w', **profile)

def _input_generator(plan):
    for index, in_geom in seamless_seg.get_plan_input_geoms(plan):
        # Read image data
        in_slices = seamless_seg.shape_to_slices(in_geom)
        img = in_tif.read(window=in_slices)

        # Push image data through model (don't forget batch dimension)
        img_th = torch.as_tensor(img[None]).to(model.device)
        out_th = model(img_th)
        out = out_th[0].detach().cpu().numpy()

        # Yield model outputs in HWC
        yield out.transpose((1, 2, 0))

# Run plan on model outputs shown above
plan, grid = seamless_seg.plan_regular_grid(in_tif.shape, (224, 224), (32, 32))
in_tiles = _input_generator(plan)
out_tiles = seamless_seg.run_plan(plan, in_tiles)
for index, out_geom, out_tile in out_tiles:
    # Convert logits to segmentation mask
    seg = out_tile.argmax(axis=-1)

    # Write segmentation mask to disk
    slices = seamless_seg.shape_to_slices(out_geom)
    out_tif.write(seg, window=slices)

in_tif.close()
out_tif.close()

```

To push model evaluation into a thread and to batch properly, you can replace the `_input_generator` above with the following:

```python
batch_size = 16
def _run_tiles(indexs, geoms):
    """A function which takes a batch of geoms and returns model outputs for those geoms"""
    # Load all images for batch
    imgs = []
    for in_geom in geoms:
        in_slices = seamless_seg.shape_to_slices(in_geom)
        imgs.append(in_tif.read(window=in_slices))

    # Push batch through model
    img_th = torch.as_tensor(np.stack(img)).to(model.device)
    out_th = model(img_th)
    out = out_th.detach().cpu().numpy()

    # model output is in BCHW, yield model outputs in BHWC
    return out.transpose((0, 2, 3, 1))

def _input_generator(plan):
    geoms = seamless_seg.get_plan_input_geoms(plan)
    return seamless_seg.threaded_batched_tile_get(geoms, batch_size, _run_tiles, batch_size*3)
```

Additionally, there are more advanced use cases supported by `seamless_seg`:
* Only running on an area within the overall image
  * Pass `area=<shapely.Geometry>` to `seamless_seg.plan_regular_grid`.
* Custom, semi-regular grids:
  * Create your grid, then use `seamless_seg.plan_from_grid` instead of `seamless_seg.plan_regular_grid`.
* Fixed RAM limits:
  * Pass `max_tiles=<int>` and `disk_cache_dir=<Path>` to `seamless_seg.run_plan`.
  * This will cache model outputs beyond `max_tiles` to disk, instead of recomputing.
* Batching utility functions:
  * See `seamless_seg.batched_tile_get` and `seamless_seg.threaded_batched_tile_get`.

## Explanation - Fixing tiling artifacts

### Where do tiling artifacts come from?

Tiling artifacts are a result of hard boundaries between adajcent tiles. The most naive approach is to select tiles with no overlap, and just let the model predict whatever it wills. At the boundary of those tiles, models will often make significantly different predictions. This results in sharp lines in your output segmentation.

<!-- TODO: Real example -->

This is not a model failure per se. The problem is just that the model is using a different context for pixels on one side of a boundary to the other side of that boundary. If it were given a full context around each object, it may still segment it correctly.

### Overlapping tiles

Typical solutions to this will always somehow use overlapping tiles. A slightly less naive approach commonly taken is to overlap tiles, and only keep a smaller window of the outputs. That solution *reduces* tiling artifacts, but does not remove them entirely. So long as there are hard boundaries between the tiles, tiling artifacts will appear.

<!-- TODO: Diagram explaining output crop margin. -->

<!-- TODO: Example of reducing tiling artifacts. -->

In the extreme case, we could evaluate a tile centered on every single pixel independently and only trust that central pixel. But this involves lots of redundant calculation. We need a better solution.

### Trusting model outputs

Pixels at the edge of each tile have a truncated context because of their position within the tile. This lack of context degrades model performance at the edges of each tile.

<!-- TODO: Diagram showing truncated context -->

In some sense, this means that we should inherently trust the pixels at the edges less than those at the centre. So, we define a map of trustworthiness of each pixel. This is simply a measure of the distance of that pixel to the centre of the tile.

<!-- TODO: image of circle of trust -->

We can use the trust values to determine how much we should use from each overlapping tile. This gives us a distinct weighted sum at each pixel. Using a weighted sum based on distance produces a smooth transition across tiles. Pixels at the centre of an output tile come entirely from the model output for that tile, and pixels halfway between two tiles come 50% from each, etc.

<!-- TODO: diagram showing spatial relationship between these tiles. -->

![Eight-way smoothing with 50% overlap](img/8_way_smoothing.png)
![Eight-way smoothing with approx 10% overlap](img/8_way_small_overlap.png)

These weights can be obtained by calling `seamless_seg.overlap_weights`, but you probably don't want to do that. It is recommended to use `seamless_seg.plan_regular_grid` instead.

## Tiling plan - optimising read/writes

Utilising the overlap weights described in the previous section is not inherently linked to using a regular grid of overlapping tiles, but usually that's what we want. To make the typical use case easier, `seamless_seg` includes `seamless_seg.plan_regular_grid` to create a tiling plan in the shape of a grid.

![Grid of colour blocks, smoothly transitioning between each](img/mid_grid.png)

All you need to do is provide an image size, tile size and overlap amount (in pixels). The plan is created entirely within geometric pixel space. That is, the plan is made before any real data is read from disk. This allows you to inspect the plan and optimise reading from disk (e.g. batching, threaded/async).

Finally, `seamless_seg.run_plan` is provided to actually run the plan. To control memory usage, this can optionally be given a maximum number of tiles to keep in RAM at once.

### Memory

Often large segmentation tasks have images that are too large to fit into RAM. So, the tiling plan includes explicit load/unload instructions. Following this plan ensures that tiles are never requested more than once **and** that the minimum number of tiles are kept in memory. For some perspective, given a (40000, 40000) image with a tile size of (256, 256) and an overlap of (64, 64), there will be at most 1.0% of the image held in memory at once.

If even this is too large, you can use the `max_tiles` and `disk_cache_dir` arguments to hold as few tiles in memory as you need. Tiles beyond this limit will be cached to disk.


