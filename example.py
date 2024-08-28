import time
from pathlib import Path

import shapely
import numpy as np

import seamless_seg

from PIL import Image

def test_overlap_trim():
    geom_a = shapely.box(0, 0, 100, 100)
    geom_b = shapely.box(0, 60, 100, 160)

    out_geom_a, _, _, _, _ = seamless_seg.overlap_weights(geom_a, [geom_b])
    out_geom_b, _, _, _, _ = seamless_seg.overlap_weights(geom_b, [geom_a])

    print(shapely.area(shapely.intersection(out_geom_a, out_geom_b)))
    # is 40*100 = 4000

    out_geom_a, _, _, _, _ = seamless_seg.overlap_weights(geom_a, [geom_b], (0, 0, None, -20))
    out_geom_b, _, _, _, _ = seamless_seg.overlap_weights(geom_b, [geom_a], (0, 20, None, None))

    print(shapely.area(shapely.intersection(out_geom_a, out_geom_b)))


def show_overlap_weights_regular():
    np.random.seed(123459)
    # Creating test data
    # Regular grid
    boxes = [
        shapely.box(-50, -150, 250, 350), #tl
        shapely.box(450, -150, 750, 350), #bl
        shapely.box(-50, 750, 250, 1250), #tr
        shapely.box(450, 750, 750, 1250), #br
        shapely.box(200, -150, 500, 350),  # l
        shapely.box(-50, 300, 250, 800),  # t
        shapely.box(200, 750, 500, 1250),  # r
        shapely.box(450, 300, 750, 800),  # b
    ]
    N = len(boxes)+1
    data = np.ones((N, 300, 500, 3), dtype=np.uint8)
    for i in range(N):
        data[i] *= np.random.randint(0, 255, (3,), dtype=np.uint8)
    central = shapely.box(200, 300, 500, 800)

    # Finding overlap weights for this geom
    _, central_weights, _, nearby_weights, slice_pairs = seamless_seg.overlap_weights(central, boxes)

    # Using these weights on real data
    out = central_weights[..., None] * data[-1]
    for i, (weights, (central_slices, nearby_slices)) in enumerate(zip(nearby_weights, slice_pairs)):
        out[central_slices] += data[i][nearby_slices] * weights[central_slices][..., None]

    # Printing output for inspection
    Image.fromarray(out.astype(np.uint8)).save('test_overlap_weights_regular.png')

def show_overlap_weights_irregular():
    np.random.seed(123459)
    # Creating test data
    # Irregular
    boxes = [
        shapely.box(150, 150, 450, 650),
        shapely.box(300, 200, 600, 700),
        shapely.box(100, 500, 400, 1000),
        shapely.box(450, 650, 750, 1150),
    ]
    N = len(boxes)+1
    data = np.ones((N, 300, 500, 3), dtype=np.uint8)
    for i in range(N):
        data[i] *= np.random.randint(0, 255, (3,), dtype=np.uint8)
    central = shapely.box(200, 300, 500, 800)

    # Finding overlap weights for this geom
    _, central_weights, _, nearby_weights, slice_pairs = seamless_seg.overlap_weights(central, boxes)

    # Using these weights on real data
    out = central_weights[..., None] * data[-1]
    for i, (weights, (central_slices, nearby_slices)) in enumerate(zip(nearby_weights, slice_pairs)):
        out[central_slices] += data[i][nearby_slices] * weights[central_slices][..., None]

    # Printing output for inspection
    Image.fromarray(out.astype(np.uint8)).save('test_overlap_weights_irregular.png')

def minimal_random_colour_grid(image_size, tile_size, overlap):
    plan, grid = seamless_seg.plan_run_grid(image_size, tile_size, overlap, area)

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

def minimal_random_colour_grid_batched(image_size, tile_size, overlap):
    plan, grid = seamless_seg.plan_run_grid(image_size, tile_size, overlap, area)

    out_img = np.zeros((*image_size, 3))
    def _get_tiles_batched(indices, geoms):
        base = np.ones((*tile_size, 3), dtype=np.uint8)
        rand_rgb = np.random.randint(20, 255, (1, 1, 3), dtype=np.uint8)
        return base * rand_rgb

    def _write_tile(index, out_geom, out_tile):
        y_slc, x_slc = seamless_seg.shape_to_slices(out_geom)
        out_img[y_slc, x_slc] = out_tile

    batch_size = 8
    max_tiles = 1024
    seamless_seg.run_plan(plan, batch_size, max_tiles, _write_tile, get_tiles_batched=_get_tiles_batched)

def random_colour_grid(
    image_size,
    tile_size,
    overlap,
    area=None,
    actually_run=False,
    visualise_cache=False,
    fname='out_img.png'
):
    np.random.seed(123459)
    draw_area = area is not None

    print(f'For an image sized {image_size}, with tile {tile_size} and overlap {overlap}')

    start = time.perf_counter()
    plan, grid = seamless_seg.plan_run_grid(image_size, tile_size, overlap, area)
    end = time.perf_counter()
    print(f'Planning takes: {end-start:4.2f}s')

    max_loaded, load_actions, write_actions = seamless_seg.analyse_plan(plan)
    print(f'Plan holds a maximum of {max_loaded} tiles in memory at once.')
    print(f'Plan loads {load_actions} tiles and writes {write_actions} tiles.')
    print(f'That is, plan holds {max_loaded/load_actions:4.1%} of tiles in memory')

    if not actually_run:
        print()
        return

    out_img = np.zeros((*image_size, 3))

    vis_folder = Path('vis')
    vis_folder.mkdir(exist_ok=True)
    if visualise_cache:
        visualisation = np.zeros((*grid.shape[:2], 3), dtype=np.uint8)
        vis_cache_folder = Path('vis/cache')
        vis_cache_folder.mkdir(exist_ok=True)
    def _get_tiles_batched(indices, geoms):
        base = np.ones((len(geoms), *tile_size, 3), dtype=np.uint8)
        rand_rgb = np.random.randint(20, 255, (len(geoms), 1, 1, 3), dtype=np.uint8)
        if visualise_cache:
            for vy, vx in indices:
                visualisation[vy, vx, 0] = 255
        return base * rand_rgb

    def _get_tile(index, geom):
        base = np.ones((*tile_size, 3), dtype=np.uint8)
        rand_rgb = np.random.randint(20, 255, (1, 1, 3), dtype=np.uint8)
        if visualise_cache:
            vy, vx = index
            visualisation[vy, vx, 0] = 255
        return base * rand_rgb

    def _on_unload(index):
        if visualise_cache:
            vy, vx = index
            visualisation[vy, vx, 0] = 0

    def _write_tile(index, out_geom, out_tile):
        if visualise_cache:
            vy, vx = index
            visualisation[vy, vx, 1] = 255
        y_slc, x_slc = seamless_seg.shape_to_slices(out_geom)
        out_img[y_slc, x_slc] = out_tile

    def _on_step(n):
        if visualise_cache:
            Image.fromarray(visualisation).save(vis_cache_folder / f'cache_{n:04d}.png')

    start = time.perf_counter()
    fncs = {
        'write_tile': _write_tile,
        'on_unload': _on_unload,
        'on_step': _on_step,
    }
    if visualise_cache:
        fncs['get_tile'] = _get_tile
    else:
        fncs['get_tiles_batched'] = _get_tiles_batched
    seamless_seg.run_plan(plan, 8, 1024, **fncs)
    end = time.perf_counter()
    print(f'Running plan takes: {end-start:4.2f}s')
    print()

    if draw_area:
        coords = shapely.get_coordinates(area)
        rr, cc = skimage.draw.polygon(coords[:, 0], coords[:, 1])
        out_img[rr, cc] = 255
    Image.fromarray(out_img.astype(np.uint8)).save(vis_folder / fname)

def main():

    # area = shapely.Polygon([
    #     [210, 100], [0, 400], [400, 450], [550, 375], [250, 275], [450, 150]
    # ])
    # area = shapely.box(0, 0, 20000, 20000)
    area = None

    random_colour_grid((48, 64), (5,5), (2,2), actually_run=True, visualise_cache=True, fname='small_grid.png')
    random_colour_grid((256, 256), (58, 84), (6, 12), actually_run=True, fname='mid_grid.png')
    random_colour_grid((20000,20000), (128,256), (48,64), actually_run=False)
    # random_colour_grid((100000,100000), (256,256), (64,64), actually_run=False)

if __name__ == '__main__':
    main()
