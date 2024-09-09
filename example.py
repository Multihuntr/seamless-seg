import time
from pathlib import Path

import shapely
import skimage
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
    central_geom = shapely.box(200, 300, 500, 800)
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
    central_tile = np.ones((300, 500, 3), dtype=np.uint8)
    central_tile *= np.random.randint(20, 255, (3,), dtype=np.uint8)
    N = len(boxes)
    nearby_tiles = []
    for i in range(N):
        nearby_tiles.append(np.ones((300, 500, 3), dtype=np.uint8))
        nearby_tiles[i] *= np.random.randint(20, 255, (3,), dtype=np.uint8)

    # Run seamless_seg to calculate new central tile
    weights = seamless_seg.overlap_weights(central_geom, boxes)
    out = seamless_seg.apply_weights(central_tile, nearby_tiles, weights)

    # Printing output for inspection
    Image.fromarray(out.astype(np.uint8)).save('test_overlap_weights_regular.png')

def show_overlap_weights_irregular():
    np.random.seed(123459)
    # Creating test data
    # Irregular
    central_geom = shapely.box(200, 300, 500, 800)
    boxes = [
        shapely.box(150, 150, 450, 650),
        shapely.box(300, 200, 600, 700),
        shapely.box(100, 500, 400, 1000),
        shapely.box(450, 650, 750, 1150),
    ]
    central_tile = np.ones((300, 500, 3), dtype=np.uint8)
    central_tile *= np.random.randint(20, 255, (3,), dtype=np.uint8)
    N = len(boxes)
    nearby_tiles = []
    for i in range(N):
        nearby_tiles.append(np.ones((300, 500, 3), dtype=np.uint8))
        nearby_tiles[i] *= np.random.randint(20, 255, (3,), dtype=np.uint8)

    # Run seamless_seg to calculate new central tile
    weights = seamless_seg.overlap_weights(central_geom, boxes)
    out = seamless_seg.apply_weights(central_tile, nearby_tiles, weights)

    # Printing output for inspection
    Image.fromarray(out.astype(np.uint8)).save('test_overlap_weights_irregular.png')


def _random_tile_gen(shape):
    # Generates infinite fake tiles, where each tile is a single, randomly selected colour
    while True:
        tile = np.ones(shape, dtype=np.uint8)
        tile *= np.random.randint(20, 255, (shape[-1],), dtype=np.uint8)
        yield tile


def minimal_random_colour_grid(image_size, tile_size, overlap):
    def _input_generator(plan):
        shape = (*tile_size, 3)
        for index, geom in seamless_seg.get_plan_input_geoms(plan):
            # Creating fake data; in real use cases, should yield image data from within geom
            # Note: geom is a shapely.Geometry
            tile = np.ones(shape, dtype=np.uint8)
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


def random_colour_grid(
    image_size,
    tile_size,
    overlap,
    area=None,
    actually_run=False,
    fname='out_img.png'
):
    np.random.seed(123459)
    draw_area = area is not None

    print(f'For an image sized {image_size}, with tile {tile_size} and overlap {overlap}')

    start = time.perf_counter()
    plan, grid = seamless_seg.plan_regular_grid(image_size, tile_size, overlap, area)
    end = time.perf_counter()
    print(f'Planning takes: {end-start:4.2f}s')

    max_loaded, load_actions, write_actions = seamless_seg.analyse_plan(plan)
    print(f'Plan holds a maximum of {max_loaded} tiles in memory at once.')
    print(f'Plan loads {load_actions} tiles and writes {write_actions} tiles.')
    print(f'That is, plan holds {max_loaded/load_actions:4.1%} of tiles in memory')

    if not actually_run:
        print()
        return

    start = time.perf_counter()

    # Create fake random data
    in_tile_gen = _random_tile_gen((*tile_size, 3))

    # Run plan
    out_img = np.zeros((*image_size, 3))
    for index, out_geom, out_tile in seamless_seg.run_plan(plan, in_tile_gen):
        y_slc, x_slc = seamless_seg.shape_to_slices(out_geom)
        out_img[y_slc, x_slc] = out_tile

    end = time.perf_counter()
    print(f'Running plan takes: {end-start:4.2f}s')
    print()

    # Save out_img to disk
    if draw_area:
        coords = shapely.get_coordinates(area)
        rr, cc = skimage.draw.polygon(coords[:, 0], coords[:, 1])
        out_img[rr, cc] = 255
    vis_folder = Path('vis')
    vis_folder.mkdir(exist_ok=True)
    Image.fromarray(out_img.astype(np.uint8)).save(vis_folder / fname)

def test_batched_colour_grid(
    image_size,
    tile_size,
    overlap,
    fname='out_img.png'
):
    random_tiles = _random_tile_gen((*tile_size, 3))
    batch_size = 16
    def _get_tiles(indexs, geoms):
        return np.stack([next(random_tiles) for _ in geoms])

    def _input_generator(plan):
        geoms = seamless_seg.get_plan_input_geoms(plan)
        return seamless_seg.threaded_batched_tile_get(geoms, batch_size, _get_tiles, batch_size*3)

    # Iterate over output tiles; in this case, write directly to a np array
    # But in real use cases, you can write the tile to disk (e.g. rasterio/tifffile)
    plan, grid = seamless_seg.plan_regular_grid(image_size, tile_size, overlap)
    in_tiles = _input_generator(plan)
    out_img = np.zeros((*image_size, 3))
    for index, out_geom, out_tile in seamless_seg.run_plan(plan, in_tiles):
        y_slc, x_slc = seamless_seg.shape_to_slices(out_geom)
        out_img[y_slc, x_slc] = out_tile

    # Save out_img to disk
    vis_folder = Path('vis')
    vis_folder.mkdir(exist_ok=True)
    Image.fromarray(out_img.astype(np.uint8)).save(vis_folder / fname)


def random_colour_grid_visualise_cache(image_size, tile_size, overlap, do_print=False):
    plan, grid = seamless_seg.plan_regular_grid(image_size, tile_size, overlap)
    visualisation = np.zeros((*grid.shape[:2], 3), dtype=np.uint8)
    vis_cache_folder = Path('vis/cache-vis')
    vis_cache_folder.mkdir(exist_ok=True)
    i = 0

    def _on_load(index):
        nonlocal i
        if do_print:
            print(f'{i:04d}: loading {index}')
        visualisation[index[0], index[1], 0] = 255
        Image.fromarray(visualisation).save(vis_cache_folder / f'cache_{i:04d}.png')
        i += 1

    def _on_unload(index):
        nonlocal i
        if do_print:
            print(f'{i:04d}: unloading {index}')
        visualisation[index[0], index[1], 0] = 0
        Image.fromarray(visualisation).save(vis_cache_folder / f'cache_{i:04d}.png')
        i += 1

    def _on_disk_evict(index):
        nonlocal i
        if do_print:
            print(f'{i:04d}: evicting {index} to disk')
        visualisation[index[0], index[1], 1] = 255
        visualisation[index[0], index[1], 0] = 0
        Image.fromarray(visualisation).save(vis_cache_folder / f'cache_{i:04d}.png')
        i += 1

    def _on_disk_restore(index):
        nonlocal i
        if do_print:
            print(f'{i:04d}: restoring {index} from disk')
        visualisation[index[0], index[1], 1] = 0
        visualisation[index[0], index[1], 0] = 255
        Image.fromarray(visualisation).save(vis_cache_folder / f'cache_{i:04d}.png')
        i += 1

    def _on_step(n):
        # Image.fromarray(visualisation).save(vis_cache_folder / f'cache_{n:04d}.png')
        pass

    in_tile_gen = _random_tile_gen((*tile_size, 3))
    out_tiles = seamless_seg.run_plan(
        plan,
        in_tile_gen,
        10,
        disk_cache_dir=Path('vis/data-cache'),
        on_load=_on_load,
        on_unload=_on_unload,
        on_disk_evict=_on_disk_evict,
        on_disk_restore=_on_disk_restore,
        on_step=_on_step,
    )

    for index, out_geom, out_tile in out_tiles:
        if do_print:
            print(f'{i:04d}: writing {index}')
        visualisation[index[0], index[1], 2] = 255
        Image.fromarray(visualisation).save(vis_cache_folder / f'cache_{i:04d}.png')
        i += 1


def main():

    area = shapely.Polygon([
        [26, 14], [5, 20], [19, 28], [5, 44], [15, 55], [22, 40],
        [38, 55], [44, 37], [26, 28], [44, 19], [40, 4], [17, 6]
    ])

    random_colour_grid((48, 64), (5, 5), (2, 2), actually_run=True, fname='small_grid.png')
    random_colour_grid((128, 86), (7, 7), (2, 2), area=area, actually_run=True, fname='small_grid_w_area.png')
    random_colour_grid((256, 256), (58, 84), (6, 12), actually_run=True, fname='mid_grid.png')
    # random_colour_grid((40000, 40000), (256, 256), (64, 64), actually_run=False)
    # random_colour_grid_visualise_cache((48, 64), (11,11), (2, 2))
    test_batched_colour_grid((128, 86), (7, 7), (2, 2), fname='small_grid_batched.png')

if __name__ == '__main__':
    main()
