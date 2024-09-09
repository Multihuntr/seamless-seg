import collections
import dataclasses
import queue
import threading
from pathlib import Path
from typing import Sequence, Iterable, Generator

import numpy as np
import scipy
import shapely
import shapely.affinity

# Consistently arbitrarily ordered list of 8 directions to look for adjacent tiles
GRID_DIR = np.array([(j, i) for j in (-1, 0, 1) for i in (-1, 0, 1) if not (i==j==0)])

def shape_to_slices(shp: shapely.Geometry):
    ylo, xlo, yhi, xhi = shp.bounds
    ylo, xlo = round(ylo), round(xlo)
    yhi, xhi = round(yhi), round(xhi)
    return slice(ylo, yhi), slice(xlo, xhi)

def mk_circle_of_trust(h, w):
    trust_coords_T = np.array([(-1, h // 2, h), (-1, w // 2, w)])
    trust_values = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    interpolator = scipy.interpolate.RegularGridInterpolator(trust_coords_T, trust_values)

    eval_coords = tuple(np.indices((h, w)))
    return interpolator(eval_coords)

def get_trimmed_bounds(margin: tuple[int, int], dirs: Sequence[tuple[int, int]]):
    my, mx = margin
    ylo, xlo, yhi, xhi = 0, 0, None, None
    for (j, i) in dirs:
        if j == -1:
            ylo = my
        if j == 1:
            yhi = -my
        if i == -1:
            xlo = mx
        if i == 1:
            xhi = -mx
    return ylo, xlo, yhi, xhi

def trim_array(arr: np.ndarray, bounds: tuple[int,int,int,int]):
    ylo, xlo, yhi, xhi = bounds
    return arr[..., ylo:yhi, xlo:xhi]

def trim_box(shp: shapely.Geometry, bounds: tuple[int,int,int,int]):
    bylo, bxlo, byhi, bxhi = shp.bounds
    tylo, txlo, tyhi, txhi = bounds
    slices = (slice(tylo, tyhi), slice(txlo, txhi))
    tyhi = 0 if tyhi is None else tyhi
    txhi = 0 if txhi is None else txhi
    new_box = shapely.box(bylo+tylo, bxlo+txlo, byhi+tyhi, bxhi+txhi)
    return new_box, slices



def overlap_weights(
    central: shapely.Geometry,
    nearby: Sequence[shapely.Geometry],
    trim_bounds: tuple[int,int,int,int] = None,
) -> (
    shapely.Geometry,
    np.ndarray,
    tuple[slice, slice],
    np.ndarray,
    list[tuple[tuple[slice, slice], tuple[slice, slice]]]
):
    """
    Calculates everything needed to combine a central geometry with N nearby geometries.
    The nearby geometries need not be in a regular grid. They can be arbitrarily arranged.
    Invoking this does not depend on any real data.
    When trim_bounds is provided, it forces the output to be sliced to fit those bounds.

    For simple use cases, use in conjunction with seamless_seg.apply_weights.

    By default, overlap_weights describes the full area of the central geometry.
    Thus, using it once each on adjacent tiles describes the overlapping area between
    them twice (i.e. in each call).
    To account for this, provide a trim_bounds of half the overlapping area.

    e.g. Say we have two tiles 100 pixels wide next to each other, and they overlap
    40 pixels with each other.
    ```
    geom_a = shapely.box(0, 0, 100, 100)
    geom_b = shapely.box(0, 60, 100, 160)

    out_geom_a, _, _, _, _ = overlap_weights(geom_a, [geom_b])
    out_geom_b, _, _, _, _ = overlap_weights(geom_b, [geom_a])

    print(shapely.area(shapely.intersection(out_geom_a, out_geom_b)))
    # is 40*100 = 4000

    out_geom_a, _, _, _, _ = overlap_weights(geom_a, [geom_b], (0, 0, None, -20))
    out_geom_b, _, _, _, _ = overlap_weights(geom_b, [geom_a], (0, 20, None, None))

    print(shapely.area(shapely.intersection(out_geom_a, out_geom_b)))
    # is 0 because the overlapping region has been trimmed a bit on each side
    ```

    Returns:
        out_geom: defines the space in the output to which the central_weights refers
        central_weights: how much to use the data from central per-pixel (0 to 1)
        centre_from_tile_slc: slices into tile defined by central to select out_geom
        nearby_weights: how much to use each of the nearby geometries
        slice_pairs: how to read from central_weights and nearby_weights for combining
    """
    # Make circle of trust for the central geom
    ylo, xlo, yhi, xhi = central.bounds
    h, w = int(yhi-ylo), int(xhi-xlo)
    circle_of_trust = mk_circle_of_trust(h, w)

    # Make circles of trust for nearby geoms
    nearby_bounds = [n.bounds for n in nearby]
    nearby_shp = [(int(b[2]-b[0]),int(b[3]-b[1])) for b in nearby_bounds]
    nearby_circles_of_trust = np.stack([mk_circle_of_trust(nh, nw) for nh, nw in nearby_shp])

    # Initialise trusts to be read from nearby geoms
    nearby_trusts = np.zeros((len(nearby), h, w))

    # If we need to trim the bounds, we trim only the central geom and associated arrays
    if trim_bounds is not None:
        tylo, txlo, _, _ = trim_bounds
        ylo += tylo
        xlo += txlo
        circle_of_trust = trim_array(circle_of_trust, trim_bounds)
        nearby_trusts = trim_array(nearby_trusts, trim_bounds)
        central, centre_from_tile_slc = trim_box(central, trim_bounds)
    else:
        centre_from_tile_slc = (slice(None, None), slice(None, None))

    # Calcuate nearby trusts and how to slice these trusts for each nearby geom
    overlaps = shapely.intersection(np.array([central]), np.array(nearby))
    slice_pairs = []
    for i, overlap in enumerate(overlaps):
        # Get slices into central and nearby
        oylo, oxlo, _, _ = nearby[i].bounds
        central_slices = shape_to_slices(shapely.affinity.translate(overlap, -ylo, -xlo))
        nearby_slices = shape_to_slices(shapely.affinity.translate(overlap, -oylo, -oxlo))
        slice_pairs.append((central_slices, nearby_slices))

        # Write just for the overlapping parts
        i_c_slices = (i, *central_slices)
        i_n_slices = (i, *nearby_slices)
        nearby_trusts[i_c_slices] = nearby_circles_of_trust[i_n_slices]

    # Normalise pixel-wise
    total = np.concatenate([circle_of_trust[None], nearby_trusts], axis=0).sum(axis=0)
    central_weights = circle_of_trust / total
    nearby_weights = nearby_trusts / total

    return central, central_weights, centre_from_tile_slc, nearby_weights, slice_pairs


def apply_weights(central_tile: np.ndarray, nearby_tiles: list[np.ndarray], weights):
    """
    Apply overlap weights to real tile data.

    Example usage:
    ```
    # Assuming we have: central_geom, nearby_geoms, central_tile, nearby_tiles
    weights = seamless_seg.overlap_weights(central_geom, nearby_geoms)
    out_geom, out_tile = seamless_seg.apply_weights(central_tile, nearby_tiles, weights)
    ```
    """
    out_geom, central_weights, centre_from_tile_slc, nearby_weights, slice_pairs = weights
    out_tile = central_tile[centre_from_tile_slc] * central_weights[..., None]
    for i, (nearby_weight, (central_slices, nearby_slices)) in enumerate(zip(nearby_weights, slice_pairs)):
        out_tile[central_slices] += nearby_tiles[i][nearby_slices] * nearby_weight[central_slices][..., None]
    return out_geom, out_tile


def mk_box_grid(width, height, x_offset=0, y_offset=0, box_width=1, box_height=1, overlap_x=0, overlap_y=0):
    """
    Create a grid of box geometries, stored in a vectorised Shapely array.
    """
    gap_width = box_width - overlap_x
    gap_height = box_height - overlap_y
    xs = np.arange((width-overlap_x) // gap_width) * gap_width
    ys = np.arange((height-overlap_y) // gap_height) * gap_height
    yss, xss = np.meshgrid(ys, xs)
    # fmt: off
    coords = np.array([ # Clockwise squares
        [xss+x_offset,           yss+y_offset],
        [xss+x_offset+box_width, yss+y_offset],
        [xss+x_offset+box_width, yss+y_offset+box_height],
        [xss+x_offset,           yss+y_offset+box_height],
    ]).transpose((2,3,0,1)) # shapes [4, 2, W, H] -> [W, H, 4, 2]
    # fmt: on
    return shapely.polygons(coords)


def calc_gridcell_needed(grid_mask):
    # Calculate which grid cells are needed to calculate grid cells that are in grid_mask
    any_masks = [grid_mask]
    # For each direction, grab an offset grid_mask, indicating which cells are needed due
    # to there being a needed grid cell in that direction
    def _dir_to_slice(v):
        if v == -1:
            return slice(None, -1), slice(1, None)
        elif v == 1:
            return slice(1,None), slice(None, -1)
        else:
            return slice(None), slice(None)
    for (j,i) in GRID_DIR:
        orig_y_slc, out_y_slc = _dir_to_slice(j)
        orig_x_slc, out_x_slc = _dir_to_slice(i)
        mask = np.zeros_like(grid_mask, dtype=bool)
        mask[out_y_slc, out_x_slc] = grid_mask[orig_y_slc, orig_x_slc]
        any_masks.append(mask)
    return np.any(any_masks, axis=0)


def row_by_row_traversal(grid, add_load, add_unload, add_write):
    """
    Traverses a grid, deciding when to load/unload/write tiles.
    The responsibility of this function is to ensure that for every write action marked,
    at that point in the plan, all nearby tiles would be loaded into the cache.
    It is not the responsibility of this function to determine if any such tile is in bounds.

    This traverses row-by-row, keeping two full rows of tiles in the cache at once.
    This will ensure that no tile is read more than once and has a significantly smaller
    memory requirement than keeping all tiles in memory at once.
    This may not be optimal in all cases.
    """
    gh, gw = grid.shape[:2]
    if gh >= gw:
        for gx in range(gw):
            add_load(0, gx)
        for gy in range(gh):
            # Visualising what is in cache:
            #  ("|" means the tile is loaded, "." means the tile is not)
            # The cache should look like this for the row
            # gy-1:  ||||||||
            # gy:    ||||||||
            # gy+1:  ........
            add_load(gy + 1, 0)
            # gy-1:  ||||||||
            # gy:    ||||||||
            # gy+1:  |.......
            for gx in range(gw):
                # |||
                # |||
                # ||.
                add_load(gy + 1, gx + 1)
                add_write(gy, gx)
                add_unload(gy - 1, gx - 1)
                # .||
                # |||
                # |||
            # gy-1: .......|
            # gy:   ||||||||
            # gy+1: ||||||||
            add_unload(gy - 1, gw - 1)
            # gy-1: ........
            # gy:   ||||||||
            # gy+1: ||||||||
        for gx in range(gw):
            add_unload(gh-1, gx)
    else:
        # As above, but transposed
        for gy in range(gh):
            add_load(gy, 0)
        for gx in range(gw):
            add_load(0, gx + 1)
            for gy in range(gh):
                add_load(gy + 1, gx + 1)
                add_write(gy, gx)
                add_unload(gy - 1, gx - 1)
            add_unload(gh - 1, gx - 1)
        for gy in range(gh):
            add_unload(gy, gw-1)


def regular_grid(
        image_size: tuple[int, int],
        tile_size: tuple[int, int],
        overlap: tuple[int, int],
        area: shapely.Geometry=None
) -> np.ndarray[shapely.Geometry]:
    # Unpack sizes
    ih, iw = image_size
    th, tw = tile_size
    if area is None:
        area = shapely.box(0, 0, ih, iw)

    ylo, xlo, yhi, xhi = area.bounds

    # If the area is smaller than the image, then we want to include tiles
    # just outside the area so we can blend into the area properly
    gpylo = max(0, ylo - th)
    gpxlo = max(0, xlo - tw)
    gpyhi = min(ih, yhi + th)
    gpxhi = min(iw, xhi + tw)

    # Make an initial regular grid
    gph, gpw = gpyhi - gpylo, gpxhi - gpxlo
    grid = mk_box_grid(gph, gpw, gpylo, gpxlo, th, tw, *overlap)
    # If the grid doesn't cover the area perfectly (very likely),
    # add another layer of boxes along the edges
    gbyhi, gbxhi = grid[-1, -1].bounds[-2:]
    if gbyhi < yhi:
        # Create a new strip of boxes by copying the last one and then offsetting it such
        # that it is flush with the area boundary.
        gap = int(yhi-gbyhi)
        grid_strip = np.array([shapely.affinity.translate(cell, gap, 0) for cell in grid[-1, :]])
        grid = np.concatenate([grid, grid_strip[None]], axis=0)
    if gbxhi < xhi:
        gap = int(xhi-gbxhi)
        grid_strip = np.array([shapely.affinity.translate(cell, 0, gap) for cell in grid[:, -1]])
        grid = np.concatenate([grid, grid_strip[:, None]], axis=1)
    return grid


def _mk_cache_hash(geom, dir_mask, nearby):
    # Assuming tiles are always the same size, then
    gylo, gxlo, _, _ = geom.bounds
    ylos = np.asarray([gylo] + [shp.bounds[0] for shp in nearby])
    xlos = np.asarray([gxlo] + [shp.bounds[1] for shp in nearby])
    return dir_mask.sum().item(), ylos.mean()-gylo, xlos.mean()-gxlo

@dataclasses.dataclass
class Step:
    action: str
    index: tuple[int, int] # grid index (can be used as cache key)
@dataclasses.dataclass
class LoadStep(Step):
    geom: shapely.Geometry # geometry to load
@dataclasses.dataclass
class WriteStep(Step):
    geom: shapely.Geometry # reference central geometry
    nearby: Sequence[tuple[int, int]] # indexes of geoms defined as nearby
    weight: tuple # outputs of overlap_weights

def plan_from_grid(
        grid: np.ndarray[shapely.Geometry],
        margin: tuple[int, int] = (0, 0),
        area: shapely.Geometry = None,
        traversal_fnc: callable = row_by_row_traversal
) -> list[Step]:
    """
    Create a plan for running on a somewhat arbitrary grid.

    There is a restriction/assumption that must be satisfied:
    For each geometry at grid[y, x] the only geoms which overlap a tile are within +-1
    e.g. for grid[5, 5], the only geoms which overlap it are in the range grid[4:7, 4:7]

    IMPORTANT: All inputs should be YX, not XY.

    `margin` if provided, will subtract a margin from every tile not at the edges
            WARNING: strongly coupled with the grid; currently only proven to work with regular grid.
    `area` can be any arbitrary geometry (i.e. need not be a rectangle)
    `traversal_fnc` lets you define a custom grid traversal algorithm, a callable with:
        traversal_fnc(grid, add_load_step, add_unload_step, add_write_step)
        Which decides when to load which tiles, when to unload them, and when to write them.
        Doesn't need to worry about whether those grid tiles are actually possible or not.

    Returns:
        plan (list[Step]): Describes how to manage the cache, and when/how to write tiles.
            Steps can be load, unload or write.
        grid (np.ndarray[shapely.Geometry]): shaped [H, W], a grid of geometries describing
            where each tile is placed within the image.
    """
    if area is None:
        _, _, gyhi, gxhi = shapely.unary_union(grid).bounds
        area = shapely.box(0, 0, gyhi, gxhi)

    # Determine grid boundaries and which cells are possible
    gh, gw = grid.shape[:2]
    grid_in_area = shapely.intersects(grid, area)
    gridcell_needed = calc_gridcell_needed(grid_in_area)

    plan = []
    weight_cache = {}
    # By pushing these to helper functions we separate the traversal logic from
    # deciding to load/unload/write only for tiles that need it (based on provided area)
    def _in_bounds(gy, gx):
        return 0 <= gy < gh and 0 <= gx < gw
    def _add_load_step(gy, gx):
        if _in_bounds(gy, gx) and gridcell_needed[gy, gx]:
            plan.append(LoadStep(action='load', index=(gy, gx), geom=grid[gy, gx]))
    def _add_unload_step(gy, gx):
        if _in_bounds(gy, gx) and gridcell_needed[gy, gx]:
            plan.append(Step(action='unload', index=(gy, gx)))
    def _calc_weight(gy, gx, geom, dir_mask):
        # Check which directions are within the grid
        nearby = [(int(gy + j), int(gx + i)) for j, i in GRID_DIR[dir_mask]]
        nearby_geom = np.array([grid[y, x] for y, x in nearby])
        # Based on which directions have a tile, determine how to trim the output
        trim_bounds = get_trimmed_bounds(margin, GRID_DIR[dir_mask])

        # Only create new weights if we have to
        cache_hash = _mk_cache_hash(geom, dir_mask, nearby_geom)
        if cache_hash in weight_cache:
            # All but one of the weights are relative. The absolute output is the out_geom.
            # So, here we account for a different input geom after-the-fact.
            (out_geom, a, b, c, d), other_geom = weight_cache[cache_hash]
            oylo, oxlo, _, _ = other_geom.bounds
            tylo, txlo, _, _ = geom.bounds
            out_geom = shapely.affinity.translate(out_geom, tylo-oylo, txlo-oxlo)
            return (out_geom, a, b, c, d), nearby

        # Finally calculate the weights for combining this tile with its nearby.
        weight = overlap_weights(geom, nearby_geom, trim_bounds)
        weight_cache[cache_hash] = (weight, geom)
        return weight, nearby
    def _add_write_step(gy, gx):
        if grid_in_area[gy, gx]:
            geom = grid[gy, gx]
            dir_mask = np.asarray([_in_bounds(gy+j, gx+i) for j,i in GRID_DIR])
            weight, nearby = _calc_weight(gy, gx, geom, dir_mask)
            base = {'geom': geom, 'index': (gy, gx), 'weight': weight}
            plan.append(WriteStep(action='write', **base, nearby=nearby))

    traversal_fnc(grid, _add_load_step, _add_unload_step, _add_write_step)

    return plan

def plan_regular_grid(
    image_size: tuple[int, int],
    tile_size: tuple[int, int],
    overlap: tuple[int, int],
    area: shapely.Geometry = None,
    traversal_fnc: callable = row_by_row_traversal,
) -> tuple[list[Step], np.ndarray[shapely.Geometry]]:
    """
    Plans out running segmentation over a single large image by tiling, overlapping
    and blending between adjacent tiles in a regular grid.

    IMPORTANT: All inputs should be YX, not XY.

    Does not depend on any real data; merely creates a geometry plan based on size data.

    `area` can be any arbitrary geometry (i.e. need not be a rectangle)
    `traversal_fnc` lets you define a custom grid traversal algorithm, a callable with:
        traversal_fnc(grid, add_load_step, add_unload_step, add_write_step)
        Which decides when to load which tiles, when to unload them, and when to write them.
        Doesn't need to worry about whether those grid tiles are actually possible or not.

    Returns:
        plan (list[Step]): Describes how to manage the cache, and when/how to write tiles.
            Steps can be load, unload or write.
        grid (np.ndarray[shapely.Geometry]): shaped [H, W], a grid of geometries describing
            where each tile is placed within the image.
    """
    oh, ow = overlap
    if not(oh % 2 == 0 or ow % 2 == 0):
        raise ValueError('Overlap must be an even number')
    margin = oh // 2, ow // 2
    grid = regular_grid(image_size, tile_size, overlap, area)
    return plan_from_grid(grid, margin, area, traversal_fnc), grid


def batched_tile_get(
        geoms: list[tuple[tuple[int, int], shapely.Geometry]],
        batch_size: int,
        get_tiles_fnc: callable,
):
    """
    Takes some function to get tiles `get_tiles_fnc` which is to expect a batch of geoms at once.
    Yields individual tiles
    """
    batch_indices = []
    batch_geoms = []
    for index, geom in geoms:
        batch_indices.append(index)
        batch_geoms.append(geom)
        if len(batch_geoms) == batch_size:
            tiles = get_tiles_fnc(batch_indices, batch_geoms)
            for past_index, tile in zip(batch_indices, tiles):
                yield tile
            batch_indices = []
            batch_geoms = []
    tiles = get_tiles_fnc(batch_indices, batch_geoms)
    for past_index, tile in zip(batch_indices, tiles):
        yield tile


def threaded_batched_tile_get(
        geoms: list[tuple[tuple[int, int], shapely.Geometry]],
        batch_size: int,
        get_tiles_fnc: callable,
        max_prefetched: int,
) -> Generator[tuple[tuple[int, int], np.ndarray], None, None]:
    """
    Takes some function to get tiles `get_tiles_fnc` which is to expect a batch of geoms at once.
    Executes that function in a thread, prefetching those tiles before they are needed.
    Yields individual tiles
    """
    out_queue = queue.Queue(max_prefetched)
    def _wrap_queue():
        for tile in batched_tile_get(geoms, batch_size, get_tiles_fnc):
            out_queue.put(tile)

    thread = threading.Thread(target=_wrap_queue)
    thread.start()
    for _ in geoms:
        yield out_queue.get()



def analyse_plan(plan: list[Step]) -> tuple[int, int, int]:
    """ Counts maximum tiles loaded at once, total tiles loaded, and total write calls."""
    loaded = 0
    total_loaded = 0
    max_loaded = 0
    write = 0
    for step in plan:
        if step.action == 'load':
            loaded += 1
            total_loaded += 1
        elif step.action == 'unload':
            loaded -= 1
        if loaded > max_loaded:
            max_loaded = loaded
        if step.action == 'write':
            write += 1
    return max_loaded, total_loaded, write


def get_plan_input_geoms(plan):
    return [(step.index, step.geom) for step in plan if step.action == 'load']


def _check_plan_doesnt_exceed(plan, max_tiles):
    if max_tiles is None:
        # No maximum set
        return

    max_loaded, _, _ = analyse_plan(plan)
    if max_loaded > max_tiles:
        raise Exception('Traversal method in plan would hold more than max tiles in memory')


def noop(*args, **kwargs): pass
def serialise_index(index):
    return f"{index[0]}-{index[1]}.npy"
def run_plan(
        plan: list[Step],
        tiles: Iterable,
        max_tiles: int = None,
        disk_cache_dir: Path = None,
        on_load: callable = noop,
        on_unload: callable = noop,
        on_step: callable = noop,
        on_disk_evict: callable = noop,
        on_disk_restore: callable = noop,
) -> Generator[tuple[tuple[int, int], shapely.Geometry, np.ndarray], None, None]:
    cache = collections.OrderedDict()
    disk_cache = {}

    if max_tiles is not None and max_tiles <= 8:
        raise ValueError("If provided, max_tiles must be greater than 8")

    if disk_cache_dir is None:
        _check_plan_doesnt_exceed(plan, max_tiles)
    else:
        if max_tiles is None:
            raise ValueError("If disk_cache_dir is set, then max_tiles should be set")
        disk_cache_dir.mkdir(exist_ok=True)

    # Two-level cache management functions; evicting to disk and restoring from disk.
    def _evict_oldest():
        oldest_index, oldest_tile = cache.popitem(False)
        on_disk_evict(oldest_index)
        fpath = disk_cache_dir / serialise_index(oldest_index)
        np.save(fpath, oldest_tile)
        disk_cache[oldest_index] = fpath

    def _resolve_restore(index):
        if index in cache:
            cache.move_to_end(index)
            return cache[index]
        if len(cache) == max_tiles:
            _evict_oldest()
        cache[index] = np.load(disk_cache[index])
        on_disk_restore(index)
        del disk_cache[index]
        return cache[index]


    # Run plan
    for n, step in enumerate(plan):
        if step.action == 'load':
            # Put tile into cache
            if disk_cache_dir is not None:
                if len(cache) == max_tiles:
                    _evict_oldest()
            cache[step.index] = next(tiles)
            on_load(step.index)
        elif step.action == 'unload':
            # Remove tile from cache
            del cache[step.index]
            on_unload(step.index)
        elif step.action == 'write':
            # Collect nearby tiles
            nearby_tiles = []
            for index in step.nearby:
                if disk_cache_dir is None:
                    tile = cache[index]
                else:
                    tile = _resolve_restore(index)
                nearby_tiles.append(tile)

            # Collect central tile
            if disk_cache_dir is None:
                central_tile = cache[step.index]
            else:
                central_tile = _resolve_restore(step.index)

            # Apply weights from plan to create final output tile
            out_geom, out_tile = apply_weights(central_tile, nearby_tiles, step.weight)
            yield step.index, out_geom, out_tile
        else:
            raise Exception('Unknown plan action')
        on_step(n)
