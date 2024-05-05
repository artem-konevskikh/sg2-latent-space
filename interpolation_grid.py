import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


def interpolate_arrays(array1, array2, alpha):
    # Ensure arrays have the same shape
    assert array1.shape == array2.shape, "Arrays must have the same shape"

    # Interpolate between the two arrays
    interpolated_array = (1 - alpha) * array1 + alpha * array2
    return interpolated_array


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c'
    or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


@click.command()
@click.pass_context
@click.option(
    '--network', 'network_pkl',
    help='Network pickle filename',
    required=True
)
@click.option(
    '--outdir',
    help='Where to save the output images',
    type=str,
    required=True,
    metavar='DIR'
)
@click.option(
    '--seeds',
    type=num_range,
    required=True,
    help='List of random seeds'
)
@click.option(
    '--trunc', 'truncation_psi',
    type=float,
    help='Truncation psi',
    default=1,
    show_default=True
)
@click.option(
    '--noise-mode',
    help='Noise mode',
    type=click.Choice(['const', 'random', 'none']),
    default='const',
    show_default=True
)
@click.option(
    '--x', 'num_steps_x',
    type=int,
    help='Number of steps in x direction',
    default=10,
    required=True,
    show_default=True
)
@click.option(
    '--y', 'num_steps_y',
    type=int,
    help='Number of steps in y direction',
    default=10,
    required=True,
    show_default=True
)
def interpolation_grid(
    ctx: click.Context,
    network_pkl: str,
    outdir: str,
    seeds: List[int],
    truncation_psi: float,
    num_steps_x: int,
    num_steps_y: int,
    noise_mode: str
):
    if seeds is None:
        ctx.fail('Seeds option is required')

    if len(seeds) != 4:
        ctx.fail('Must provide exactly 4 seeds')

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)
    label = torch.zeros([1, G.c_dim], device=device)
    np_zs = [np.random.RandomState(seed).randn(1, G.z_dim) for seed in seeds]
    first_row_zs = np.zeros((num_steps_x+1, G.z_dim))
    last_row_zs = np.zeros((num_steps_x+1, G.z_dim))
    grid_zs = np.zeros(
        (num_steps_y+1, first_row_zs.shape[0], first_row_zs.shape[1])
    )

    for x in range(num_steps_x+1):
        alpha = x / num_steps_x
        first_row_zs[x] = interpolate_arrays(np_zs[0], np_zs[1], alpha)
        last_row_zs[x] = interpolate_arrays(np_zs[2], np_zs[3], alpha)
    for y in range(num_steps_y+1):
        alpha_y = y / num_steps_y
        grid_zs[y] = interpolate_arrays(
            first_row_zs,
            last_row_zs,
            alpha_y
        )

    for y, row in enumerate(grid_zs):
        for x, z in enumerate(row):
            z = torch.from_numpy(z).to(device)
            img = G(z, label, truncation_psi=truncation_psi,
                    noise_mode=noise_mode)
            img = (
                img.permute(0, 2, 3, 1) * 127.5 + 128
            ).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(
                img[0].cpu().numpy(), 'RGB'
            ).save(
                f'{outdir}/frame{x:04d}_{y:04d}.png'
            )

# def interpolation_grid():
#     array1 = np.array([1, 2, 3])
#     array2 = np.array([4, 5, 6])
#     array3 = np.array([7, 8, 9])
#     array4 = np.array([10, 11, 12])

#     num_steps_x = 10
#     num_step_y = 10

#     top_row = np.zeros((num_steps_x+1, array1.shape[0]))
#     bottom_row = np.zeros((num_steps_x+1, array1.shape[0]))
#     grid = np.zeros((num_step_y+1, top_row.shape[0], top_row.shape[1]))

#     for x in range(num_steps_x+1):
#         alpha = x / num_steps_x
#         top_row[x] = interpolate_arrays(array1, array2, alpha)
#         bottom_row[x] = interpolate_arrays(array3, array4, alpha)
#     for y in range(num_step_y+1):
#         alpha_y = y / num_step_y
#         grid[y] = interpolate_arrays(
#             top_row,
#             bottom_row,
#             alpha_y
#         )
#     # print(top_row, bottom_row)
#     print(grid)


if __name__ == "__main__":
    interpolation_grid()
