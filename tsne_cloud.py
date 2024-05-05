import glob
import struct
from typing import List
from sklearn.manifold import TSNE
import numpy as np

from generate import generate_images, num_range


def get_tsne_coordinates(
    input_files: List[str],
    metric: str = "euclidean",
    n_iter: int = 500,
    perplexity: int = 10,
    random_state: int = 12
):
    vectors = []
    for vec in enumerate(input_files):
        with open(vec, 'rb') as f:
            data = np.load(f)
            vectors.append(data)

    tsne_3d = TSNE(perplexity=perplexity,
                   n_components=3,
                   init='pca',
                   metric=metric,
                   n_iter=n_iter,
                   andom_state=random_state,
                   verbose=0)
    points = tsne_3d.fit_transform(vectors)
    return points


def write_pointcloud(filename, xyz_points, rgb_points=None):
    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3, 'Input XYZ points should be\
          Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert xyz_points.shape == rgb_points.shape, 'Input RGB colors should be\
          Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    with open(filename, 'wb') as fid:
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes(f'element vertex {xyz_points.shape[0]}\n', 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))

        # Write 3D points to .ply file
        for i in range(xyz_points.shape[0]):
            fid.write(bytearray(struct.pack("fffBBB",
                                            xyz_points[i, 0],
                                            xyz_points[i, 1],
                                            xyz_points[i, 2],
                                            rgb_points[i, 0],
                                            rgb_points[i, 1],
                                            rgb_points[i, 2]
                                            )))


def generate_atlas(img_list: List[str], out_file: str):
    pass


def main():
    input_dir = "data"
    model = "network.pkl"
    seeds = num_range('0-100')

    generate_images(
        network_pkl=model,
        seeds=seeds,
        truncation_psi=1,
        noise_mode='const',
        outdir=input_dir,
        class_idx=None,
        projected_w=None,
        vector=True
    )

    vectors = sorted(glob.glob(f'{input_dir}/*.npy'))
    images = sorted(glob.glob(f'{input_dir}/*.png'))

    points = get_tsne_coordinates(vectors)
    write_pointcloud('pointcloud.ply', points)

    generate_atlas(images, 'atlas.png')
