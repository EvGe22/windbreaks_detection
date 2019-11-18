import argparse

from clearcut_data_prep.binary_mask_converter import poly2mask, split_mask
from clearcut_data_prep.prepare_tiff import get_tile_img_folder_path, create_tif
from clearcut_data_prep.tile_tiff import divide_into_pieces
from os.path import join
from glob import glob
import geopandas as gp
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument(
        '--data_folder', '-d', dest='data_folder',
        required=True, help='Path to downloaded images'
    )
    parser.add_argument(
        '--output_folder', '-o', dest='output_folder', required=True,
        help='Path to directory where results will be stored'
    )
    parser.add_argument(
        '--vegetation_path', '-v', dest='vegetation_path', required=True,
        help='Path to file with shapely polygons'
    )
    return parser.parse_args()


def get_data_mask_paths(tiles_folder, masks_folder):
    data_files = glob(join(tiles_folder, '*.png'))
    mask_files = [f.replace(tiles_folder, masks_folder) for f in data_files]
    return data_files, mask_files


if __name__ == '__main__':

    args = parse_args()

    markup = gp.read_file(args.vegetation_path)
    polys = markup.loc[:, 'geometry']
    all_image_path_dfs = []

    for satellite_dir in glob(join(args.data_folder, '*')):
        tile_folder, img_folder = get_tile_img_folder_path(satellite_dir)
        save_folder = join(args.output_folder, tile_folder)
        os.makedirs(save_folder, exist_ok=True)
        tiff_path = join(save_folder, f'{tile_folder}.tif')
        create_tif(img_folder, tiff_path)

        tiles_path = join(save_folder, f'{tile_folder}_tiles')
        pieces_path = join(save_folder, f'{tile_folder}_pieces.csv')
        divide_into_pieces(image_path=tiff_path, save_path=tiles_path,
                           pieces_file=pieces_path, width=512, height=512)

        mask_path = poly2mask(polys, tiff_path, save_folder)
        mask_tiles_path = join(save_folder, f'{tile_folder}_mask_tiles')
        split_mask(mask_path, mask_tiles_path, pieces_path)

        tile_paths, mask_paths = get_data_mask_paths(tiles_path, mask_tiles_path)
        all_image_path_dfs.append(pd.DataFrame({
            'tile_paths': tile_paths,
            'mask_paths': mask_paths
        }))

    pd.concat(all_image_path_dfs).to_csv(join(args.output_folder, 'result_files.csv'), index=False)
