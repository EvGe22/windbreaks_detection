import os
import argparse
import rasterio
import numpy as np

from os.path import join, splitext


def search_band(band, folder, file_type):
    for file in os.listdir(folder):
        if band in file and file.endswith(file_type):
            return splitext(file)[0]

    return None


def to_tiff(img_file, output_type='Float32'):
    os.system(
        f'gdal_translate -ot {output_type} \
        {img_file} {splitext(img_file)[0]}.tif'
    )


def scale_img(img_file, min_value=0, max_value=255, output_type='Byte'):
    with rasterio.open(img_file) as src:
        img = src.read(1)
        img = np.nan_to_num(img)
        mean_ = img.mean()
        std_ = img.std()
        min_ = max(img.min(), mean_ - 2 * std_)
        max_ = min(img.max(), mean_ + 2 * std_)

        os.system(
            f'gdal_translate -ot {output_type} \
            -scale {min_} {max_} {min_value} {max_value} \
            {img_file} {os.path.splitext(img_file)[0]}_scaled.tif'
        )


def get_ndvi(b4_file, b8_file, ndvi_file):
    os.system(
        f'gdal_calc.py -A {b4_file} -B {b8_file} \
        --outfile={ndvi_file} \
        --calc="(B-A)/(A+B+0.001)" --type=Float32'
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Script for predicting masks.')
    parser.add_argument(
        '--data_folder', '-f', dest='data_folder',
        required=True, help='Path to downloaded images'
    )
    parser.add_argument(
        '--save_path', '-s', dest='save_path', default='data',
        help='Path to directory where results will be stored'
    )
    return parser.parse_args()


def get_tile_img_folder_path(base_path):
    granule_folder = join(base_path, 'GRANULE')
    tile_folder = list(os.walk(granule_folder))[0][1][-1]
    img_folder = join(granule_folder, tile_folder, 'IMG_DATA', 'R10m')
    return tile_folder, img_folder


def create_tif(img_folder_path, save_path):
    b4_name = join(img_folder_path, search_band('B04', img_folder_path, 'jp2'))
    b8_name = join(img_folder_path, search_band('B08', img_folder_path, 'jp2'))
    rgb_name = join(img_folder_path, search_band('TCI', img_folder_path, 'jp2'))
    ndvi_name = join(img_folder_path, 'ndvi')

    to_tiff(f'{b4_name}.jp2')
    to_tiff(f'{b8_name}.jp2')
    to_tiff(f'{rgb_name}.jp2', 'Byte')

    get_ndvi(f'{b4_name}.tif', f'{b8_name}.tif', f'{ndvi_name}.tif')

    scale_img(f'{ndvi_name}.tif')
    scale_img(f'{b8_name}.jp2')

    os.system(
        f'gdal_merge.py -separate -o {save_path} \
        {rgb_name}.tif {ndvi_name}_scaled.tif'
    )

    for item in os.listdir(img_folder_path):
        if item.endswith('.tif'):
            os.remove(join(img_folder_path, item))


if __name__ == '__main__':
    args = parse_args()

    tile_folder, img_folder = get_tile_img_folder_path(args.data_folder)
    save_file = join(args.save_path, f'{tile_folder}.tif')
    create_tif(img_folder, save_file)

