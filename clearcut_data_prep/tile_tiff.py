import os
import csv
import argparse

import imageio
import rasterio
import numpy as np

from tqdm import tqdm
from rasterio.windows import Window
from rasterio.plot import reshape_as_image


def divide_into_pieces(image_path, save_path, pieces_file, width, height):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print('Data directory created.')

    with rasterio.open(image_path) as src, open(pieces_file, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([
            'original_image', 'piece_image', 'piece_geojson',
            'start_x', 'start_y', 'width', 'height'
        ])

        for j in tqdm(range(0, src.height // height)):
            for i in range(0, src.width // width):
                raster_window = src.read(
                    window=Window(i * width, j * height, width, height)
                )
                image_array = reshape_as_image(raster_window)[:, :, [3, 0, 1]]

                if np.count_nonzero(image_array) > image_array.size * 0.9:
                    filename_w_ext = os.path.basename(image_path)
                    filename, _ = os.path.splitext(filename_w_ext)
                    image_format = 'png'
                    piece_name = f'{filename}_nrg_{j}_{i}.{image_format}'

                    imageio.imwrite(f'{save_path}/{piece_name}', image_array)

                    writer.writerow([
                        filename_w_ext, piece_name, 'piece_geojson_name',
                        i * width, j * height, width, height
                    ])

    csvFile.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for slicing images into smaller pieces.'
    )
    parser.add_argument(
        '--image_path', '-ip', dest='image_path',
        required=True, help='Path to source image'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../../data', help='Path to directory where pieces will be stored'
    )
    parser.add_argument(
        '--width', '-w', dest='width',
        default=224, type=int, help='Width of a piece'
    )
    parser.add_argument(
        '--height', '-hgt', dest='height',
        default=224, type=int, help='Height of a piece'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    divide_into_pieces(args.image_path, args.save_path, args.width, args.height)
    # python tile_tiff.py -ip /home/evge/Tasks/Geo-DIplom/data/test_process/L2A.tif -sp /home/evge/Tasks/Geo-DIplom/data/test_process/L2A_T36UXU_A012960_20190830T084008_png_tiles