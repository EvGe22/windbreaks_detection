import os
import re
import imageio
import datetime
import argparse
import numpy as np
import pandas as pd
import rasterio as rs
import geopandas as gp

from rasterio import features


def poly2mask(polys, image_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Output directory created.")

    with rs.open(image_path) as image:
        polys = polys.to_crs({'init': image.crs})

        mask = features.rasterize(
            shapes=polys,
            out_shape=(image.height, image.width),
            transform=image.transform,
            default_value=255
        )

    image.close()

    filename = f'{save_path}/{re.split(r"[/.]", image_path)[-2]}_mask.png'
    imageio.imwrite(filename, mask)

    return filename


def split_mask(mask_path, save_path, image_pieces_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Output directory created.")

    pieces_info = pd.read_csv(
        image_pieces_path, dtype={
            'start_x': np.int64, 'start_y': np.int64,
            'width': np.int64, 'height': np.int64
        }
    )
    mask = imageio.imread(mask_path)
    for i in range(pieces_info.shape[0]):
        piece = pieces_info.loc[i]
        piece_mask = mask[
             piece['start_y']: piece['start_y'] + piece['height'],
             piece['start_x']: piece['start_x'] + piece['width']
        ]
        filename = '{}/{}.png'.format(
            save_path,
            re.split(r'[/.]', piece['piece_image'])[-2]
        )
        imageio.imwrite(filename, piece_mask)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for creating binary mask from geojson.')
    parser.add_argument(
        '--polys_path', '-pp', dest='polys_path',
        required=True, help='Path to the polygons'
    )
    parser.add_argument(
        '--image_path', '-ip', dest='image_path',
        required=True, help='Path to source image'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../output',
        help='Path to directory where mask will be stored'
    )
    parser.add_argument(
        '--mask_path', '-mp', dest='mask_path',
        help='Path to the mask',
        required=False
    )
    parser.add_argument(
        '--pieces_path', '-pcp', dest='pieces_path',
        help='Path to directory where pieces will be stored',
        default='../output/masks'
    )
    parser.add_argument(
        '--pieces_info', '-pci', dest='pieces_info',
        help='Path to the image pieces info'
    )
    parser.add_argument(
        '--type_filter', '-tf', dest='type_filter',
        help='Type of clearcut: "open" or "closed")'
    )
    parser.add_argument(
        '--filter_by_date', '-fd', dest='filter_by_date',
        action='store_true', default=False,
        help='Filter by date is enabled'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mask_path = poly2mask(
        args.polys_path, args.image_path,
        args.save_path
    )
    if args.mask_path is not None:
        mask_path = args.mask_path

    split_mask(mask_path, args.pieces_path, args.pieces_info)
