import argparse

import numpy as np
import pandas as pd
import imageio
from sklearn.model_selection import train_test_split
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Script for cleaning masks.')
    parser.add_argument('--remove', type=int, default=0,
                        help='1/0 boolean value. Will remove filtered values if set to 1')
    parser.add_argument('--threshold', default=0., type=float)
    parser.add_argument('--input_csv_path', required=True)
    parser.add_argument('--test_size', default=0., type=float,
                        help='Fraction of the data to be used as test, set to 0 if not split needed')
    parser.add_argument('--out_path', help='Folder to save the resulting files')

    return parser.parse_args()


def filter_on_threshold(result_csv_path, threshold, remove, out_folder):
    df = pd.read_csv(result_csv_path)
    non_zero_pixel_ratio = []

    for m_p in df['mask_paths'].values:
        image_array = imageio.imread(m_p)
        non_zero_pixel_ratio.append(np.count_nonzero(image_array) / image_array.size)

    df['non_zero_mask'] = non_zero_pixel_ratio

    gt_threshold = df[df['non_zero_mask'] > threshold][['tile_paths', 'mask_paths']]
    save_path = os.path.join(out_folder, f'gt_{threshold}.csv')
    gt_threshold.to_csv(save_path, index=False)

    if remove:
        to_remove = df[df['non_zero_mask'] <= threshold][['tile_paths', 'mask_paths']]
        for tp, mp in to_remove.values:
            os.remove(tp)
            os.remove(mp)

    return save_path


def split_dataset(df_path, test_size, out_folder):
    if test_size <= 0:
        return
    df_name = os.path.basename(df_path).split('.')[0]
    df = pd.read_csv(df_path)

    indices = np.arange(len(df))
    train_ind, test_ind = train_test_split(indices, test_size=test_size, shuffle=True)

    df.iloc[train_ind].to_csv(os.path.join(out_folder, f'{df_name}_train.csv'), index=False)
    df.iloc[test_ind].to_csv(os.path.join(out_folder, f'{df_name}_test.csv'), index=False)


if __name__ == '__main__':
    args = parse_args()
    out_path = args.out_path
    if out_path is None:
        out_path = os.path.dirname(args.input_csv_path)
    filtered_df_path = filter_on_threshold(args.input_csv_path, args.threshold, args.remove, out_path)
    split_dataset(filtered_df_path, args.test_size, out_path)
