import argparse
import json
import numpy as np
import os
import tifffile as tiff
import sys


def format_without_zero(value):
    if value.is_integer():
        return str(int(value))
    else:
        return str(value)


def main(dataset_name, tile_number, tile_second_index, selected_classes, samples_num, seed, new_labels,
         svmlight_out_file):
    # Seed for Numpy RNG
    np.random.seed(seed)

    # Data loading
    tile, tile_gt_rgb, rgb_to_class_number = None, None, {}
    if dataset_name == 'toulouse':
        tile_filepath = 'SemCity-Toulouse/img_multispec_05/TLS_BDSD_M/TLS_BDSD_M_{:02d}.tif'.format(tile_number)
        tile_gt_rgb_filepath = 'SemCity-Toulouse/semantic_05/TLS_GT/TLS_GT_{:02d}.tif'.format(tile_number)

        tile = tiff.imread(tile_filepath)
        tile_gt_rgb = tiff.imread(tile_gt_rgb_filepath)

        rgb_to_class_number = {
            (255, 255, 255): 0,  # Void
            (38,   38,  38): 1,  # Impervious surface
            (238, 118,  33): 2,  # Building
            (34,  139,  34): 3,  # Pervious surface
            (0,   222, 137): 4,  # High vegetation
            (255,   0,   0): 5,  # Car
            (0,     0, 238): 6,  # Water
            (160,  30, 230): 7   # Sport venues
        }

    elif dataset_name == 'potsdam':
        tile_dsm_filepath = 'Potsdam/1_DSM/dsm_potsdam_{:02d}_{:02d}.tif'.format(tile_number, tile_second_index)
        tile_top_filepath = 'Potsdam/4_Ortho_RGBIR/top_potsdam_{:02d}_{:02d}_RGBIR.tif'.format(tile_number,
                                                                                               tile_second_index)
        tile_gt_rgb_filepath = 'Potsdam/5_Labels_all/top_potsdam_{:02d}_{:02d}_label.tif'.format(tile_number,
                                                                                                 tile_second_index)

        tile_dsm = tiff.imread(tile_dsm_filepath)
        tile_dsm_reshaped = np.reshape(tile_dsm, (tile_dsm.shape[0], tile_dsm.shape[1], 1))
        tile_top = tiff.imread(tile_top_filepath)
        tile = np.concatenate((tile_dsm_reshaped, tile_top), axis=2)

        tile_gt_rgb = tiff.imread(tile_gt_rgb_filepath)

        rgb_to_class_number = {
            (255, 255, 255): 0,  # Impervious surface
            (255,   0,   0): 1,  # Clutter/background
            (0,     0, 255): 2,  # Building
            (0,   255, 255): 3,  # Low vegetation
            (0,   255,   0): 4,  # Tree
            (255, 255,   0): 5   # Car
        }
    else:
        print('Unknown dataset name: {}'.format(dataset_name), file=sys.stderr)

    # Reshaping
    pixels_num = tile.shape[0] * tile.shape[1]
    bands_num = tile.shape[2]
    bands_gt_rgb_num = tile_gt_rgb.shape[2]

    tile_reshaped = np.reshape(tile, (pixels_num, bands_num))
    tile_gt_rgb_reshaped = np.reshape(tile_gt_rgb, (pixels_num, bands_gt_rgb_num))

    # Useful variables
    classes_num = len(selected_classes)
    samples_per_class = int(samples_num / classes_num)
    remainder = samples_num % classes_num

    class_number_to_rgb = {v: np.asarray(k) for k, v in rgb_to_class_number.items()}

    new_labels = new_labels if len(new_labels) == len(selected_classes) else selected_classes
    class_number_to_label = {cn: nl for cn, nl in zip(selected_classes, new_labels)}
    label_to_class_number = {v: k for k, v in class_number_to_label.items()}

    # Samples random selection
    dataset_x_vals = np.zeros((samples_num, bands_num))
    dataset_y_vals_rgb = np.zeros((samples_num, bands_gt_rgb_num))
    dataset_y_vals = np.zeros(samples_num)
    pixels_positions = np.zeros(samples_num, dtype=np.int64)
    for i, class_label in enumerate(selected_classes):
        class_i_samples_num = samples_per_class
        if remainder > 0:
            class_i_samples_num += 1
            remainder -= 1

        selection_mask_vec = np.full(pixels_num, 0, dtype=np.int8)
        class_i_indices = np.where(np.all(np.equal(tile_gt_rgb_reshaped, class_number_to_rgb[class_label]), axis=1))[0]
        np.random.shuffle(class_i_indices)
        class_i_selected_indices = sorted(class_i_indices[:class_i_samples_num])
        selection_mask_vec[class_i_selected_indices] = 1

        x_vals_tmp = np.squeeze(tile_reshaped[np.argwhere(selection_mask_vec == 1)])
        y_vals_rgb_tmp = np.squeeze(tile_gt_rgb_reshaped[np.argwhere(selection_mask_vec == 1)])
        # Round-robin initialization
        for j in range(class_i_samples_num):
            dataset_index = classes_num * j + selected_classes.index(class_label)
            dataset_x_vals[dataset_index] = x_vals_tmp[j]
            dataset_y_vals_rgb[dataset_index] = y_vals_rgb_tmp[j]
            pixels_positions[dataset_index] = class_i_selected_indices[j]

    # Conversion to new labels (using class numbers)
    for i in range(samples_num):
        dataset_y_vals[i] = class_number_to_label[rgb_to_class_number[tuple(dataset_y_vals_rgb[i])]]
    dataset_y_vals = dataset_y_vals.astype(int)

    # Save the dataset
    if svmlight_out_file is not None:
        svmlight_out_file = os.path.splitext(svmlight_out_file)[0] + '.txt'
        os.makedirs(os.path.dirname(svmlight_out_file), exist_ok=True)
    else:
        svmlight_out_file = '{}_t{:02d}{}_{}c_{}s.txt'.format(
            dataset_name, tile_number, '.{:02d}'.format(tile_second_index) if dataset_name == 'potsdam' else '',
            classes_num, samples_num
        )
    with open(svmlight_out_file, 'w') as svmlight_file:
        for x_vals, y_val in zip(dataset_x_vals, dataset_y_vals):
            svmlight_file.write('{} {}\n'.format(
                y_val, ' '.join([f'{i+1}:{format_without_zero(f)}' for i, f in enumerate(x_vals)]))
            )

    # Save the original pixels positions
    pixels_pos_dict = {
        'original_img_shape': (tile.shape[0], tile.shape[1]),
        'flattened_indices': pixels_positions.tolist()
    }
    pixels_pos_out_file = svmlight_out_file.replace('.txt', '_pixels_pos.json')
    with open(pixels_pos_out_file, 'w') as pixels_pos_file:
        json.dump(pixels_pos_dict, pixels_pos_file, ensure_ascii=False, indent=True)

    # Save the mappings for a later usage
    mappings = {
        'label_to_class_number': label_to_class_number,
        'class_number_to_rgb': {k: v.tolist() for k, v in class_number_to_rgb.items()}
    }
    mappings_out_file = svmlight_out_file.replace('.txt', '_mappings.json')
    with open(mappings_out_file, 'w') as mappings_file:
        json.dump(mappings, mappings_file, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to convert part of a tiff image taken from SemCity-Toulouse '
                                                 'or Potsdam into a svmlight format dataset.')
    parser.add_argument('--dataset-name', metavar='dataset_name', type=str, nargs='?', default='toulouse',
                        help='name of the input dataset, either \'toulouse\' or \'potsdam\'.')
    parser.add_argument('--tile-number', metavar='tile_number', type=int, nargs='?', default=-1,
                        help='number of the tile to convert.')
    parser.add_argument('--tile-second-index', metavar='tile_second_index', type=int, nargs='?', default=-1,
                        help='second index needed to identify the tile (only for potsdam).')
    parser.add_argument('--selected-classes', metavar='selected_classes', type=int, nargs='+', default=[],
                        help='classes to take into account.')
    parser.add_argument('--samples-num', metavar='samples_num', type=int, nargs='?', default=1000,
                        help='number of pixels to select.')
    parser.add_argument('--seed', metavar='seed', type=int, nargs='?', default=27,
                        help='seed for the random number generator.')
    parser.add_argument('--new-labels', metavar='new_labels', type=int, nargs='+', default=[],
                        help='list of (integer) labels to use in the output file')
    parser.add_argument('--svmlight-out-file', metavar='svmlight_out_file', type=str, nargs='?', default=None,
                        help='output file name in svmlight format (.txt extension).')
    args = parser.parse_args()

    dataset_name = args.dataset_name

    tile_number = args.tile_number
    if tile_number == -1:
        tile_number = 4 if args.dataset_name == 'toulouse' else 6

    tile_second_index = args.tile_second_index
    if dataset_name == 'potsdam' and tile_second_index == -1:
        tile_second_index = 9

    selected_classes = args.selected_classes
    if len(selected_classes) == 0:
        selected_classes = [2, 3, 6] if dataset_name == 'toulouse' else [2, 3, 4]

    main(dataset_name, tile_number, tile_second_index, selected_classes, args.samples_num, args.seed, args.new_labels,
         args.svmlight_out_file)
