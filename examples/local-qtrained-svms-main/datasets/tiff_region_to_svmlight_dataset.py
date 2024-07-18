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


def main(dataset_name, tile_number, tile_second_index, selected_classes, region_ul_corner, region_br_corner,
         new_labels, svmlight_out_file):
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

    # Useful variables
    tile_height, tile_width = tile.shape[0], tile.shape[1]

    classes_num = len(selected_classes)

    class_number_to_rgb = {v: np.asarray(k) for k, v in rgb_to_class_number.items()}

    new_labels = new_labels if len(new_labels) == len(selected_classes) else selected_classes
    class_number_to_label = {cn: nl for cn, nl in zip(selected_classes, new_labels)}
    label_to_class_number = {v: k for k, v in class_number_to_label.items()}

    region_ul_corner = region_ul_corner if len(region_ul_corner) == 2 else [0, 0]
    region_br_corner = region_br_corner if len(region_br_corner) == 2 else [tile_height - 1, tile_width - 1]

    # Samples selection
    dataset_x_vals, dataset_y_vals = [], []
    pixels_positions = []
    classes_occurences = [0 for _ in range(0, classes_num)]
    for i in range(region_ul_corner[0], region_br_corner[0] + 1):
        for j in range(region_ul_corner[1], region_br_corner[1] + 1):
            y_val_class_number = rgb_to_class_number[tuple(tile_gt_rgb[i][j])]
            if y_val_class_number in selected_classes:
                dataset_x_vals.append(tile[i][j])
                dataset_y_vals.append(class_number_to_label[y_val_class_number])
                pixels_positions.append((i, j))
                classes_occurences[selected_classes.index(y_val_class_number)] += 1

    # Print some information about the new dataset
    num_samples = len(dataset_x_vals)
    print('Number of samples selected: {}'.format(num_samples))
    for class_i, class_i_occurrences in zip(selected_classes, classes_occurences):
        print('\tclass {} samples: {}'.format(class_i, class_i_occurrences))

    # Save the dataset
    if svmlight_out_file is not None:
        svmlight_out_file = os.path.splitext(svmlight_out_file)[0] + '.txt'
        os.makedirs(os.path.dirname(svmlight_out_file), exist_ok=True)
    else:
        svmlight_out_file = '{}_t{:02d}{}_{}c_{}s.txt'.format(
            dataset_name, tile_number, '.{:02d}'.format(tile_second_index) if dataset_name == 'potsdam' else '',
            classes_num, num_samples
        )
    with open(svmlight_out_file, 'w') as svmlight_file:
        for x_vals, y_val in zip(dataset_x_vals, dataset_y_vals):
            svmlight_file.write('{} {}\n'.format(
                y_val, ' '.join([f'{i+1}:{format_without_zero(f)}' for i, f in enumerate(x_vals)]))
            )

    # Save the original pixels positions
    pixels_pos_dict = {
        'original_img_ul_corner': region_ul_corner,
        'original_img_br_corner': region_br_corner,
        'original_img_indices': pixels_positions
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
    parser = argparse.ArgumentParser(description='Script to convert a region of a tiff image taken from '
                                                 'SemCity-Toulouse or Potsdam into a svmlight format dataset.')
    parser.add_argument('--dataset-name', metavar='dataset_name', type=str, nargs='?', default='toulouse',
                        help='name of the input dataset, either \'toulouse\' or \'potsdam\'.')
    parser.add_argument('--tile-number', metavar='tile_number', type=int, nargs='?', default=-1,
                        help='number of the tile to convert.')
    parser.add_argument('--tile-second-index', metavar='tile_second_index', type=int, nargs='?', default=-1,
                        help='second index needed to identify the tile (only for potsdam).')
    parser.add_argument('--selected-classes', metavar='selected_classes', type=int, nargs='+', default=[],
                        help='classes to take into account.')
    parser.add_argument('--region-ul-corner', metavar='region_ul_corner', type=int, nargs='+', default=[],
                        help='coordinates of the upper left corner of the region to select.')
    parser.add_argument('--region-br-corner', metavar='region_br_corner', type=int, nargs='+', default=[],
                        help='coordinates of the bottom right corner of the region to select.')
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

    main(dataset_name, tile_number, tile_second_index, selected_classes, args.region_ul_corner, args.region_br_corner,
         args.new_labels, args.svmlight_out_file)
