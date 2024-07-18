import argparse
import json
import numpy as np
import os
import pandas as pd
import tifffile as tiff


def main(labels_file, separator, mappings_file, pixels_pos_file, tiff_out_file):
    # Load the labels
    labels_df = pd.read_csv(labels_file, sep=separator, header=None)
    labels = [str(int(float_label)) for float_label in labels_df.iloc[:, 0].values]

    # Load the labels to RGB values mapping
    with open(mappings_file) as m_file:
        mappings = json.load(m_file)
    label_to_rgb = {
        label: mappings['class_number_to_rgb'][str(mappings['label_to_class_number'][label])]
        for label in mappings['label_to_class_number'].keys()
    }

    # Load the pixels positions
    with open(pixels_pos_file) as pp_file:
        pixels_pos_dict = json.load(pp_file)
    img_height = pixels_pos_dict['original_img_br_corner'][0] - pixels_pos_dict['original_img_ul_corner'][0] + 1
    img_width = pixels_pos_dict['original_img_br_corner'][1] - pixels_pos_dict['original_img_ul_corner'][1] + 1
    original_img_positions = pixels_pos_dict['original_img_indices']

    # Create the Numpy vector representing the image
    np_img = np.full(shape=(img_height, img_width, 3), fill_value=255, dtype=np.uint8)
    for label, original_img_position in zip(labels, original_img_positions):
        np_img_row = original_img_position[0] - pixels_pos_dict['original_img_ul_corner'][0]
        np_img_col = original_img_position[1] - pixels_pos_dict['original_img_ul_corner'][1]
        np_img[np_img_row][np_img_col] = label_to_rgb[label]

    # Save the output TIFF image
    os.makedirs(os.path.dirname(tiff_out_file), exist_ok=True)
    tiff.imwrite(tiff_out_file, np_img, resolution=(400, 400))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for plotting labels as a TIFF image.')
    parser.add_argument('--labels-file', metavar='labels_file', type=str, nargs='?', default=None,
                        help='CSV file with the labels in the first column.')
    parser.add_argument('--separator', metavar='separator', type=str, nargs='?', default=' ',
                        help='column separator (default value \' \').')
    parser.add_argument('--mappings-file', metavar='mappings-file', type=str, nargs='?', default=None,
                        help='JSON file containing the \'labels\' to \'RGB values\' mapping.')
    parser.add_argument('--pixels-pos-file', metavar='pixels_pos_file', type=str, nargs='?', default=None,
                        help='JSON file containing the position of the pixels labels in the image.')
    parser.add_argument('--tiff-out-file', metavar='tiff_out_file', type=str, nargs='?', default=None,
                        help='TIFF output file name (.tif extension).')
    args = parser.parse_args()

    if args.labels_file is not None and args.mappings_file is not None and args.pixels_pos_file is not None \
            and args.tiff_out_file is not None:
        main(args.labels_file, args.separator, args.mappings_file, args.pixels_pos_file, args.tiff_out_file)
