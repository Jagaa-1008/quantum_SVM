import argparse
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


def main(expected_labels_file, predicted_labels_file, separator, json_out_file):
    # Load the expected labels
    exp_labels_df = pd.read_csv(expected_labels_file, sep=separator, header=None)
    exp_labels = exp_labels_df.iloc[:, 0].values.astype(int)

    # Load the predicted labels
    pred_labels_df = pd.read_csv(predicted_labels_file, sep=separator, header=None)
    pred_labels = pred_labels_df.iloc[:, 0].values.astype(int)

    # Determine the list of unique labels
    unique_labels = np.unique(exp_labels)
    unique_labels.sort()

    # Compute the performance metrics
    metrics_dict = dict()
    metrics_dict['class_labels'] = unique_labels.tolist()

    metrics_dict['accuracy'] = accuracy_score(exp_labels, pred_labels)
    metrics_dict['balanced_accuracy'] = balanced_accuracy_score(exp_labels, pred_labels)

    metrics_dict['precision'] = precision_score(exp_labels, pred_labels, labels=unique_labels, average=None).tolist()
    metrics_dict['avg_precision'] = np.mean(metrics_dict['precision']).item()

    metrics_dict['recall'] = recall_score(exp_labels, pred_labels, labels=unique_labels, average=None).tolist()
    metrics_dict['avg_recall'] = np.mean(metrics_dict['recall']).item()

    metrics_dict['f1_score'] = f1_score(exp_labels, pred_labels, labels=unique_labels, average=None).tolist()
    metrics_dict['avg_f1_score'] = np.mean(metrics_dict['f1_score']).item()

    # Save the results in the output file
    with open(json_out_file, 'w') as json_file:
        json.dump(metrics_dict, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for computing performance metrics on experiment results.')
    parser.add_argument('--expected-labels-file', metavar='expected_labels_file', type=str, nargs='?', default=None,
                        help='CSV file with the expected labels in the first column.')
    parser.add_argument('--predicted-labels-file', metavar='predicted_labels_file', type=str, nargs='?', default=None,
                        help='CSV file with the predicted labels in the first column.')
    parser.add_argument('--separator', metavar='separator', type=str, nargs='?', default=' ',
                        help='column separator (default value \' \').')
    parser.add_argument('--json-out-file', metavar='json_out_file', type=str, nargs='?', default=None,
                        help='JSON output file name (.json extension).')
    args = parser.parse_args()

    if args.expected_labels_file is not None and args.predicted_labels_file is not None \
            and args.json_out_file is not None:
        main(args.expected_labels_file, args.predicted_labels_file, args.separator, args.json_out_file)
