import argparse
import os

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler


def save_data_to_svmlight_file(filename, features, labels):
    with open(filename, 'w') as svmlight_file:
        for x_vals, y_val in zip(features, labels):
            svmlight_file.write('{} {}\n'.format(
                int(y_val) if y_val.is_integer() else y_val,
                ' '.join(['{}:{:.3f}'.format(i+1, f) for i, f in enumerate(x_vals)]))
            )


def main(training_data_file, test_data_file):
    sparse_training_dataset = load_svmlight_file(training_data_file)
    training_features, training_labels = sparse_training_dataset[0].toarray(), sparse_training_dataset[1]

    std_scaler = StandardScaler()
    std_scaler.fit(training_features)

    save_data_to_svmlight_file(os.path.splitext(training_data_file)[0]+'_standardized.txt',
                               std_scaler.transform(training_features), training_labels)

    if test_data_file is not None:
        sparse_test_dataset = load_svmlight_file(test_data_file)
        test_features, test_labels = sparse_test_dataset[0].toarray(), sparse_test_dataset[1]
        save_data_to_svmlight_file(os.path.splitext(test_data_file)[0]+'_standardized.txt',
                                   std_scaler.transform(test_features), test_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for standardizing libsvm datasets.')
    parser.add_argument('--training-data-file', metavar='training_data_file', type=str, nargs='?', default=None,
                        help='file containing the training data (libsvm format).')
    parser.add_argument('--test-data-file', metavar='test_data_file', type=str, nargs='?', default=None,
                        help='file containing the test data (libsvm format), including the labels.')
    args = parser.parse_args()

    if args.training_data_file is not None:
        main(args.training_data_file, args.test_data_file)
