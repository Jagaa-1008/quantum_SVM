import argparse
import os
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score

from PythonSVM import PythonSVM


def main(training_data_file, python_svm_type, kernel_type, gamma, cost, base, binary_vars_num, penalty_coef,
         multicl_reg, embeddings_dir, pickle_filepath, test_data_file, predictions_out_file):
    python_svm = None
    if training_data_file is not None:
        sparse_training_dataset = load_svmlight_file(training_data_file)
        training_dataset = np.c_[sparse_training_dataset[0].toarray(), sparse_training_dataset[1]]
        training_dataset = tuple(map(tuple, training_dataset))

        python_svm = PythonSVM(python_svm_type, kernel_type, gamma, cost, base, binary_vars_num, penalty_coef,
                               multicl_reg, embeddings_dir)
        python_svm.train(training_dataset, pickle_filepath)

    if test_data_file is not None:
        sparse_test_dataset = load_svmlight_file(test_data_file)
        test_dataset_np = np.c_[sparse_test_dataset[0].toarray(), sparse_test_dataset[1]]
        test_dataset = tuple(map(tuple, test_dataset_np))

        if python_svm is None:
            python_svm = PythonSVM.load_model(pickle_filepath)

        predictions = []
        for test_instance in test_dataset:
            predictions.append(python_svm.predict(test_instance[:-1]))

        if predictions_out_file is not None:
            os.makedirs(os.path.dirname(predictions_out_file), exist_ok=True)
            with open(predictions_out_file, 'w') as pred_out_file:
                pred_out_file.write('\n'.join([str(pred_label) for pred_label in predictions]) + '\n')
        else:
            print('Predicted labels: {}'.format(predictions))

        accuracy = accuracy_score(test_dataset_np[:, -1], predictions)
        print(f'Accuracy on {os.path.basename(test_data_file)} = {accuracy}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line main file for the execution of the Python SVM.')
    parser.add_argument('--training-data-file', metavar='training_data_file', type=str, nargs='?', default=None,
                        help='file containing the training data (libsvm format).')
    parser.add_argument('--python-svm-type', metavar='python_svm_type', type=int, nargs='?', default=0,
                        help='type of Python SVM, allowed values: 0 (quantum for binary classification), '
                             '1 (quantum for multiclass classification), 2 (classical for multiclass classification, '
                             'Crammer-Singer).')
    parser.add_argument('--kernel-type', metavar='kernel_type', type=int, nargs='?', default=2,
                        help='kernel for the Python SVM, allowed values: 0 (linear), 2 (rbf).')
    parser.add_argument('--gamma', metavar='gamma', type=float, nargs='?', default=1.0,
                        help='gamma value for the rbf kernel, set this value to -1 when using the linear kernel.')
    parser.add_argument('--cost', metavar='cost', type=float, nargs='?', default=1.0,
                        help='cost (C) value for the Python SVM.')
    parser.add_argument('--base', metavar='base', type=int, nargs='?', default=2,
                        help='base used to encode the coefficients of the Python quantum SVMs.')
    parser.add_argument('--binary-vars-num', metavar='binary_vars_num', type=int, nargs='?', default=2,
                        help='number of binary variables used to encode each coefficient of the Python quantum SVMs.')
    parser.add_argument('--penalty-coef', metavar='penalty_coef', type=float, nargs='?', default=1.0,
                        help='multiplier for penalty terms (ksi, mu) in Python quantum SVMs.')
    parser.add_argument('--multicl-reg', metavar='multicl_reg', type=float, nargs='?', default=1.0,
                        help='regularization parameter for the multiclass Python quantum SVM.')
    parser.add_argument('--embeddings-dir', metavar='embeddings_dir', type=str, nargs='?', default='',
                        help='directory where to store/load embeddings for the Python quantum SVMs.')
    parser.add_argument('--pickle-filepath', metavar='pickle_filepath', type=str, nargs='?', default='',
                        help='path of the file where to store/load the Python SVM instance.')
    parser.add_argument('--test-data-file', metavar='test_data_file', type=str, nargs='?', default=None,
                        help='file containing the test data (libsvm format), including the labels.')
    parser.add_argument('--predictions-out-file', metavar='predictions_out_file', type=str, nargs='?', default=None,
                        help='path of the (CSV) file where to store the label predictions.')
    args = parser.parse_args()
    
    if args.training_data_file is not None or (args.pickle_filepath != '' and args.test_data_file is not None):
        main(args.training_data_file, args.python_svm_type, args.kernel_type, args.gamma, args.cost, args.base, 
             args.binary_vars_num, args.penalty_coef, args.multicl_reg, args.embeddings_dir, args.pickle_filepath,
             args.test_data_file, args.predictions_out_file)
