import numpy as np
import os
import subprocess

from sklearn.datasets import dump_svmlight_file


class CSSVM:
    @staticmethod
    def kernel(xm, xn, gamma=-1):
        if gamma == -1:
            return np.dot(xm, xn)
        return np.exp(-gamma * np.sum((xm - xn) ** 2))

    def __init__(self, config):
        self.kernel_type = config['kernel_type']
        self.gamma = -1 if self.kernel_type == 0 else config['gamma']
        self.beta = config['multicl_reg']
        self.C = 1.0 / self.beta

        self.executable_files_dir = '{}classical/crammer_singer_svm/CSSVM_lib/'.format(
            'Python_models/' if not os.getcwd().endswith('Python_models') else ''
        )
        self.tmp_dir = '/tmp/falk_svm_cs/'

        self.classes_mapping = {}

        self.file_input_training = True
        self.python_evaluation = False
        self.file_io_test = True

        self.pickle_filepath_no_extension = None
        self.model_filepath = None

        self.model = {}

    def train(self, training_data, training_labels, unique_labels, pickle_filepath):
        assert len(unique_labels) >= 2

        training_labels_mapped = training_labels
        if not all(y in list(range(1, len(unique_labels) + 1)) for y in unique_labels):
            training_labels_mapped = np.array([unique_labels.index(label) + 1 for label in training_labels])
            self.classes_mapping = {index + 1: label for index, label in enumerate(unique_labels)}

        if self.file_input_training or not self.python_evaluation:
            self.pickle_filepath_no_extension = os.path.splitext(pickle_filepath)[0]
            if self.pickle_filepath_no_extension == '':
                os.makedirs(self.tmp_dir, exist_ok=True)
                self.pickle_filepath_no_extension = self.tmp_dir + 'py_svm'
            else:
                os.makedirs(os.path.dirname(pickle_filepath), exist_ok=True)

        if self.file_input_training:
            training_data_filepath = self.pickle_filepath_no_extension + '_tr_data.txt'
            dump_svmlight_file(training_data, training_labels_mapped, f=training_data_filepath, zero_based=False)
        else:
            command_line_training_set = [
                '{} {}'.format(label, ' '.join([f'{index+1}:{feature}' for index, feature in enumerate(features)]))
                for features, label in zip(training_data, training_labels_mapped)
            ]
            training_data_filepath = 'command-line  {}  {}'.format(
                len(command_line_training_set), '  '.join(command_line_training_set)
            )

        if self.python_evaluation:
            self.model_filepath = 'console'
            self.model = {
                'labels': list(range(1, len(unique_labels) + 1)),
                'support_vectors': [[] for _ in range(len(unique_labels))],
                'coefficients': [[] for _ in range(len(unique_labels))],
                'b': 0
            }
        else:
            self.model_filepath = self.pickle_filepath_no_extension + '_cs_model.txt'

        exec_string = '{}svm_multiclass_learn  -c  {}  -t  {}  -g  {}  -v  0  {}  {}'.format(
            self.executable_files_dir, self.C, self.kernel_type, self.gamma, training_data_filepath,
            self.model_filepath
        )
        try:
            exec_out = subprocess.run(exec_string.split('  '), capture_output=True)
        except:
            raise Exception('An error occurred during the training step of the CS SVM!')

        if self.python_evaluation:
            lines = exec_out.stdout.decode("utf-8").strip().split('\n')

            self.model['b'] = float(lines[0])

            previous_line_id = ''
            for line in sorted(lines[1:]):
                line_id = line.rsplit(' ', 1)[0]
                line_split = line.split(' ')

                sv_label, sv_features, sv_coefficient = \
                    int(line_split[0]), np.array([float(x) for x in line_split[1: -1]]), float(line_split[-1])
                sv_label_index = sv_label - 1

                if line_id == previous_line_id:
                    self.model['coefficients'][sv_label_index][-1] += sv_coefficient
                else:
                    self.model['support_vectors'][sv_label_index].append(sv_features)
                    self.model['coefficients'][sv_label_index].append(sv_coefficient)
                    previous_line_id = line_id

        if self.file_input_training:
            os.remove(training_data_filepath)

    def predict(self, test_instance, current_pickle_filepath):
        if self.python_evaluation:
            scores = [
                sum([
                    coeff * CSSVM.kernel(sv, test_instance, gamma=self.gamma)
                    for sv, coeff in zip(self.model['support_vectors'][indx], self.model['coefficients'][indx])
                ]) - self.model['b']
                for indx in range(0, len(self.model['labels']))
            ]
            test_label = self.model['labels'][np.argmax(scores)]
        else:
            current_pickle_filepath_no_extension = os.path.splitext(current_pickle_filepath)[0]
            if current_pickle_filepath_no_extension not in ['', self.pickle_filepath_no_extension]:
                self.pickle_filepath_no_extension = current_pickle_filepath_no_extension
                self.model_filepath = self.pickle_filepath_no_extension + '_cs_model.txt'

            if self.file_io_test:
                test_data_filepath = self.pickle_filepath_no_extension + '_te_data.txt'
                dump_svmlight_file(np.array([test_instance]), [1], f=test_data_filepath, zero_based=False)

                prediction_filepath = self.pickle_filepath_no_extension + '_pred.txt'
            else:
                test_data_filepath = 'command-line  1  1 {}'.format(
                    ' '.join([f'{index+1}:{feature}' for index, feature in enumerate(test_instance)])
                )

                prediction_filepath = 'console'

            exec_string = '{}svm_multiclass_classify  -v  0  {}  {}  {}'.format(
                self.executable_files_dir, test_data_filepath, self.model_filepath, prediction_filepath
            )
            try:
                exec_out = subprocess.run(exec_string.split('  '), capture_output=True)
            except:
                raise Exception('An error occurred during the prediction step of the CS SVM!')

            if self.file_io_test:
                with open(prediction_filepath) as pred_file:
                    line = pred_file.readline()
                    test_label = int(line.split(' ')[0])

                os.remove(test_data_filepath)
                os.remove(prediction_filepath)
            else:
                test_label = int(exec_out.stdout.decode("utf-8").strip().split()[0])

        if len(self.classes_mapping) != 0:
            test_label = self.classes_mapping[test_label]

        return test_label
