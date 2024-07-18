import numpy as np
import os

from quantum.quantum_binary_svm.QBSVM import QBSVM
from quantum.quantum_multiclass_svm.QMSVM import QMSVM


class QuantumSVM:
    def __init__(self, config, seed=None, max_model_size_enabled=False, data_based_model_selection=False):
        self.config = config
        self.seed = seed
        self.data_based_model_selection = data_based_model_selection

        if self.config['embeddings_dir'] != '' and (not os.path.exists(self.config['embeddings_dir'])):
            os.makedirs(self.config['embeddings_dir'])

        self.max_model_size_enabled = max_model_size_enabled
        self.qbsvm_max_model_size = 80 if self.max_model_size_enabled else 0
        self.qmsvm_max_model_size = 24 if self.max_model_size_enabled else 0

        self.single_class_label = None
        self.quantum_svm_instance = None

    def train(self, training_data, training_labels, pickle_filepath=None):
        unique_labels = sorted(np.unique(training_labels))
        unique_labels_num = len(unique_labels)

        if unique_labels_num == 1:
            self.single_class_label = training_labels[0]
        else:
            if (self.config['python_svm_type'] == 0 and not self.data_based_model_selection) or \
                    (self.data_based_model_selection and unique_labels_num == 2):
                self.quantum_svm_instance = QBSVM(self.config, seed=self.seed, max_model_size=self.qbsvm_max_model_size)
            elif (self.config['python_svm_type'] == 1 and not self.data_based_model_selection) or \
                    (self.data_based_model_selection and unique_labels_num > 2):
                self.quantum_svm_instance = QMSVM(self.config, seed=self.seed, max_model_size=self.qmsvm_max_model_size)
            else:
                print('The quantum SVM parameters (unique_labels_num={}, python_svm_type={}, data_based_model_select'
                      'ion={} do not match any valid configuration'.format(unique_labels_num,
                                                                           self.config['python_svm_type'],
                                                                           self.data_based_model_selection))
                exit(-1)

            self.quantum_svm_instance.train(training_data, training_labels, unique_labels)

    def predict(self, test_instance, pickle_filepath=None):
        if self.single_class_label is not None:
            test_label = self.single_class_label
        else:
            test_label = self.quantum_svm_instance.predict(test_instance)

        return test_label
