import numpy as np

from classical.crammer_singer_svm.CSSVM import CSSVM


class ClassicalSVM:
    def __init__(self, config):
        self.config = config

        self.single_class_label = None
        self.classical_svm_instance = None

    def train(self, training_data, training_labels, pickle_filepath):
        unique_labels = sorted(np.unique(training_labels))
        unique_labels_num = len(unique_labels)

        if unique_labels_num == 1:
            self.single_class_label = training_labels[0]
        else:
            if self.config['python_svm_type'] == 2:
                self.classical_svm_instance = CSSVM(self.config)
                self.classical_svm_instance.train(training_data, training_labels, unique_labels, pickle_filepath)

    def predict(self, test_instance, pickle_filepath):
        if self.single_class_label is not None:
            test_label = self.single_class_label
        else:
            test_label = self.classical_svm_instance.predict(test_instance, pickle_filepath)

        return test_label
