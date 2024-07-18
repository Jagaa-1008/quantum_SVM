import numpy as np
import os
import pickle
import random

from quantum.QuantumSVM import QuantumSVM
from classical.ClassicalSVM import ClassicalSVM


class PythonSVM:
    seeds_rng = random.Random(42)

    def __init__(self, python_svm_type, kernel_type, gamma, cost, base, binary_vars_num, penalty_coef, multicl_reg,
                 embeddings_dir):
        self.config = {
            'python_svm_type': python_svm_type,
            'kernel_type': kernel_type,
            'gamma': gamma,
            'cost': cost,
            'base': base,
            'binary_vars_num': binary_vars_num,
            'penalty_coef': penalty_coef,
            'multicl_reg': multicl_reg,
            'embeddings_dir': embeddings_dir
        }

        if self.config['python_svm_type'] in [0, 1]:
            self.svm_instance = QuantumSVM(self.config, seed=PythonSVM.seeds_rng.randint(0, 1000000000),
                                           max_model_size_enabled=True, data_based_model_selection=False)
        elif self.config['python_svm_type'] in [2]:
            self.svm_instance = ClassicalSVM(self.config)
        else:
            print(f'PythonSVM: unrecognized python_svm_type {self.config["python_svm_type"]}!')
            exit(-1)

        self.training_set = None
        self.pickle_filepath = None

    def train(self, training_set, pickle_filepath):
        self.training_set = training_set

        loaded_from_file = False
        if os.path.isfile(pickle_filepath):
            loaded_python_svm = PythonSVM.load_model(pickle_filepath)
            if loaded_python_svm.config == self.config and loaded_python_svm.training_set == self.training_set:
                self.svm_instance = loaded_python_svm.svm_instance
                self.pickle_filepath = pickle_filepath
                loaded_from_file = True

        if not loaded_from_file:
            np_training_set = np.array(training_set)
            training_data = np_training_set[:, 0:-1]
            training_labels = np_training_set[:, -1]

            self.svm_instance.train(training_data, training_labels, pickle_filepath=pickle_filepath)
            if pickle_filepath != '':
                self.save_model(pickle_filepath)
            else:
                self.pickle_filepath = ''

    def predict(self, test_instance):
        np_test_instance = np.array(test_instance)
        return self.svm_instance.predict(np_test_instance, pickle_filepath=self.pickle_filepath)

    def save_model(self, pickle_filepath):
        self.pickle_filepath = pickle_filepath

        pickle_dirname = os.path.dirname(pickle_filepath)
        if not os.path.exists(pickle_dirname):
            os.makedirs(pickle_dirname)

        with open(pickle_filepath, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @staticmethod
    def load_model(pickle_filepath):
        with open(pickle_filepath, 'rb') as pickle_file:
            python_svm_instance = pickle.load(pickle_file)
        return python_svm_instance

    def get_relative_filepath(self):
        if self.pickle_filepath is not None:
            return os.path.relpath(self.pickle_filepath, os.path.dirname(os.path.dirname(self.pickle_filepath)))
        else:
            return ''
