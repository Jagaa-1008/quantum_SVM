import numpy as np
from pyqubo import Array, solve_qubo
import math
import neal
from dwave.system import DWaveSampler, FixedEmbeddingComposite, EmbeddingComposite
from itertools import product
import matplotlib.pyplot as plt

class qSVM:
    def __init__(self, data, label, B=2, K=2, Xi=1, gamma=10, C=3, kernel="rbf", optimizer="SA"):
        self.label = label
        self.B = B
        self.K = K
        self.N = data.shape[0]
        self.Xi = Xi
        self.gamma = gamma
        self.C = C
        self.kernel = kernel
        self.optimizer = optimizer
        self.alpha = Array.create('alpha', shape=self.K * self.N, vartype='BINARY')

    def rbf(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x-y)**2)

    def transform(self, X):
        K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = self.rbf(X[i], X[j])
        return K

    def makeQUBO(self, data, label):
        energy = sum(self.alpha[self.K * i + k] * self.alpha[self.K * j + l] * label[i] * label[j] * self.rbf(data[i], data[j]) * self.B ** (k+l)
                     for i, j in product(range(self.N), repeat=2) for k, l in product(range(self.K), repeat=2))
        const_1 = sum(self.alpha[self.K * i + k] * self.B ** k for i in range(self.N) for k in range(self.K))
        const_2 = sum(self.alpha[self.K * i + k] * label[i] * self.B ** k for i in range(self.N) for k in range(self.K)) ** 2
        h = 0.5 * energy - const_1 + self.Xi * const_2
        model = h.compile()
        return model.to_qubo()

    def solve(self, data, label):
        qubo, offset = self.makeQUBO(data, label)
        if self.optimizer == "SA":
            sampler = neal.SimulatedAnnealingSampler()
            sample_set = sampler.sample_qubo(qubo, num_reads=100)
        elif self.optimizer == "QA":
            sampler = EmbeddingComposite(DWaveSampler())
            sample_set = sampler.sample_qubo(qubo, num_reads=100)
        else:
            raise ValueError("Unsupported optimizer type")
        self.process_solution(sample_set)

    def process_solution(self, sample_set):
        best_sample = sample_set.first.sample
        self.alpha_result = np.array([best_sample[f'alpha[{i}]'] for i in range(self.K * self.N)])
        print("Solution process complete.")

    def predict(self, X):
        K = self.transform(X)
        scores = np.dot(K, self.alpha_result * self.label) + self.intercept
        return scores

    def evaluate(self, X, y):
        predictions = np.sign(self.predict(X))
        accuracy = np.mean(predictions == y)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        return accuracy

import numpy as np
from sklearn.metrics import accuracy_score

class OneVsRestClassifier:
    def __init__(self, class_num, classifier, params=None):
        """
        Initialize an ensemble of binary classifiers, one for each class.
        
        Parameters:
        - class_num (int): Number of classes.
        - classifier (class): Classifier class to be instantiated for each class.
        - params (dict, optional): Parameters to pass to each classifier instance.
        """
        self.class_num = class_num
        self.classifiers = [classifier(**params) for _ in range(class_num)] if params else [classifier() for _ in range(class_num)]

    def re_label(self, y, pos_label):
        """
        Relabel the dataset for a binary classification task.
        
        Parameters:
        - y (array): Original labels.
        - pos_label (int): The positive class label for the current binary classifier.
        
        Returns:
        - Array of binary labels where the positive class is 1 and all others are -1.
        """
        return np.where(y == pos_label, 1, -1)

    def solve(self, x, y):
        """
        Train each binary classifier on the dataset.
        
        Parameters:
        - x (array): Feature data.
        - y (array): True labels.
        """
        for i in range(self.class_num):
            print(f"Training classifier for class {i}...")
            binary_labels = self.re_label(y, i)
            self.classifiers[i].solve(x, binary_labels)
        print("All classifiers trained.")

    def argmax_by_energy(self, results):
        """
        Determine the class with the highest probability based on energy minimization.
        
        Parameters:
        - results (array): Results from all classifiers.
        
        Returns:
        - Array of predicted class labels.
        """
        energies = np.array([clf.energy for clf in self.classifiers])
        return np.argmin(energies, axis=0)

    # def predict(self, X):
    #     """
    #     Predict the class labels for the provided data.
        
    #     Parameters:
    #     - X (array): Feature data to predict.
        
    #     Returns:
    #     - Predicted class labels.
    #     """
    #     results = np.array([clf.predict(X) for clf in self.classifiers])
    #     return self.argmax_by_energy(results)

    def predict(self, X):
        K = self.transform(X)
        # Assuming `self.alpha_result` is an array of coefficients for the support vectors:
        scores = np.dot(K, self.alpha_result) + self.intercept  # Using the dot product with the kernel matrix
        return scores


    def evaluate(self, X, y):
        """
        Evaluate the classifier's performance on the given test data.
        
        Parameters:
        - X (array): Feature data.
        - y (array): True labels.
        
        Returns:
        - Accuracy of the classifier as a float.
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        return accuracy

