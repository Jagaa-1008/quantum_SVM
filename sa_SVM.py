import numpy as np
from sklearn.metrics import accuracy_score
from neal import SimulatedAnnealingSampler as SA

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
        if params is None:
            self.classifiers = [classifier() for _ in range(class_num)]
        else:
            self.classifiers = [classifier(**params) for _ in range(class_num)]

    def solve(self, x, y):
        """
        Train each binary classifier on the dataset.
        
        Parameters:
        - x (array): Feature data.
        - y (array): True labels.
        """
        for i in range(self.class_num):
            print(f"Training classifier {i}...")
            self.classifiers[i].solve(x, self.re_labaling(y, i), i)
        return self

    def re_labaling(self, y, pos_label):
        """
        Relabel the dataset for a binary classification task.
        
        Parameters:
        - y (array): Original labels.
        - pos_label (int): The positive class label for the current binary classifier.
        
        Returns:
        - Array of binary labels where the positive class is 1 and all others are -1.
        """
        return np.where(y == pos_label, 1, -1)

    def argmax_by_E(self, result):
        """
        Determine the class with the highest probability based on energy minimization.
        
        Parameters:
        - results (array): Results from all classifiers.
        
        Returns:
        - Array of predicted class labels.
        """
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if np.sum(result[:, j], axis=0) == -1: #case (1,-1,-1)
                    pass
                elif np.sum(result[:, j], axis=0) == 1: #case (1,1,-1)
                    a = np.array(np.where(result[:, j] == 1))[0][0]
                    b = np.array(np.where(result[:, j] == 1))[0][1]
                    if self.classifiers[a].energy > self.classifiers[b].energy:
                        result[a, j] = -1
                    else:
                        result[b, j] = -1
                elif np.sum(result[:, j], axis=0) == 3: #case (1,1,1)
                    min_e = np.argmin(np.array([self.classifiers[0].energy, self.classifiers[1].energy, self.classifiers[2].energy]))
                    result[0:min_e, j] = -1
                    result[min_e:i, j] = -1
                elif np.sum(result[:, j], axis=0) == -3: #case (-1,-1,-1)
                    min_e = np.argmin(
                        np.array([self.classifiers[0].energy, self.classifiers[1].energy, self.classifiers[2].energy]))
                    result[min_e, j] = 1

        # print("result", result)
        return np.argmax(result, axis=0)

    def predict(self, X):
        result = np.array([model.predict(X) for model in self.classifiers])
        # print("result123", result)
        #print("result type and shape", type(result), result.shape)
        #result type and shape <class 'numpy.ndarray'> (3, 150)
        return self.argmax_by_E(result)

    def evaluate(self, X, y):
        """
        Evaluate the classifier's performance on the given test data.
        
        Parameters:
        - X (array): Feature data.
        - y (array): True labels.
        
        Returns:
        - Accuracy of the classifier as a float.
        """
        pred = self.predict(X)
        print("pred result",pred)
        return accuracy_score(y, pred)

class qSVM():
    def __init__(self, data, label, B=2, K=2, Xi=1, gamma = 0.1, C=3, kernel="rbf", optimizer="SA"):
        """
        :param B:
        :param K:
        :param Xi:
        :param gamma:
        :param C:
        :param kernel: default; rbf only rbf for now,
        :param optimizer:SA
        """
        # self.data = data
        # self.label = label
        self.B = B
        self.K = K
        self.N = data.shape[0]
        self.Xi = Xi
        self.gamma = float(gamma)
        self.C = C
        self.kernel = kernel

        self.optimizer = optimizer
        self.alpha = None

        self.alpha_result = None
        self.alpha_result_array = None
        self.alpha_real = np.zeros((self.N,))

        self._support_vectors = None
        self._n_support = None
        self._alphas = None
        self._support_labels = None
        self._indices = None
        self.intercept = None
        self.energy = None


    def rbf(self, x, y):
        return np.exp(-1.0 * self.gamma * np.dot(np.subtract(x, y).T, np.subtract(x, y)))

    def transform(self, X):
        K = np.zeros([X.shape[0], X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = self.rbf(X[i], X[j])
        return K
    
    def makeQUBO(self, data, label):
        Q = np.zeros((self.K*self.N,self.K*self.N))
        print(f'Creating the QUBO of size {Q.shape}')
        for n in range(self.N):
            for m in range(self.N):
                for k in range(self.K):
                    for j in range(self.K):
                        Q[self.K*n+k,self.K*m+j] = .5 * self.B**(k+j) * label[n] * label[m] * (self.rbf(data[n], data[m]) + self.Xi)
                        if n == m and k == j:
                            Q[self.K*n+k,self.K*m+j] += - self.B**k

        Q = np.triu(Q) + np.tril(Q,-1).T # turn the symmetric matrix into upper triangular
        
        return Q
    
    def solve(self, data, label, i=None):
        print("solving...")
        qubo = self.makeQUBO(data, label)

        sampleset = SA().sample_qubo(qubo, num_reads=5000)
        best_sample = sampleset.first
        
        self.energy = best_sample[1]
        self.alpha_result = list(best_sample[0].values())
        self.alpha_result = np.reshape(self.alpha_result,(self.K * self.N))

        K = self.transform(data)

        print("K,N",self.K, self.N)
        for i in range(self.N):
            for j in range(self.K):
                self.alpha_real[i] += self.alpha_result[self.K*i+j] * self.B ** j

        is_sv = self.alpha_real > 1e-5
        # print("(self.alpha_real)", self.alpha_real)
        self._support_vectors = data[is_sv]
        self._n_support = np.sum(is_sv)
        self._alphas = self.alpha_real[is_sv]
        self._support_labels = label[is_sv]
        self._indices = np.arange(data.shape[0])[is_sv]  # the index of supported vector
        self.intercept = 0

        for i in range(self._alphas.shape[0]):
            self.intercept += self._support_labels[i]
            self.intercept -= np.sum(self._alphas * self._support_labels * K[self._indices[i], is_sv])
        self.intercept /= self._alphas.shape[0]
        print("self.intercept", self.intercept)

        return self.alpha_result

    def signum(self, X):
        return np.where(X > 0, 1, -1)

    def predict(self, X):
        score = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            score[i] = sum(self._alphas[j] * self._support_labels[j] * self.rbf(X[i], self._support_vectors[j])
                           for j in range(len(self._alphas)))
        return score + self.intercept
        # return np.sign(score + self.intercept)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)

