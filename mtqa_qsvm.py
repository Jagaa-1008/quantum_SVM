import numpy as np
import time
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from minorminer import find_embedding
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from neal import SimulatedAnnealingSampler as SA
from dwave.embedding.chain_breaks import MinimizeEnergy as me

from MTQA import *
from solver_config import dwave_QA, dwave_MTQA

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
            self.classifiers[i].fit(x, self.re_labaling(y, i), i)
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
    def __init__(self, B=2, K=2, Xi=1, gamma=0.1, kernel='rbf', optimizer = {SA, dwave_QA, dwave_MTQA}, num_reads=1000, qubo_list = None, embeddings = None, annealing_time = 20, vis = None):
        self.B = B  # Base of the qubit representation (default: 2).
        self.K = K  # Number of qubits per alpha (default: 2).
        self.Xi = Xi  # Regularization parameter for the QUBO (default: 1).
        self.gamma = gamma  # Kernel coefficient for the RBF kernel (default: 0.1).
        self.C = np.sum(self.B ** (self.K)) # C = sum( B ** k)
        self.kernel = kernel # Kernel function ('rbf' for now, can be extended) (default: 'rbf').
        self.N = None # Number of data points
        self.optimizer = optimizer # QUBO optimizer
        self.num_reads = num_reads # Number of samples to generate during QUBO solving (default: 1000).
        self.annealing_time = annealing_time
        
        # Attributes for storing model parameters and results
        self.support_vectors = None # Support vectors identified during training.
        self.alphas = None # Lagrange multipliers for the support vectors.
        self.support_labels = None # Labels of the support vectors.
        self.intercept = None # Intercept of the hyperplane.
        self.energy = None  # Energy of the best QUBO solution found during training.
        
        self.emb = embeddings
        self.qubo_list = qubo_list
        self.vis = vis

    def rbf_kernel(self, X, Y):
        """Calculates the Radial Basis Function (RBF) kernel matrix."""
        XX = np.atleast_2d(X)
        YY = np.atleast_2d(Y)
        return np.exp(-self.gamma * np.sum((XX[:, np.newaxis] - YY[np.newaxis, :]) ** 2, axis=-1))
    
    def _make_qubo(self, X, y):
        """Constructs the Quadratic Unconstrained Binary Optimization (QUBO) problem."""
        self.N = X.shape[0]
        qubo = {}  # Use a more descriptive variable name

        for n in range(self.N):
            for m in range(n, self.N):  # Iterate upper triangle only
                for k in range(self.K):
                    for j in range(k, self.K):
                        coefficient = (
                            0.5 * self.B**(k + j) * y[n] * y[m] * (self.rbf_kernel(X[n], X[m]) + self.Xi)
                            - (self.B**k if n == m and k == j else 0)
                        )
                        
                        if coefficient != 0:  # Store only non-zero elements
                            qubo[(self.K * n + k, self.K * m + j)] = coefficient

        return qubo
        # Q = np.zeros((self.K * self.N, self.K * self.N))
        # print(f'Creating the QUBO of size {Q.shape}')
        # for n in range(self.N):
        #     for m in range(self.N):
        #         for k in range(self.K):
        #             for j in range(self.K):
        #                 Q[self.K * n + k, self.K * m + j] = (
        #                     0.5 * self.B**(k + j) * y[n] * y[m] * (self.rbf_kernel(X[n], X[m]) + self.Xi)
        #                     - (self.B**k if n == m and k == j else 0)
        #                 )
                        
        # Q = np.triu(Q) + np.tril(Q, -1).T  # nsure symmetry and upper triangular form
                
        # Q_dict = {}  # Initialize the QUBO dictionary
        # for n in range(self.N):
        #     for m in range(self.N):
        #         for k in range(self.K):
        #             for j in range(self.K):
        #                 if Q[self.K * n + k, self.K * m + j] != 0:
        #                     Q_dict[(self.K * n + k, self.K * m + j)] = Q[self.K * n + k, self.K * m + j]
        

    def MTQA_solve(self, binary_result, energy, X, y):
        alpha_real = self._decode(binary_result)
        self.energy = energy
        
        support_indices = (alpha_real >= 0) & (alpha_real <= self.C)
        # print("support_indices : ", support_indices)
        self.support_vectors = X[support_indices]
        self.alphas = alpha_real[support_indices]
        self.support_labels = y[support_indices]
        # print("self.support_vectors : ", self.support_vectors)
        # print("support labels : ", self.support_labels)
        
        self.intercept = self.b_offset(X, y)
        print("intercept : ", self.intercept)
        
        return self
    
    def _decode(self, binary_sol):
        """Decodes binary QUBO results into alpha values."""
        # print(binary_sol)
        Bvec = self.B ** np.arange(self.K)
        avec = np.array(binary_sol, float).reshape(self.N, self.K)
        alpha = avec @ Bvec
        # print("alphas : ", alpha)
        return alpha
        # return (np.fromiter(binary_sol, float).reshape(self.N, self.K) @ Bvec).flatten()

    def b_offset(self, X, y):
        """Evaluates the offset  value of b (intercept)"""

        if self.alphas is None or self.support_vectors is None:
            raise ValueError("The model has not been trained yet. Please call `fit` first.")
        
        K = self.rbf_kernel(X, self.support_vectors)
    
        # Calculate the average offset based on margin points
        b = np.sum(self.alphas * (self.C-self.alphas) * (self.support_labels - np.dot(K, self.alphas * self.support_labels))) / np.sum(self.alphas * (self.C-self.alphas))
        return b

        # # More robust offset calculation (Numerical Recipes approach)
        # # Sort f_values and corresponding labels
        # sorted_indices = np.argsort(f_values)
        # f_values_sorted = f_values[sorted_indices]
        # y_sorted = y[sorted_indices]

        # # Find candidate b values between consecutive f_values of opposite classes
        # class_changes = np.where(y_sorted[:-1] != y_sorted[1:])[0]

        # # Ensure at least two candidate intercepts
        # if len(class_changes) < 2:
        #     # If there's only one class change, add the average of the first and last f_values as a candidate
        #     b_candidates = [-(f_values_sorted[0] + f_values_sorted[-1]) / 2]
        # else:
        #     b_candidates = -(f_values_sorted[class_changes] + f_values_sorted[class_changes + 1]) / 2

        # # Choose the b that maximizes the number of correctly classified points
        # num_correct = [(y == self.signum(f_values + b)).sum() for b in b_candidates]
        # best_b_index = np.argmax(num_correct)
        # return b_candidates[best_b_index]
    
    def fit(self, X, y, i=None):
        """Trains the QSVM model by solving the QUBO and extracting support vectors."""
        # Input validation
        if not isinstance(X, np.ndarray):
            raise ValueError("Input features (X) must be a NumPy array.")
        if not isinstance(y, np.ndarray):
            raise ValueError("Target labels (y) must be a NumPy array.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        y = y.astype(int)  # Ensure integer labels for QUBO calculation
        if np.unique(y).tolist() != [-1, 1]:
            raise ValueError("Labels (y) must contain only -1 and 1 values.")

        qubo = self._make_qubo(X, y)
        if self.optimizer == SA:
            sampleset = self.optimizer().sample_qubo(qubo, num_reads=self.num_reads)
        elif self.optimizer == dwave_QA:
            # ------- Set up D-Wave parameters -------
            token = 'DEV-3adb6333e41cfa2b8e54f63c2634a6fb2e333f71' #jagaa@t-ksj.co.jp
            endpoint = 'https://cloud.dwavesys.com/sapi/'
            dw_sampler = DWaveSampler(solver='Advantage_system6.4', token=token, endpoint=endpoint)

            if not self.qubo_list:
                hardware = nx.Graph(dw_sampler.edgelist)
                emb = find_embedding(qubo, hardware, tries=3, max_no_improvement=3, chainlength_patience=10, timeout=5, threads=100)
                sampler = FixedEmbeddingComposite(dw_sampler, embedding=emb)
                sampleset = sampler.sample_qubo(qubo, num_reads=1000, annealing_time = 20, label='QA_SVM')
            else:
                response = self.optimizer().sample_qubo(self.qubo_list[i], num_reads = self.num_reads, annealing_time = self.annealing_time, answer_mode = 'raw', auto_scale = False, label='QA_SVM')
                bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
                chain_break_method = me(bqm, self.emb[i])

                if not isinstance(response, dimod.SampleSet):
                    response = dimod.SampleSet.from_samples(response, energy=0, vartype=dimod.BINARY)

                sampleset = unembed_sampleset(response, self.emb[i], bqm, chain_break_method=chain_break_method)
                
                if self.vis:
                    clr_dict = {}
                    for i, j in self.emb[i].items():
                        clr_dict[i] = "Red"
                        
                    dnx.draw_pegasus_embedding(dnx.pegasus_graph(16), self.emb[i], crosses=False, chain_color = clr_dict, unused_color = "#CCCCCC", width=0.3, node_size = 0.5)
                    
                    plt.show()
        else:
            raise ValueError("Selected solver is not supported.")
        
        # print(sampleset.record['sample'])
        sample = sampleset.record['sample'][0]
        self.energy = sampleset.first.energy

        alpha_real = self._decode(sample)
        
        support_indices = (alpha_real >= 0) & (alpha_real <= self.C)
        
        # print("support_indices : ", support_indices)
        self.support_vectors = X[support_indices]
        self.alphas = alpha_real[support_indices]
        self.support_labels = y[support_indices]
        # print("self.support_vectors : ", self.support_vectors)
        # print("support labels : ", self.support_labels)
        
        self.intercept = self.b_offset(X, y)
        print("intercept : ", self.intercept)

    def predict(self, X):
        """Predicts labels for new data points."""
        K = self.rbf_kernel(X, self.support_vectors)
        scores = np.dot(K, self.alphas * self.support_labels) + self.intercept
        return np.sign(scores)

    def evaluate(self, X, y):
        """Evaluates the model's accuracy, AUROC, and AUPRC on given data."""
        score = self.predict(X)
        accuracy = accuracy_score(y, score)
        auroc = roc_auc_score(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        auprc = auc(recall, precision)
        return accuracy, auroc, auprc
    
def slice_dict(d, start, end):
    """Slices a dictionary from start index to end index."""
    keys = list(d.keys())
    sliced_keys = keys[start:end]
    return {key: d[key] for key in sliced_keys}

from dwave.system import DWaveSampler, FixedEmbeddingComposite, EmbeddingComposite

class MTQA_OneVsRestClassifier:
    def __init__(self, class_num, classifier, params, vis = 0):
        """
        Initialize an ensemble of binary classifiers, one for each class.
        
        Parameters:
        - class_num (int): Number of classes.
        - classifier (class): Classifier class to be instantiated for each class.
        - params (dict, optional): Parameters to pass to each classifier instance.
        """
        
        self.vis = vis
        self.class_num = class_num
        if params is None:
            self.classifiers = [classifier() for _ in range(self.class_num)]
        else:
            self.classifiers = [classifier(**params) for _ in range(self.class_num)]

    def solve(self, x, y):
        """
        Train each binary classifier on the dataset on parallel.
        
        Parameters:
        - x (array): Feature data.
        - y (array): True labels.
        """
        
        # ------- Set up D-Wave parameters -------
        token = 'DEV-3adb6333e41cfa2b8e54f63c2634a6fb2e333f71' #jagaa@t-ksj.co.jp
        endpoint = 'https://cloud.dwavesys.com/sapi/'
        dw_sampler = DWaveSampler(solver='Advantage_system6.4', token=token, endpoint=endpoint)

        hardware = nx.Graph(dw_sampler.edgelist)

        qubolist = []
        for i in range(self.class_num):
            qubolist.append(self.classifiers[i]._make_qubo(x, self.re_labaling(y, i)))

        embeddings, embedding_list, TotalQubo, Qubo_list, offset_list, Qubits, Qubo_logic = embedding_search(hardware, qubolist, chain_strategy="utc")

        sorted_dict = {k: embeddings[k] for k in sorted(embeddings)}
        identical_emb = []
        for i in range(self.class_num):
            identical_emb.append(slice_dict(sorted_dict, offset_list[i], offset_list[i+1]))

        # Solve all classifiers simultaneously with D-Wave mtqa
        response = dw_sampler.sample_qubo(TotalQubo, num_reads = 1000, annealing_time = 20, answer_mode = 'raw', auto_scale = False, label='mtqa_SVM_UTC')
        time.sleep(10)

        energy = []
        binary_sol = []
        comb_sol, unembed_t = unembed_combined_solution(response, identical_emb, Qubo_logic, offset_list, 'minimize_energy')
        for i in range(self.class_num):
            sol = []
            for i, sol_res in enumerate(comb_sol):
                xx, yy = offset_list[i], offset_list[i+1]
                re, decode_time = response_decoder(sol_res, xx, yy, Qubo_logic)
                if re:
                    sol += re
            
            energy.append(min(sol, key=lambda x: x[1])[1] if sol else None)
            binary_sol.append(min(sol, key=lambda x: x[1])[0] if sol else None)

        for i in range(self.class_num):
            self.classifiers[i].MTQA_solve(binary_sol[i], energy[i], x, self.re_labaling(y, i))

        if self.vis:
            problems = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
            colors = ["Red", "Blue", "Green", "Yellow"]

            clr_dict = {}
            indx = 0
            for i, j in embeddings.items():
                if i >= offset_list[indx+1]:
                    indx += 1

                clr_dict[i] = colors[indx]
                    
            dnx.draw_pegasus_embedding(dnx.pegasus_graph(16), embeddings, crosses=False, chain_color = clr_dict, unused_color = "#CCCCCC", width=0.3, node_size = 0.5)

            # Custom legend
            color_patch0 = mpatches.Patch(color=colors[0], label=problems[0])
            color_patch1 = mpatches.Patch(color=colors[1], label=problems[1])
            color_patch2 = mpatches.Patch(color=colors[2], label=problems[2])
            color_patch3 = mpatches.Patch(color=colors[3], label=problems[3])

            plt.legend(handles=[color_patch0, color_patch1, color_patch2, color_patch3], loc = 'upper left', bbox_to_anchor=(0.92,0.9))

            plt.savefig("results\iris_embeddings_0_3.png",  bbox_inches='tight', dpi = 1000)

            plt.show

        return self, embedding_list, TotalQubo, Qubo_list

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
