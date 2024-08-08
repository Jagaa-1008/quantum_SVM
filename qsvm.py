import numpy as np
from itertools import combinations
import math
from collections import Counter
from itertools import product
import time 
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from minorminer import find_embedding
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from neal import SimulatedAnnealingSampler as SA
from dwave.system import DWaveSampler, FixedEmbeddingComposite, EmbeddingComposite
from MTQA import *

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
    def __init__(self, B=2, K=2, Xi=1, gamma=0.1, kernel='rbf', optimizer = SA, num_reads=1000, qubo_list = None, embeddings = None):
        self.B = B  # Base of the qubit representation (default: 2).
        self.K = K  # Number of qubits per alpha (default: 2).
        self.Xi = Xi  # Regularization parameter for the QUBO (default: 1).
        self.gamma = gamma  # Kernel coefficient for the RBF kernel (default: 0.1).
        self.C = np.sum(self.B ** (self.K)) # C = sum( B ** k)
        self.kernel = kernel # Kernel function ('rbf' for now, can be extended) (default: 'rbf').
        self.N = None # Number of data points
        self.optimizer = optimizer # QUBO optimizer
        self.num_reads = num_reads # Number of samples to generate during QUBO solving (default: 1000).
        
        # Attributes for storing model parameters and results
        self.support_vectors = None # Support vectors identified during training.
        self.alphas = None # Lagrange multipliers for the support vectors.
        self.support_labels = None # Labels of the support vectors.
        self.intercept = None # Intercept of the hyperplane.
        self.energy = None  # Energy of the best QUBO solution found during training.

        self._support_vectors = None
        self._n_support = None
        self._alphas = None
        self._support_labels = None
        self._indices = None
        self.intercept = None
        self.energy = None
        
        self.emb = embeddings
        self.qubo_list = qubo_list

    def rbf(self, x, y):
        # return np.exp(-self.gamma*(np.linalg.norm(x-y, ord=2)))
        return np.exp(-1.0 * self.gamma * np.dot(np.subtract(x, y).T, np.subtract(x, y)))

    def transform(self, X):
        K = np.zeros([X.shape[0], X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = self.rbf(X[i], X[j])
        return K
    
    def makeQUBO(self, data, label):
        N = len(data)

        Q = np.zeros((self.K*N,self.K*N))
        print(f'Creating the QUBO of size {Q.shape}')
        for n in range(N):
            for m in range(N):
                for k in range(self.K):
                    for j in range(self.K):
                        Q[self.K*n+k, self.K*m+j] = .5 * self.B**(k+j) * label[n] * label[m] * (self.rbf(data[n], data[m]) + self.Xi)
                        if n == m and k == j:
                            Q[self.K*n+k,self.K*m+j] += - self.B**k

        Q = np.triu(Q) + np.tril(Q,-1).T # turn the symmetric matrix into upper triangular
        
        return Q

    def MTQA_solve(self, energy, alpha_result, data, label):
        self.energy = energy
        self.alpha_result = alpha_result
        self.alpha_result = np.reshape(self.alpha_result,(self.K * self.N))

        K = self.transform(data)

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

        return self
    
    def solve(self, data, label, i=None):
        print("solving...")
        qubo = self.makeQUBO(data, label)

        if self.optimizer == "SA":
            sampleset = neal.SimulatedAnnealingSampler().sample_qubo(qubo, num_reads=5000)
            best_sample = sampleset.first
            
            self.energy = best_sample[1]
            self.alpha_result = list(best_sample[0].values())
            self.alpha_result = np.reshape(self.alpha_result,(self.K * self.N))
        
        elif self.optimizer == "QA":
            # ------- Set up D-Wave parameters -------
            token = 'DEV-3adb6333e41cfa2b8e54f63c2634a6fb2e333f71' #jagaa@t-ksj.co.jp
            endpoint = 'https://cloud.dwavesys.com/sapi/'
            dw_sampler = DWaveSampler(solver='Advantage_system6.4', token=token, endpoint=endpoint)

            if not self.qubo_list:
                hardware = nx.Graph(dw_sampler.edgelist)
                emb = find_embedding(qubo, hardware, tries=10, max_no_improvement=10, chainlength_patience=10, timeout=10, threads=100)
                sampler = FixedEmbeddingComposite(dw_sampler, embedding=emb)
                best_sample = sampler.sample_qubo(qubo, num_reads=4000, annealing_time = 20, label='QA_SVM').first

            else:
                from dwave.embedding.chain_breaks import MinimizeEnergy as me

                response = dw_sampler.sample_qubo(self.qubo_list[i], num_reads = 1000, annealing_time = 20, answer_mode = 'raw', auto_scale = False, label='QA_SVM')
                bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
                chain_break_method = me(bqm, self.emb[i])

                if not isinstance(response, dimod.SampleSet):
                    response = dimod.SampleSet.from_samples(response, energy=0, vartype=dimod.BINARY)

                best_sample = unembed_sampleset(response, self.emb[i], bqm, chain_break_method=chain_break_method).first
                
            self.energy = best_sample[1]
            self.alpha_result = list(best_sample[0].values())
            self.alpha_result = np.reshape(self.alpha_result,(self.K * self.N))

            if self.vis:
                m=16
                P16 = dnx.pegasus_graph(m)

                clr_dict = {}
                for i, j in emb.items():
                    clr_dict[i] = "Red"
                    
                # Draw the graph
                dnx.draw_pegasus_embedding(P16, emb, crosses=False, chain_color = clr_dict, unused_color = "#CCCCCC", width=0.3, node_size = 0.5)
                
                plt.show()
            
        else:
            print("This optimizer is not available")

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
    
def slice_dict(d, start, end):
    """Slices a dictionary from start index to end index."""
    keys = list(d.keys())
    sliced_keys = keys[start:end]
    return {key: d[key] for key in sliced_keys}

class MTQA_OneVsRestClassifier:
    def __init__(self, class_num, classifier, params=None, vis = 0):
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
            self.classifiers = [classifier() for _ in range(class_num)]
        else:
            self.classifiers = [classifier(**params) for _ in range(class_num)]

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
            q1 = self.classifiers[i].makeQUBO(x, self.re_labaling(y, i))
            qubolist.append(q1)

        embeddings, embedding_list, TotalQubo, Qubo_list, offset_list, Qubits, Qubo_logic = embedding_search(hardware, qubolist, chain_strategy="utc")

        sorted_dict = {k: embeddings[k] for k in sorted(embeddings)}
        identical_emb = []
        for i in range(self.class_num):
            identical_emb.append(slice_dict(sorted_dict, offset_list[i], offset_list[i+1]))

        # Solve all classifiers simultaneously with D-Wave mtqa
        response = dw_sampler.sample_qubo(TotalQubo, num_reads = 1000, annealing_time = 20, answer_mode = 'raw', auto_scale = False, label='mtqa_SVM_UTC')
        time.sleep(10)

        energy = []
        alpha = []
        comb_sol, unembed_t = unembed_combined_solution(response, identical_emb, Qubo_logic, offset_list, 'minimize_energy')
        for i in range(self.class_num):
            sol = []
            for i, sol_res in enumerate(comb_sol):
                xx, yy = offset_list[i], offset_list[i+1]
                re, decode_time = response_decoder(sol_res, xx, yy, Qubo_logic)
                if re:
                    sol += re
            
            energy.append(min(sol, key=lambda x: x[1])[1] if sol else None)
            alpha.append(min(sol, key=lambda x: x[1])[0] if sol else None)

        for i in range(self.class_num):
            q1 = self.classifiers[i].MTQA_solve(energy[i], alpha[i], x, self.re_labaling(y, i))

        if self.vis:
            m=16
            P16 = dnx.pegasus_graph(m)
            problems = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
            colors = ["Red", "Blue", "Green", "Yellow"]

            clr_dict = {}
            indx = 0
            for i, j in embeddings.items():
                if i >= offset_list[indx+1]:
                    indx += 1

                clr_dict[i] = colors[indx]
                    
            dnx.draw_pegasus_embedding(P16, embeddings, crosses=False, chain_color = clr_dict, unused_color = "#CCCCCC", width=0.3, node_size = 0.5)

            # Custom legend
            color_patch0 = mpatches.Patch(color=colors[0], label=problems[0])
            color_patch1 = mpatches.Patch(color=colors[1], label=problems[1])
            color_patch2 = mpatches.Patch(color=colors[2], label=problems[2])
            color_patch3 = mpatches.Patch(color=colors[3], label=problems[3])

            plt.legend(handles=[color_patch0, color_patch1, color_patch2, color_patch3], loc = 'upper left', bbox_to_anchor=(0.92,0.9))

            plt.savefig("results\digit_embeddings_0_3.png",  bbox_inches='tight', dpi = 1000)

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
