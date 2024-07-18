import numpy as np
from pyqubo import Array, Binary, Placeholder, Constraint, solve_qubo
from itertools import combinations
from sklearn.metrics import accuracy_score
import math
import neal
from dwave.system import DWaveSampler, FixedEmbeddingComposite, EmbeddingComposite
from collections import Counter
from itertools import product
import time 
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
            self.classifiers[i].solve(x, self.re_labaling(y, i))
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
    
import networkx as nx
from minorminer import find_embedding

class qSVM():
    def __init__(self, data, label, B=2, K=2, Xi=1, gamma = 10, C=3, kernel="rbf", optimizer="SA", vis = 0):
        """
        :param B:
        :param K:
        :param Xi:
        :param gamma:
        :param C: #still not used now
        :param kernel: default; rbf only rbf for now,
        :param optimizer:SA,QA
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
        self.vis = vis

        self.options = {
            'SA': {},
            "QA": {}
        }

        self.optimizer = optimizer
        self.alpha = Array.create('alpha', shape=self.K * self.N, vartype='BINARY') #number of spins : K*N

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
        self.emb = None


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
        x = data
        t = label
        alpha = self.alpha
                
        energy = 0
        for n in range(self.N):
            for m in range(self.N):
                for k in range(self.K):
                    for j in range(self.K):
                        alpha[self.K * n +k] * alpha[self.K * m +j] * t[n] * t[m] * self.rbf(x[n],x[m]) * self.B ** (k+j)

        const_1=0
        for n in range(self.N):
            for k in range(self.K):
                const_1 += alpha[self.K * n +k] * self.B ** k

        const_2=0
        for n in range(self.N):
            for k in range(self.K):
                const_2 += alpha[self.K * n +k] * t[n] * self.B ** k

        const_2= const_2 ** 2

        h = 0.5 * energy - const_1 + self.Xi * const_2

        model = h.compile()
        qubo, offset = model.to_qubo()

        return qubo_conversion(qubo)
    
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

    def solve(self, data, label):
        print("solving...")
        qubo = self.makeQUBO(data, label)

        if self.optimizer == "SA":
            sampleset = neal.SimulatedAnnealingSampler().sample_qubo(qubo, num_reads=1000)
            best_sample = sampleset.first
            
            self.energy = best_sample[1]
            self.alpha_result = list(best_sample[0].values())
            self.alpha_result = np.reshape(self.alpha_result,(self.K * self.N))
        
        elif self.optimizer == "QA":
            # ------- Set up D-Wave parameters -------
            token = 'DEV-b3591d636174d4dcd3584a7ea29829d424e703b3' #happynice1008@gmail.com
            endpoint = 'https://cloud.dwavesys.com/sapi/'
            dw_sampler = DWaveSampler(solver='Advantage_system6.4', token=token, endpoint=endpoint)

            hardware = nx.Graph(dw_sampler.edgelist)
            emb = find_embedding(qubo, hardware, tries=10, max_no_improvement=10, chainlength_patience=10, timeout=10, threads=100)
            self.emb = emb
            sampler = FixedEmbeddingComposite(dw_sampler, embedding=emb)
            best_sample = sampler.sample_qubo(qubo, num_reads=1000, annealing_time = 20, label='qSVM').first

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
    
def qubo_conversion(qubo_dic):
    qubo_dict = {}
    for (key, value) in qubo_dic.items():
        # Extract numbers from the strings assuming format is 'alpha[n]'
        i_num = int(key[0].split('[')[1].split(']')[0])
        j_num = int(key[1].split('[')[1].split(']')[0])
        qubo_dict[(i_num, j_num)] = value

    sorted_qubo_dict = dict(sorted(qubo_dict.items()))

    return sorted_qubo_dict
    
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
        token = 'DEV-b3591d636174d4dcd3584a7ea29829d424e703b3' #happynice1008@gmail.com
        endpoint = 'https://cloud.dwavesys.com/sapi/'
        dw_sampler = DWaveSampler(solver='Advantage_system6.4', token=token, endpoint=endpoint)
        numr = 1000
        anneal_time = 20

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
        response = dw_sampler.sample_qubo(TotalQubo, num_reads = numr, annealing_time = anneal_time, answer_mode = 'raw', auto_scale = False, label='mtqa_SVM_UTC')
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

        # m=16
        # P16 = dnx.pegasus_graph(m)
        # problems = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
        # colors = ["Red", "Blue", "Green", "Yellow"]

        # clr_dict = {}
        # indx = 0
        # for i, j in embeddings.items():
        #     if i >= offset_list[indx+1]:
        #         indx += 1

        #     clr_dict[i] = colors[indx]
                
        # dnx.draw_pegasus_embedding(P16, embeddings, crosses=False, chain_color = clr_dict, unused_color = "#CCCCCC", width=0.3, node_size = 0.5)

        # # Custom legend
        # color_patch0 = mpatches.Patch(color=colors[0], label=problems[0])
        # color_patch1 = mpatches.Patch(color=colors[1], label=problems[1])
        # color_patch2 = mpatches.Patch(color=colors[2], label=problems[2])
        # color_patch3 = mpatches.Patch(color=colors[3], label=problems[2])

        # plt.legend(handles=[color_patch0, color_patch1, color_patch2, color_patch3], loc = 'upper left', bbox_to_anchor=(0.92,0.9))

        # plt.savefig("results\digit_embeddings_0_4.png",  bbox_inches='tight', dpi = 1000)

        # plt.show

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
