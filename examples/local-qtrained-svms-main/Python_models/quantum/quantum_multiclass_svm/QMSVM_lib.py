import os
import re
import glob
import json
import gc
import time
import shutil
import numpy as np
import numpy.lib.recfunctions as rfn
from dwave.system.samplers import DWaveSampler, DWaveCliqueSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.composites import FixedEmbeddingComposite, LazyFixedEmbeddingComposite
from minorminer.busclique import find_clique_embedding
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from dwave.system import LeapHybridSampler
from hybrid.reference.kerberos import KerberosSampler


from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, auc
from sklearn.base import BaseEstimator, ClassifierMixin

from dimod import BinaryQuadraticModel

import matplotlib.pyplot as plt

from functools import reduce

from importlib import reload


## QUANTUM BINARY SVM

def kernel(xn, xm, gamma=-1):  # here (xn.shape: NxD, xm.shape: ...xD) -> Nx...
    if gamma == -1:
        return xn @ xm.T
    xn = np.atleast_2d(xn)
    xm = np.atleast_2d(xm)
    return np.exp(-gamma * np.sum((xn[:, None] - xm[None, :]) ** 2,
                                  axis=-1))  # (N,1,D) - (1,...,D) -> (N,...,D) -> (N,...); see Hsu guide.pdf for formula


def eval_classifier(x, alphas, data, label, gamma,
                    b=0):  # evaluates the distance to the hyper plane according to 16.5.32 on p. 891 (Numerical Recipes); sign is the assigned class; x.shape = ...xD
    return np.sum((alphas * label)[:, None] * kernel(data, x, gamma), axis=0) + b


def eval_acc_auroc_auprc(label, predicted):  # score is the distance to the hyper plane (output from eval_classifier)
    precision, recall, _ = precision_recall_curve(label, predicted)
    return accuracy_score(label, predicted), roc_auc_score(label, predicted), auc(recall, precision)


def eval_offset_avg(alphas, data, label, gamma, C, useavgforb=True):  # evaluates offset b according to 16.5.33
    cross = eval_classifier(data, alphas, data, label,
                            gamma)  # cross[i] = sum_j aj yj K(xj, xi) (error in Numerical Recipes)
    if useavgforb:
        return np.sum(alphas * (C - alphas) * (label - cross)) / np.sum(alphas * (C - alphas))
    else:  # this is actually not used, but we did a similar-in-spirit implementation in eval_finaltraining_avgscore.py
        if np.isclose(np.sum(alphas * (C - alphas)), 0):
            print('no support vectors found, discarding this classifier')
            return np.nan
        bcandidates = [np.sum(alphas * (C - alphas) * (label - cross)) / np.sum(
            alphas * (C - alphas))]  # average according to NR should be the first candidate
        crosssorted = np.sort(cross)
        crosscandidates = -(crosssorted[1:] + crosssorted[
                                              :-1]) / 2  # each value between f(xi) and the next higher f(xj) is a candidate
        bcandidates += sorted(crosscandidates,
                              key=lambda x: abs(x - bcandidates[0]))  # try candidates closest to the average first
        bnumcorrect = [(label == np.sign(cross + b)).sum() for b in bcandidates]
        return bcandidates[np.argmax(bnumcorrect)]


# decode binary -> alpha
def decode(binary, B=10, K=3):
    N = len(binary) // K
    Bvec = B ** np.arange(K)
    return np.fromiter(binary, float).reshape(N, K) @ Bvec


def gen_svm_qubos(X, Y, B, K, xi, gamma):
    N = len(X)

    Q = np.zeros((K * N, K * N))
    print(f'Creating the QUBO Q matrix of size {Q.shape}')
    for n in range(N):  # not optimized: size will not be so large and this way its more easily verifyable
        for m in range(N):
            for k in range(K):
                for j in range(K):
                    Q[K * n + k, K * m + j] = .5 * B ** (k + j) * Y[n] * Y[m] * (
                            kernel(X[n], X[m], gamma) + xi)
                    if n == m and k == j:
                        Q[K * n + k, K * m + j] += - B ** k

    Q = np.triu(Q) + np.tril(Q, -1).T  # turn the symmetric matrix into upper triangular

    return Q


def dwave_run(Q, B, K):
    MAXRESULTS = 20  # NOTE: to save space only 20 best results

    print('Extracting nodes and couplers from Q')
    qubo_couplers = np.asarray(
        [[n, m, Q[n, m]] for n in range(len(Q)) for m in range(n + 1, len(Q)) if not np.isclose(Q[n, m], 0)])

    print(f'{np.shape(qubo_couplers)}')
    qubo_couplers = qubo_couplers[np.argsort(-np.abs(qubo_couplers[:, 2]))]

    qubo_nodes = np.asarray(
        [[n, n, Q[n, n]] for n in range(len(Q))])  # if not np.isclose(Q[n,n],0)]) NOTE: removed due to variable order!

    # qubo_nodes = np.array([[i, i, (qubo_nodes[qubo_nodes[:, 0] == i, 2][0] if i in qubo_nodes[:, 0] else 0.)] for i in
    #                       np.arange(np.concatenate((qubo_nodes, qubo_couplers))[:, [0,
    #                                                                                 1]].max() + 1)])  # to make sure every (i,i) occurs in the qubo in increasing order such that the variable order in BinaryQuadraticModel is consistent (see locate wrongenergies-* github issue)

    print(f'The problem has {len(qubo_nodes)} nodes and {len(qubo_couplers)} couplers')

    maxcouplers = len(qubo_couplers)  # POSSIBLE INPUT if len(sys.argv) <= 2 else int(sys.argv[2])
    couplerslist = [maxcouplers]

    # Are these values still valid for the new QA advantage system with PEGASUS ????
    for trycouplers in [2500, 2000, 1800, 1600, 1400, 1200, 1000, 500]:
        if maxcouplers > trycouplers:
            couplerslist += [trycouplers]

    sampler = EmbeddingComposite(DWaveSampler())
    for couplers in couplerslist:  # try to reduce as little couplers as necessary to find an embedding
        Q = {(q[0], q[1]): q[2] for q in np.vstack((qubo_nodes, qubo_couplers[:couplers]))}

        print(f'Running with {len(qubo_nodes)} nodes and {couplers} couplers')

        ordering = np.array(list(BinaryQuadraticModel.from_qubo(Q)))
        if not (ordering == np.arange(len(ordering), dtype=ordering.dtype)).all():
            print(f'WARNING: variables are not correctly ordered! ordering={ordering}')

        try:
            response = sampler.sample_qubo(Q,
                                           num_reads=10000)  # maybe some more postprocessing can be specified here ...

        except ValueError as v:
            print(f' -- no embedding found, trying less couplers')
            continue
        break

    samples = np.array([''.join(map(str, sample)) for sample in response.record[
        'sample']])  # NOTE: it would be safer to use the labeling from record.data() for the qubit variable order
    unique_samples, unique_idx, unique_counts = np.unique(samples, return_index=True,
                                                          return_counts=True)  # unfortunately, num_occurrences seems not to be added up after unembedding
    unique_records = response.record[unique_idx]
    result = rfn.merge_arrays(
        (unique_samples, unique_records['energy'], unique_counts, unique_records['chain_break_fraction']))
    result = result[np.argsort(result['f1'])]
    # np.savetxt(pathsub + 'result.dat', result[:MAXRESULTS], fmt='%s', delimiter='\t',
    #           header='\t'.join(response.record.dtype.names),
    #           comments='')  # load with np.genfromtxt(..., dtype=['<U2000',float,int,float], names=True, encoding=None)

    alphas = np.array([decode(sample, B, K) for sample in result['f0'][:MAXRESULTS]])

    print(f'Running completed.')

    return alphas


def predict(X_test, X_train, Y_train, alphas, B, K, gamma):
    C = (B ** np.arange(K)).sum()

    # Compute the mean of the alphas
    alphas_avg = np.mean(alphas, axis=0)

    b = eval_offset_avg(alphas_avg, X_train, Y_train, gamma,
                        C)  # NOTE: this is NAN if no support vectors were found, see TODO file

    scoretest = eval_classifier(X_test, alphas_avg, X_train, Y_train, gamma, b)

    return scoretest


# Dennis Willsch
# Modified by Gabriele Cavallaro
# Function to plot the data and the decision boundaries for the classical SVM
def plot_svm(data, label, model, xylim=[-0.7, 1.7], notitle=False, filled=False,
             plot_support=True):
    result = np.sign = model.predict(data)
    # result = np.sign(eval_classifier(data, alphas, data, label, gamma, b))

    xsample = np.arange(xylim[0], xylim[1] + .01, .05)
    x1grid, x2grid = np.meshgrid(xsample, xsample)
    X = np.vstack((x1grid.ravel(), x2grid.ravel())).T  # ...xD for kernel contraction

    # FX = eval_classifier(X, alphas, data, label, gamma, b).reshape(len(xsample), len(xsample))
    FX = model.decision_function(X).reshape(len(xsample), len(xsample))

    plt.pcolor(x1grid, x2grid, FX, cmap='coolwarm')
    plt.contour(x1grid, x2grid, FX, [0.], linewidths=3, colors='black')

    if not filled:
        plt.scatter(data[result == 1][:, 0], data[result == 1][:, 1], c='r', marker=(8, 2, 0), linewidth=0.5)
        plt.scatter(data[result != 1][:, 0], data[result != 1][:, 1], c='b', marker='+', linewidth=1)
        plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], edgecolors='r', marker='s', linewidth=0.5,
                    facecolors='none')
        plt.scatter(data[label != 1][:, 0], data[label != 1][:, 1], edgecolors='b', marker='o', linewidth=1,
                    facecolors='none')
    else:
        plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], edgecolors='r', marker='s', linewidth=1,
                    facecolors='r')
        plt.scatter(data[label != 1][:, 0], data[label != 1][:, 1], edgecolors='b', marker='o', linewidth=1,
                    facecolors='b')

    # plot support vectors
    if plot_support:
        # support_vectors = data[np.nonzero(alphas)[0]]
        support_vectors = model.support_vectors_
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], edgecolors='green', facecolors='none', s=300,
                    linewidth=1)

    if not notitle:
        support_vectors = model.support_vectors_
        plt.title(
            str(len(support_vectors)) + ' SVs, ' + str(round(((result == label).sum() / len(label)), 2)) + ' accuracy')
        # plt.title('acc = %g, %d SVs' % ((result == label).sum() / len(label)))

    plt.xlim(*xylim)
    plt.ylim(*xylim)


# Dennis Willsch
# Modified by Gabriele Cavallaro
# Function to plot the data and the decision boundaries for the QA SVM
def plot_qa_svm(alphas, data, label, gamma, B, K, xylim=[-0.7, 1.7], notitle=False, filled=False,
                plot_support=True):
    C = (B ** np.arange(K)).sum()
    b = eval_offset_avg(alphas, data, label, gamma, C)
    result = np.copysign(1, eval_classifier(data, alphas, data, label, gamma, b))

    xsample = np.arange(xylim[0], xylim[1] + .01, .05)
    x1grid, x2grid = np.meshgrid(xsample, xsample)
    X = np.vstack((x1grid.ravel(), x2grid.ravel())).T  # ...xD for kernel contraction
    FX = eval_classifier(X, alphas, data, label, gamma, b).reshape(len(xsample), len(xsample))

    plt.pcolor(x1grid, x2grid, FX, cmap='coolwarm')
    plt.contour(x1grid, x2grid, FX, [0.], linewidths=3, colors='black')

    if not filled:
        plt.scatter(data[result == 1][:, 0], data[result == 1][:, 1], c='r', marker=(8, 2, 0), linewidth=0.5)
        plt.scatter(data[result != 1][:, 0], data[result != 1][:, 1], c='b', marker='+', linewidth=1)
        plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], edgecolors='r', marker='s', linewidth=0.5,
                    facecolors='none')
        plt.scatter(data[label != 1][:, 0], data[label != 1][:, 1], edgecolors='b', marker='o', linewidth=1,
                    facecolors='none')
    else:
        plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], edgecolors='r', marker='s', linewidth=1,
                    facecolors='r')
        plt.scatter(data[label != 1][:, 0], data[label != 1][:, 1], edgecolors='b', marker='o', linewidth=1,
                    facecolors='b')

    # plot support vectors
    if plot_support:
        support_vectors = data[np.nonzero(alphas)[0]]
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], edgecolors='green', facecolors='none', s=300,
                    linewidth=1)

    if not notitle:
        support_vectors = data[np.nonzero(alphas)[0]]
        plt.title(
            str(len(support_vectors)) + ' SVs, ' + str(round(((result == label).sum() / len(label)), 2)) + ' accuracy')
        # plt.title('acc = %g, %d SVs' % ((result == label).sum() / len(label)))

    plt.xlim(*xylim)
    plt.ylim(*xylim)


# Dennis Willsch
# Modified by Gabriele Cavallaro
# Function to plot the data and the decision boundaries for the QA SVM
def plot_qa_svm_test(alphas, X_test, Y_test, X_train, Y_train, gamma, B, K, xylim=[-0.7, 1.7], notitle=False,
                     filled=False,
                     plot_support=True):
    C = (B ** np.arange(K)).sum()
    b = eval_offset_avg(alphas, X_train, Y_train, gamma, C)
    Y_predict = np.copysign(1, eval_classifier(X_test, alphas, X_train, Y_train, gamma, b))

    xsample = np.arange(xylim[0], xylim[1] + .01, .05)
    x1grid, x2grid = np.meshgrid(xsample, xsample)
    X = np.vstack((x1grid.ravel(), x2grid.ravel())).T  # ...xD for kernel contraction
    FX = eval_classifier(X, alphas, X_train, Y_train, gamma, b).reshape(len(xsample), len(xsample))

    plt.pcolor(x1grid, x2grid, FX, cmap='coolwarm')
    plt.contour(x1grid, x2grid, FX, [0.], linewidths=3, colors='black')

    plt.scatter(X_test[Y_test == 1][:, 0], X_test[Y_test == 1][:, 1], edgecolors='r', marker='s', linewidth=1,
                facecolors='r')
    plt.scatter(X_test[Y_test != 1][:, 0], X_test[Y_test != 1][:, 1], edgecolors='b', marker='o', linewidth=1,
                facecolors='b')

    val_acc_qa, val_auroc_qa, val_auprc_qa = eval_acc_auroc_auprc(Y_test, Y_predict)
    plt.title('Test accuracy ' + str(round(val_acc_qa, 2)))

    '''
    if not filled:
        plt.scatter(data[result == 1][:, 0], data[result == 1][:, 1], c='r', marker=(8, 2, 0), linewidth=0.5)
        plt.scatter(data[result != 1][:, 0], data[result != 1][:, 1], c='b', marker='+', linewidth=1)
        plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], edgecolors='r', marker='s', linewidth=0.5,
                    facecolors='none')
        plt.scatter(data[label != 1][:, 0], data[label != 1][:, 1], edgecolors='b', marker='o', linewidth=1,
                    facecolors='none')
    else:
        plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], edgecolors='r', marker='s', linewidth=1,
                    facecolors='r')
        plt.scatter(data[label != 1][:, 0], data[label != 1][:, 1], edgecolors='b', marker='o', linewidth=1,
                    facecolors='b')

    # plot support vectors
    if plot_support:
        support_vectors = data[np.nonzero(alphas)[0]]
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], edgecolors='green', facecolors='none', s=300,
                    linewidth=1)

    if not notitle:
        support_vectors = data[np.nonzero(alphas)[0]]
        plt.title(str(len(support_vectors)) + ' SVs, ' + str(round(((result == label).sum() / len(label)), 2)) + ' accuracy')
        # plt.title('acc = %g, %d SVs' % ((result == label).sum() / len(label)))

    plt.xlim(*xylim)
    plt.ylim(*xylim)
    '''


def save_json(filename, var):
    with open(filename,'w') as f:
        f.write(str(json.dumps(var, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)))


# QUANTUM MULTICLASS SVM
# Amer Delilbasic
# Ref: Crammer.2001

# C = number of classes, N = number of samples, B = number of bits per unknown, taus = matrix of unknowns of size (N,C)

# The function kernel() is the same.

def print_time(t):
    # t: time in seconds. Hopefully: end-start, extracted from time.time()
    print(f"Run time: {format((int)(t)/60, '.0f')} minutes and {format((int)(t)%60, '.0f')} seconds\n")


def eval_classifier_multiclass(x, taus, data, C, gamma=-1):
    # TODO: find a way to run this for multiple times for multiple taus and computing the kernel only once!
    classifiers = np.zeros((C, len(x)))
    for c in range(C):
        classifiers[c] = np.sum(taus[:, c, None] * kernel(data, x, gamma), axis=0)
    return classifiers


# Implemented, but never turned out to be useful.
class MulticlassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, taus, x_train, C, gamma):
        self.taus=taus
        self.x_train=x_train
        self.C=C
        self.gamma=gamma

    def fit(self, X, y=None):
        pass
    
    def predict(self, X, y=None):
        decision_f = eval_classifier_multiclass(X, self.taus, self.x_train, self.C, self.gamma)
        preds = np.argmax(decision_f, axis=0).tolist()
        return preds
    
    def predict_proba(self, X, y=None):
        pass


def decode_multiclass(binary, C=2, B=2):                            # binary is the binary vector encoding the solution
    N = len(binary) // (C * B)                                      # Compute number of samples
    binary_mat = np.fromiter(binary, float).reshape(N * C, B)       # Each row is the binary encoding of tau_{n,c}
    decimal_vec = reduce(lambda a, b: 2 * a + b, np.transpose(binary_mat))  # Conversion to decimal values {0,...,2**B -1}
    decimal_mat = decimal_vec.reshape(N, C)                         # Now we can write decimal_mat[n,c]
    taus = decimal_mat * (2 / (2 ** B - 1)) - 1                     # decimal_mat==0 -> tau = -1, decimal_mat = 2**B -1 -> tau = 1
    return taus                                                     # In this way [-1,1] is discretized in an odd number of points.


def get_training_data_and_labels(max_model_size, data, labels, seed):
    if 0 < max_model_size < len(data):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=max_model_size, random_state=seed)
        _, training_indices = list(sss.split(data, labels))[0]
        training_data, training_labels = data[training_indices], labels[training_indices]
    else:
        training_data, training_labels = data, labels

    return training_data, training_labels


def gen_svm_qubos_multiclass(X, Y_indices, C, B, beta, mu, gamma, max_min_ratio):  # , path):
    N = len(X)
    Q = np.zeros((C * B * N, C * B * N))
    # print(f'Creating the QUBO Q matrix of size {Q.shape}')
    for n1 in range(N):
        for n2 in range(N):
            for c1 in range(C):
                for c2 in range(C):
                    for b1 in range(B):
                        for b2 in range(B):
                            if n1 == n2 and c1 == c2 and b1 == b2:
                                kernel_sum = 0
                                for i in range(N):
                                    kernel_sum += kernel(X[n1], X[i], gamma)
                                Q[C*B*n1+B*c1+b1, C*B*n2+B*c2+b2] += 2**(b1+1)/(2**B-1)*(-kernel_sum-2*C*mu+mu)
                                if c1 == Y_indices[n1]:
                                    Q[C*B*n1+B*c1+b1, C*B*n2+B*c2+b2] += -2**(b1+1)/(2**B-1)*(beta+mu)
                            if c1 == c2:
                                Q[C*B*n1+B*c1+b1, C*B*n2+B*c2+b2] += (2**(b1+b2+1))/(2**B-1)**2 * kernel(X[n1], X[n2], gamma)
                            if n1 == n2:
                                Q[C*B*n1+B*c1+b1, C*B*n2+B*c2+b2] += (mu*2**(b1+b2+2))/(2**B-1)**2
    Q = np.triu(Q) + np.tril(Q, -1).T  # turn the symmetric matrix into upper triangular
    
    # print('Extracting nodes and couplers')
    qubo_nodes = np.asarray([[n, n, Q[n, n]] for n in
                             range(len(Q))])  # if not np.isclose(Q[n,n],0)]) NOTE: removed due to variable order!
    max_Q = np.amax(Q)
    qubo_couplers = np.asarray(
        [[n, m, Q[n, m]] for n in range(len(Q)) for m in range(n + 1, len(Q)) if not np.isclose(Q[n, m], 0, atol=max_Q/max_min_ratio)])
    qubo_couplers = qubo_couplers[np.argsort(-np.abs(qubo_couplers[:, 2]))]

    # print(f'Saving {len(qubo_nodes)} nodes and {len(qubo_couplers)} couplers for {path}\n')
    # os.makedirs(path, exist_ok=True)
    # np.save(path + 'Q.npy', Q)
    # np.savetxt(path + 'Q.dat', Q, fmt='%7.2f')
    # np.savetxt(path + 'qubo_nodes.dat', qubo_nodes, fmt='%g', delimiter='\t')
    # np.savetxt(path + 'qubo_couplers.dat', qubo_couplers, fmt='%g', delimiter='\t')

    return Q, qubo_nodes, qubo_couplers


def compute_cost_function_taus():
    pass # TODO


def predict_multiclass(x_test, x_train, x_val, y_val, C, gamma, selected_classes, emb, rcs, at, path):
    # Solutions combination
    taus = np.load(path+f'embedding{emb}_rcs{rcs}_at{at}_taus.npy')
    taus_comb = solutions_combination(x_train, taus, x_val, y_val, C, gamma, selected_classes, comb='softmax')    
    # Validation set prediction
    decision_f = eval_classifier_multiclass(x_val, taus_comb, x_train, C, gamma)
    decision_ind = np.argmax(decision_f, axis=0).tolist()
    y_test_predict = []
    for i in range(x_test.shape[0]):
        y_test_predict.append(selected_classes[decision_ind[i]])
    y_test_predict = np.asarray(y_test_predict)
    return y_test_predict
    

def solutions_combination(x_train, taus, x_val, y_val, C, gamma, selected_classes, comb='average', i=0):
    max_taus = taus.shape[0]
    train_acc = np.zeros(max_taus)
    weights = np.zeros(max_taus)

    taus_comb = None
    if comb == 'average':
        taus_comb = np.average(taus, axis=0)
    if comb == 'single':
        taus_comb = taus[i]
    if comb == 'softmax':
        for i in range(max_taus):
            decision_f = eval_classifier_multiclass(x_val, taus[i], x_train, C, gamma)
            decision_ind = np.argmax(decision_f, axis=0).tolist()
            y_val_predict = []
            for j in decision_ind:
                y_val_predict.append(selected_classes[j])
            y_val_predict = np.asarray(y_val_predict)
            train_acc[i] = accuracy_score(y_val_predict, y_val)
            # weights[i] = np.log(train_acc[i])

        max_acc = np.max(train_acc)
        min_acc = np.min(train_acc)
        thr = 0.8

        # taking only solutions with acc higher than a certain threshold. 0<=thr<=1. 0: all solutions, 1: only the best
        weights = np.where(train_acc < (1-thr)*min_acc + thr*max_acc, -1e100, 10*train_acc)
        weights_softmax = np.exp(weights)/sum(np.exp(weights))
        # print(weights)
        # print(weights_softmax)

        taus_comb = np.average(taus, axis=0, weights=weights_softmax)

    # print("Using the following combination of taus:")
    # print(taus_comb)

    return taus_comb


def run_calibration(x_train, y_train_indices, C, B, x_val, y_val, max_taus, selected_classes, outputpath):
    betas = [0.1, 0.5, 1]
    mus = [0.1, 0.5, 1]
    gammas = [1, 5]
    chain_strengths = [2]
    annealing_times = [100]
    
    # Chosen initial values
    rcs=[2]
    at=[50]
    
    #Additional variables
    t = 0
    best_accuracy = 0.0
    emb=0
    
    for beta in betas:
        for mu in mus:
            for gamma in gammas:
                start = time.time()
                print(f'Computing beta={beta}, mu={mu}, gamma={gamma}')
                path=outputpath+f'_C={C}_B={B}/' #+f'_C={C}_B={B}_beta={beta}_mu={mu}_gamma={gamma}/'
                print('The signature of the functions gen_svm_qubos_multiclass and dwave_run_embedding_multiclass has '
                      'been changed. Exiting')
                exit(-1)
                # gen_svm_qubos_multiclass(x_train, y_train_indices, C, B, beta, mu, gamma, 1e10, path)
                # dwave_run_embedding_multiclass(path, rcs, at, emb, max_taus, calibration=True, clique=True)
                path = path+'/Advantage/'
                y_val_predict = predict_multiclass(x_val, x_train, x_val, y_val, C, gamma, selected_classes, emb, rcs[0], at[0], path)
                accuracy = accuracy_score(y_val_predict, y_val)
                print(f'Accuracy on validation set: {accuracy}')
                if accuracy > best_accuracy:
                    beta_opt = beta
                    mu_opt = mu
                    gamma_opt = gamma
                    best_accuracy = accuracy
                end = time.time()
                t += end-start
                print_time(end-start)
                
    for annealing_time in annealing_times:
        for chain_strength in chain_strengths:
            at_opt = annealing_time
            cs_opt = chain_strength
    print(f'End of calibration. Selected values: beta={beta_opt}, mu={mu_opt}, gamma={gamma_opt}')
    print_time(t)
    return beta_opt, mu_opt, gamma_opt, at_opt, cs_opt


# def dwave_run_embedding_multiclass(path_in, chain_strengths, annealing_times, em_id, MAXRESULTS = 20, num_reads = 1000, calibration = False, clique = False, force_run = False):  # em_id is a label for the embedding. In this way, it is possible to re-run a previously computed and stored embedding with e.g. different chain strength and/or annealing time
def dwave_run_embedding_multiclass(qubo_nodes, qubo_couplers, C, B, embeddings_dir, annealing_times, chain_strengths,
                                   num_reads, profile, max_results):
    # solver='Advantage_system4.1'
    # start = time.time()
    # if not calibration:
    #     match = re.search('run_([^/]*)_C=(.*)_B=(.*)_beta=(.*)_mu=(.*)_gamma=([^/]*)', path_in)
    # else:
    #     match = re.search('run_([^/]*)_C=(.*)_B=(.*)/', path_in)

    # data_key = match.group(1)
    # C = int(match.group(2))
    # B = int(match.group(3))
    # if not calibration:
    # beta = float(match.group(4))
    # mu = float(match.group(5))
    # gamma = float(match.group(6))
    # ratio = float(match.group(7))
    # all these variables are not really needed

    # path = path_in + ('/' if path_in[-1] != '/' else '')
    # qubo_couplers = np.loadtxt(path + 'qubo_couplers.dat')
    # qubo_nodes = np.loadtxt(path + 'qubo_nodes.dat')
    qubo_nodes = np.array([
        [i, i, (qubo_nodes[qubo_nodes[:, 0] == i, 2][0] if i in qubo_nodes[:, 0] else 0.)]
        for i in np.arange(np.concatenate((qubo_nodes, qubo_couplers))[:, [0, 1]].max() + 1)
    ])  # to make sure every (i,i) occurs in the qubo in increasing order such that the variable order in BinaryQuadraticModel is consistent (see locate wrongenergies-* github issue)
    nodes_num = len(qubo_nodes)
    maxcouplers = len(qubo_couplers)  # POSSIBLE INPUT if len(sys.argv) <= 2 else int(sys.argv[2])

    #if not 'train' in data_key:
    #    raise Exception(f'careful: datakey={data_key} => you\'re trying to train on a validation / test set!')

    couplerslist = [min(7500, maxcouplers)]  # The 7500 here is more or less arbitrary and may need to be adjusted. Just to be sure that the number of couplers is not larger than the number of physical couplers (EmbeddingComposite does not seem to check for this)
    for trycouplers in [5000, 2500, 2000, 1800, 1600, 1400, 1200, 1000, 500]:
        if maxcouplers > trycouplers:
            couplerslist += [trycouplers]

    # if not clique:
    #     sampler = LazyFixedEmbeddingComposite(DWaveSampler())  # use the same embedding for all chain strengths and annealing times
    # else:
    #     sampler = LazyFixedEmbeddingComposite(DWaveSampler(embedding=find_clique_embedding))
    sampler = LazyFixedEmbeddingComposite(DWaveSampler())  # use the same embedding for all chain strengths and annealing times
    # print('Using solver {}'.format(sampler.children[0].solver.id))

    Q = {}
    maxstrength = -1
    embedding_filepath = None
    if os.path.isdir(embeddings_dir):
        Q = {(q[0], q[1]): q[2] for q in np.vstack((qubo_nodes, qubo_couplers[:maxcouplers]))}
        maxstrength = np.max(np.abs(list(Q.values())))

        embedding_filepath = os.path.join(embeddings_dir, 'emb_{}_variables'.format(nodes_num))
        if os.path.isfile(embedding_filepath):
            with open(embedding_filepath) as embedding_file:
                embedding_data = eval(embedding_file.read())
                sampler._fix_embedding(embedding_data)
        else:
            tmp_Q = {(i, j): 0.5 for i in range(0, nodes_num) for j in range(i, nodes_num)}
            try:
                sampler.sample_qubo(tmp_Q, num_reads=1, chain_strength=1)
            except ValueError as _:
                print(f'Not possible to find a complete embedding for {nodes_num} binary variables')
                exit(-1)
            with open(embedding_filepath, 'w') as embedding_file:
                # print('Embedding found. Saving...')
                embedding_file.write(repr(sampler.embedding))

    response = None
    all_taus = []
    for m in range(0, len(annealing_times)):
        for n in range(0, len(chain_strengths)):
            if m == 0 and n == 0 and embedding_filepath is None:
                for couplers in couplerslist:  # try to reduce as little couplers as necessary to find an embedding
                    Q = {(q[0], q[1]): q[2] for q in np.vstack((qubo_nodes, qubo_couplers[:couplers]))}
                    maxstrength = np.max(np.abs(list(Q.values())))

                    # pathsub = path + 'Advantage/'
                    # os.makedirs(pathsub, exist_ok=True)
                    # taus_file_name = pathsub + f'embedding{em_id}_rcs{chain_strengths[n]}_at{annealing_times[m]}_taus.npy'
                    # if os.path.isfile(taus_file_name) and not force_run:
                    #     print(f'Found previously computed solution for {pathsub} with {len(qubo_nodes)} nodes and {couplers} couplers for embedding {em_id}.')
                    #     return pathsub
                    # embedding_file_name = pathsub + f'embedding_id{em_id}'
                    # if os.path.isfile(embedding_file_name) and not force_run:
                    #     embedding_file = open(embedding_file_name, 'r')
                    #     try:
                    #         embedding_data = eval(embedding_file.read())
                    #     except:
                    #         print('Reading error')
                    #         return
                    #     finally:
                    #         embedding_file.close()
                    #     sampler._fix_embedding(embedding_data)
                    # print(
                    #     f'Running {pathsub} with {len(qubo_nodes)} nodes and {couplers} couplers for embedding {em_id}')
                    print(f'Running with {nodes_num} nodes and {couplers} couplers')

                    #ordering = np.array(list(BinaryQuadraticModel.from_qubo(Q)))
                    #if not (ordering == np.arange(len(ordering), dtype=ordering.dtype)).all():
                    #    print(f'WARNING: variables are not correctly ordered! path={path} ordering={ordering}')

                    try:
                        if annealing_times[0] is not None and chain_strengths[0] is not None:
                            print(f'Running annealing time {annealing_times[0]} and chain strength {chain_strengths[0]}\n')
                            response = sampler.sample_qubo(Q, num_reads=num_reads, annealing_time=annealing_times[0],
                                                           chain_strength=chain_strengths[0] * maxstrength)
                        else:
                            response = sampler.sample_qubo(Q, num_reads=num_reads)

                        # if not os.path.isfile(embedding_file_name):
                        #     embedding_file = open(embedding_file_name, 'w')
                        #     print('Embedding found. Saving...')
                        #     embedding_file.write(repr(sampler.embedding))
                        #     embedding_file.close()
                    except ValueError as _:
                        # print(f' -- no embedding found, removing {pathsub} and trying less couplers')
                        # shutil.rmtree(pathsub)
                        # sampler = LazyFixedEmbeddingComposite(DWaveSampler())
                        print(f' -- no embedding found, trying less couplers')
                        sampler.child.client.close()
                        sampler = LazyFixedEmbeddingComposite(DWaveSampler())
                        continue
                    break
            else:
                if annealing_times[m] is not None and chain_strengths[n] is not None:
                    # print(f'Running annealing time {annealing_times[m]} and chain strength {chain_strengths[n]}\n')
                    response = sampler.sample_qubo(Q, num_reads=num_reads, annealing_time=annealing_times[m],
                                                   chain_strength=chain_strengths[n]*maxstrength)
                else:
                    response = sampler.sample_qubo(Q, num_reads=num_reads)

            # pathsub_ext = pathsub + f'embedding{em_id}_rcs{chain_strengths[n]}_at{annealing_times[m]}_' # couplers{maxcouplers}_ not needed
            # save_json(pathsub_ext + 'info.json', response.info)  # contains response.info
            # NOTE left out: pickle.dump(response, open(pathsub+'response.pkl','wb')) # contains full response incl. response.record etc; can be loaded with pickle.load(open('response.pkl','rb'))

            samples = np.array([''.join(map(str, sample)) for sample in response.record['sample']])  # NOTE: it would be safer to use the labeling from record.data() for the qubit variable order
            unique_samples, unique_idx, unique_counts = np.unique(samples, return_index=True, return_counts=True)  # unfortunately, num_occurrences seems not to be added up after unembedding
            unique_records = response.record[unique_idx]
            result = rfn.merge_arrays((unique_samples, unique_records['energy'], unique_counts,
                                       unique_records['chain_break_fraction']))  # see comment on chain_strength above
            result = result[np.argsort(result['f1'])]
            # np.savetxt(pathsub_ext + 'result.dat', result[:max_results], fmt='%s', delimiter='\t', header='\t'.join(response.record.dtype.names), comments='')  # load with np.genfromtxt(..., dtype=['<U2000',float,int,float], names=True, encoding=None)

            taus = np.array([decode_multiclass(sample, C, B) for sample in result['f0'][:max_results]])
            all_taus.append(taus)
            # np.save(pathsub_ext + 'taus.npy', taus)
            gc.collect()

    # t_annealing = time.time()-start
    # print(f'Running completed. Time: {t_annealing} s.')
    # np.save(path+'t_annealing', t_annealing)

    sampler.child.client.close()

    return all_taus


def dwave_run_embedding_multiclass_HSS(path_in, time_limit):  # em_id is a label for the embedding. In this way, it is possible to re-run a previously computed and stored embedding with e.g. different chain strength and/or annealing time
    # solver='Advantage_system4.1'
    MAXRESULTS = 20
    match = re.search('run_([^/]*)_C=(.*)_B=(.*)_beta=(.*)_mu=(.*)_gamma=([^/]*)', path_in)

    data_key = match.group(1)
    C = int(match.group(2))
    B = int(match.group(3))
    beta = float(match.group(4))
    mu = float(match.group(5))
    gamma = float(match.group(6))

    path = path_in + ('/' if path_in[-1] != '/' else '')
    qubo_couplers = np.loadtxt(path + 'qubo_couplers.dat')
    qubo_nodes = np.loadtxt(path + 'qubo_nodes.dat')
    qubo_nodes = np.array([[i, i, (qubo_nodes[qubo_nodes[:, 0] == i, 2][0] if i in qubo_nodes[:, 0] else 0.)] for i in
                           np.arange(np.concatenate((qubo_nodes, qubo_couplers))[:, [0,
                                                                                     1]].max() + 1)])  # to make sure every (i,i) occurs in the qubo in increasing order such that the variable order in BinaryQuadraticModel is consistent (see locate wrongenergies-* github issue)
    #maxcouplers = len(qubo_couplers)  # POSSIBLE INPUT if len(sys.argv) <= 2 else int(sys.argv[2])

    #couplers = maxcouplers  # All couplers should fit in the solver

    pathsub = path + 'HSS/'
    os.makedirs(pathsub, exist_ok=True)
    if 'train' not in data_key:
        raise Exception(f'careful: datakey={data_key} => you\'re trying to train on a validation / test set!')

    Q = {(q[0], q[1]): q[2] for q in np.vstack((qubo_nodes, qubo_couplers))}

    bqm = BinaryQuadraticModel.from_qubo(Q)
    sampler = LeapHybridSampler()
    response = sampler.sample(bqm, time_limit)

    pathsub_ext = pathsub + f'HSS_time_limit={time_limit}_'
    save_json(pathsub_ext + 'info.json', response.info)  # contains response.info
    # NOTE left out: pickle.dump(response, open(pathsub+'response.pkl','wb')) # contains full response incl. response.record etc; can be loaded with pickle.load(open('response.pkl','rb'))

    samples = np.array([''.join(map(str, sample)) for sample in response.record[
        'sample']])  # NOTE: it would be safer to use the labeling from record.data() for the qubit variable order
    unique_samples, unique_idx, unique_counts = np.unique(samples, return_index=True,
                                                          return_counts=True)  # unfortunately, num_occurrences seems not to be added up after unembedding
    unique_records = response.record[unique_idx]
    result = rfn.merge_arrays((unique_samples, unique_records['energy'], unique_counts))  # see comment on chain_strength above
    result = result[np.argsort(result['f1'])]
    np.savetxt(pathsub_ext + 'result.dat', result[:MAXRESULTS], fmt='%s', delimiter='\t',
               header='\t'.join(response.record.dtype.names),
               comments='')  # load with np.genfromtxt(..., dtype=['<U2000',float,int,float], names=True, encoding=None)

    taus = np.array([decode_multiclass(sample, C, B) for sample in result['f0'][:MAXRESULTS]])
    np.save(pathsub_ext + 'taus.npy', taus)
    gc.collect()

    print('Running completed.')
    return pathsub


def dwave_run_embedding_multiclass_kerberos(path_in, num_reads=1, max_iter=10, qpu_reads=100    ):  # em_id is a label for the embedding. In this way, it is possible to re-run a previously computed and stored embedding with e.g. different chain strength and/or annealing time
    #MAXRESULTS = 20
    #solver='Advantage_system4.1'
    match = re.search('run_([^/]*)_C=(.*)_B=(.*)_beta=(.*)_mu=(.*)_gamma=([^/]*)', path_in)

    data_key = match.group(1)
    C = int(match.group(2))
    B = int(match.group(3))
    beta = float(match.group(4))
    mu = float(match.group(5))
    gamma = float(match.group(6))

    path = path_in + ('/' if path_in[-1] != '/' else '')
    qubo_couplers = np.loadtxt(path + 'qubo_couplers.dat')
    qubo_nodes = np.loadtxt(path + 'qubo_nodes.dat')
    qubo_nodes = np.array([[i, i, (qubo_nodes[qubo_nodes[:, 0] == i, 2][0] if i in qubo_nodes[:, 0] else 0.)] for i in
                           np.arange(np.concatenate((qubo_nodes, qubo_couplers))[:, [0,
                                                                                     1]].max() + 1)])  # to make sure every (i,i) occurs in the qubo in increasing order such that the variable order in BinaryQuadraticModel is consistent (see locate wrongenergies-* github issue)
    #maxcouplers = len(qubo_couplers)  # POSSIBLE INPUT if len(sys.argv) <= 2 else int(sys.argv[2])

    #couplers = maxcouplers  # All couplers should fit in the solver

    pathsub = path + 'kerberos/'
    os.makedirs(pathsub, exist_ok=True)
    if 'train' not in data_key:
        raise Exception(f'careful: datakey={data_key} => you\'re trying to train on a validation / test set!')

    Q = {(q[0], q[1]): q[2] for q in np.vstack((qubo_nodes, qubo_couplers))}

    bqm = BinaryQuadraticModel.from_qubo(Q)
    sampler = KerberosSampler()
    response = sampler.sample(bqm,num_reads=num_reads,max_iter=max_iter,qpu_reads=qpu_reads)

    pathsub_ext = pathsub  # not really necessary to create a folder for each parameter combination
    save_json(pathsub_ext + 'info.json', response.info)  # contains response.info
    # NOTE left out: pickle.dump(response, open(pathsub+'response.pkl','wb')) # contains full response incl. response.record etc; can be loaded with pickle.load(open('response.pkl','rb'))

    samples = np.array([''.join(map(str, sample)) for sample in response.record[
        'sample']])  # NOTE: it would be safer to use the labeling from record.data() for the qubit variable order
    unique_samples, unique_idx, unique_counts = np.unique(samples, return_index=True,
                                                          return_counts=True)  # unfortunately, num_occurrences seems not to be added up after unembedding
    unique_records = response.record[unique_idx]
    result = rfn.merge_arrays((unique_samples, unique_records['energy'], unique_counts))  # see comment on chain_strength above
    result = result[np.argsort(result['f1'])]
    np.savetxt(pathsub_ext + 'result.dat', result[:], fmt='%s', delimiter='\t',
               header='\t'.join(response.record.dtype.names),
               comments='')  # load with np.genfromtxt(..., dtype=['<U2000',float,int,float], names=True, encoding=None)

    taus = np.array([decode_multiclass(sample, C, B) for sample in result['f0'][:]])
    np.save(pathsub_ext + 'taus.npy', taus)
    gc.collect()

    print('Running completed.')
    return pathsub
