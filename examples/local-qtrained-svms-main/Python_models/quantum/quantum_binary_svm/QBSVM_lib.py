# Created by Dennis Willsch (d.willsch@fz-juelich.de) 
# Modified by Gabriele Cavallaro (g.cavallaro@fz-juelich.de)
#         and Madita Willsch (m.willsch@fz-juelich.de)

import gc
import json
import matplotlib.colors as cols
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rfn
import os
import re
import sys

# import numpy.lib.recfunctions as rfn
from dimod import BinaryQuadraticModel
from dwave.system.composites import LazyFixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, accuracy_score, auc
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

np.set_printoptions(precision=4, suppress=True)


def split_data_and_labels(max_model_size, data, labels, seed):
    data_len = len(data)
    if 0 < max_model_size < data_len:
        n_folds, remainder = int(data_len / max_model_size), data_len % max_model_size

        if remainder not in [0, 1]:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=remainder, random_state=seed)
            indices_for_folds, remainder_indices = list(sss.split(data, labels))[0]
            data_for_folds, labels_for_folds = data[indices_for_folds], labels[indices_for_folds]
            remainder_data, remainder_labels = data[remainder_indices], labels[remainder_indices]
        else:
            data_for_folds, labels_for_folds = data, labels
            remainder_data, remainder_labels = [], []

        split_data, split_labels = [], []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed+1)
        for _, fold_indices in skf.split(data_for_folds, labels_for_folds):
            split_data.append(data_for_folds[fold_indices])
            split_labels.append(labels_for_folds[fold_indices])

        if len(remainder_data) != 0 and len(remainder_labels) != 0:
            split_data.append(remainder_data)
            split_labels.append(remainder_labels)
    else:
        split_data = [data]
        split_labels = [labels]

    return split_data, split_labels


def gen_svm_qubos(data, label, B, K, xi, gamma, E):  # , path):
    N = len(data)

    # if not os.path.isfile(path+'Q.npy'):
    Q = np.zeros((K*N, K*N))

    # print(f'Creating the QUBO of size {Q.shape}')
    for n in range(N):  # not optimized: size will not be so large and this way its more easily verifiable
        for m in range(N):
            for k in range(K):
                for j in range(K):
                    Q[K*n+k, K*m+j] = .5 * B**(k+j-2*E) * label[n] * label[m] * (kernel(data[n], data[m], gamma) + xi)
                    if n == m and k == j:
                        Q[K*n+k, K*m+j] += - B**(k-E)

    Q = np.triu(Q) + np.tril(Q, -1).T  # turn the symmetric matrix into upper triangular
    # else:
    #   Q = np.load(path+'Q.npy')

    # print(f'Extracting nodes and couplers')
    qubo_nodes = np.asarray([[n, n, Q[n, n]] for n in range(len(Q))])  # if not np.isclose(Q[n,n],0)]) NOTE: removed due to variable order!
    qubo_couplers = np.asarray([[n, m, Q[n, m]] for n in range(len(Q)) for m in range(n+1, len(Q)) if not np.isclose(Q[n, m], 0)])
    qubo_couplers = qubo_couplers[np.argsort(-np.abs(qubo_couplers[:, 2]))]

    # print(f'Saving {len(qubo_nodes)} nodes and {len(qubo_couplers)} couplers for {path}')
    # os.makedirs(path, exist_ok=True)
    # np.save(path+'Q.npy', Q)
    # np.savetxt(path+'qubo_nodes.dat', qubo_nodes, fmt='%g', delimiter='\t')
    # np.savetxt(path+'qubo_couplers.dat', qubo_couplers, fmt='%g', delimiter='\t')

    return Q, qubo_nodes, qubo_couplers


# def dwave_run_embedding(data,label,path_in,annealing_times,chain_strengths,em_id,solver='Advantage_system1.1'): #em_id is a label for the embedding. In this way, it is possible to re-run a previously computed and stored embedding with e.g. different chain strength and/or annealing time
def dwave_run_embedding(qubo_nodes, qubo_couplers, B, K, E, embeddings_dir, annealing_times, chain_strengths, num_reads,
                        profile, max_results):
    # MAXRESULTS = 20  # NOTE: to save space only 20 best results
    # match = re.search('run_([^/]*)_B=(.*)_K=(.*)_xi=(.*)_E=(.*)_gamma=([^/]*)', path_in)

    # data_key = match.group(1)
    # B = int(match.group(2))
    # K = int(match.group(3))
    # xi = float(match.group(4))
    # gamma = float(match.group(6))
    # E = int(match.group(5))

    # path = path_in+ ('/' if path_in[-1] != '/' else '')
    # qubo_couplers = np.loadtxt(path+'qubo_couplers.dat')
    # qubo_nodes = np.loadtxt(path+'qubo_nodes.dat')
    qubo_nodes = np.array([
        [i, i, (qubo_nodes[qubo_nodes[:, 0] == i, 2][0] if i in qubo_nodes[:, 0] else 0.)]
        for i in np.arange(np.concatenate((qubo_nodes, qubo_couplers))[:, [0, 1]].max()+1)
    ])  # to make sure every (i,i) occurs in the qubo in increasing order such that the variable order in BinaryQuadraticModel is consistent (see locate wrongenergies-* github issue)
    nodes_num = len(qubo_nodes)
    maxcouplers = len(qubo_couplers)  # POSSIBLE INPUT if len(sys.argv) <= 2 else int(sys.argv[2])

    # if not 'train' in data_key:
    #     raise Exception(f'careful: datakey={data_key} => you're trying to train on a validation / test set!')

    couplerslist = [min(7500, maxcouplers)]  # The 7500 here is more or less arbitrary and may need to be adjusted. Just to be sure that the number of couplers is not larger than the number of physical couplers (EmbeddingComposite does not seem to check for this)
    for trycouplers in [5000, 2500, 2000, 1800, 1600, 1400, 1200, 1000, 500]:
        if maxcouplers > trycouplers:
            couplerslist += [trycouplers]

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
                # print(f'Embedding found for {nodes_num} binary variables')
                embedding_file.write(repr(sampler.embedding))

    response = None
    all_alphas = []
    for m in range(0, len(annealing_times)):
        for n in range(0, len(chain_strengths)):
            if m == 0 and n == 0 and embedding_filepath is None:
                for couplers in couplerslist:  # try to reduce as little couplers as necessary to find an embedding
                    Q = {(q[0], q[1]): q[2] for q in np.vstack((qubo_nodes, qubo_couplers[:couplers]))}
                    maxstrength = np.max(np.abs(list(Q.values())))

                    # pathsub = path + f'result_couplers={couplers}/'
                    # os.makedirs(pathsub, exist_ok=True)
                    # embedding_file_name=pathsub+f'embedding_id{em_id}'
                    # if os.path.isfile(embedding_file_name):
                    #     embedding_file = open(embedding_file_name,'r')
                    #     embedding_data = eval(embedding_file.read())
                    #     embedding_file.close()
                    #     sampler._fix_embedding(embedding_data)
                    # print(f'running {pathsub} with {nodes_num} nodes and {couplers} couplers for embedding {em_id}')
                    print(f'Running with {nodes_num} nodes and {couplers} couplers')

                    ordering = np.array(list(BinaryQuadraticModel.from_qubo(Q)))
                    if not (ordering == np.arange(len(ordering), dtype=ordering.dtype)).all():
                        # print(f'WARNING: variables are not correctly ordered! path={path} ordering={ordering}')
                        print(f'WARNING: variables are not correctly ordered! ordering={ordering}')

                    try:
                        if annealing_times[0] is not None and chain_strengths[0] is not None:
                            print(f'Running annealing time {annealing_times[0]} and chain strength {chain_strengths[0]}\n')
                            response = sampler.sample_qubo(Q, num_reads=num_reads, annealing_time=annealing_times[0],
                                                           chain_strength=chain_strengths[0]*maxstrength)
                        else:
                            response = sampler.sample_qubo(Q, num_reads=num_reads)

                        # if not os.path.isfile(embedding_file_name):
                        #     embedding_file=open(embedding_file_name,'w')
                        #     print('Embedding found. Saving...')
                        #     embedding_file.write(repr(sampler.embedding))
                        #     embedding_file.close()
                    except ValueError as _:
                        # print(f' -- no embedding found, removing {pathsub} and trying less couplers')
                        # shutil.rmtree(pathsub)
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

            # pathsub_ext=pathsub+f'embedding{em_id}_rcs{chain_strengths[n]}_ta{annealing_times[m]}_'
            # save_json(pathsub_ext+'info.json', response.info) # contains response.info
            # NOTE left out: pickle.dump(response, open(pathsub+'response.pkl','wb'))  # contains full response incl. response.record etc; can be loaded with pickle.load(open('response.pkl','rb'))

            samples = np.array([''.join(map(str, sample)) for sample in response.record['sample']])  # NOTE: it would be safer to use the labeling from record.data() for the qubit variable order
            unique_samples, unique_idx, unique_counts = np.unique(samples, return_index=True, return_counts=True)  # unfortunately, num_occurrences seems not to be added up after unembedding
            unique_records = response.record[unique_idx]
            result = rfn.merge_arrays((unique_samples, unique_records['energy'], unique_counts,
                                       unique_records['chain_break_fraction']))  # see comment on chain_strength above
            result = result[np.argsort(result['f1'])]
            # np.savetxt(pathsub_ext+'result.dat', result[:MAXRESULTS], fmt='%s', delimiter='\t', header='\t'.join(response.record.dtype.names), comments='')  # load with np.genfromtxt(..., dtype=['<U2000',float,int,float], names=True, encoding=None)

            alphas = np.array([decode(sample, B, K, E) for sample in result['f0'][:max_results]])
            all_alphas.append(alphas)
            # np.save(pathsub_ext+f'alphas.npy', alphas)
            gc.collect()

    sampler.child.client.close()

    return all_alphas


def eval_run_trainaccuracy(path_in):
    regex = 'run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_E=(.*)_gamma=([^/]*)/result_couplers.*/?$'
    match = re.search(regex, path_in)

    path = path_in + ('/' if path_in[-1] != '/' else '')
    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(6))
    E = int(match.group(5))
    data,label = loaddataset(data_key)

    alphas_file = path+f'alphas{data_key}_gamma={gamma}.npy'
    if not os.path.isfile(alphas_file):
        print('result '+alphas_file+' doesnt exist, exiting')
        sys.exit(-1)

    alphas = np.atleast_2d(np.load(alphas_file))
    nalphas = len(alphas)
    assert len(data) == alphas.shape[1], "alphas do not seem to be for the right data set?)"

    result = np.genfromtxt(path+'result.dat', dtype=['<U2000',float,int,float], names=True, encoding=None, max_rows=nalphas)

    Cs = [100, 10, (B**np.arange(K-E)).sum(), 1.5]
    evaluation = np.zeros(nalphas, dtype=[('sum_antn',float)]+[(f'acc(C={C})',float) for C in Cs])

    for n,alphas_n in enumerate(alphas):
        evaluation[n]['sum_antn'] = (label * alphas_n).sum()
        for j,field in enumerate(evaluation.dtype.names[1:]):
            b = eval_offset_avg(alphas_n, data, label, gamma, Cs[j]) # NOTE: this is NAN if no support vectors were found, see TODO file
            label_predicted = np.sign(eval_classifier(data, alphas_n, data, label, gamma, b)) # NOTE: this is only train accuracy! (see eval_result_roc*)
            evaluation[n][field] = (label == label_predicted).sum() / len(label)

    result_evaluated = rfn.merge_arrays((result,evaluation), flatten=True)
    fmt = '%s\t%.3f\t%d\t%.3f' + '\t%.3f'*len(evaluation.dtype.names)
    #NOTE: left out
    # np.savetxt(path+'result_evaluated.dat', result_evaluated, fmt=fmt, delimiter='\t', header='\t'.join(result_evaluated.dtype.names), comments='') # load with np.genfromtxt(..., dtype=['<U2000',float,int,float,float,float,float,float], names=True, encoding=None)

    print(result_evaluated.dtype.names)
    print(result_evaluated)


def eval_run_rocpr_curves(path_data_key,path_in,plotoption):
    regex = 'run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_E=(.*)_gamma=([^/]*)/result_couplers.*/?$'
    match = re.search(regex, path_in)

    path = path_in + ('/' if path_in[-1] != '/' else '')
    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(6))
    E = int(match.group(5))
    data,label = loaddataset(path_data_key+data_key)

    dwavesolutionidx=0
    C=(B**np.arange(K-E)).sum()

    if 'calibtrain' in data_key:
        testname = 'Validation'
        datatest,labeltest = loaddataset(path_data_key+data_key.replace('calibtrain','calibval'))
    else:
        print('be careful: this does not use the aggregated bagging classifier but only the simple one as in calibration')
        testname = 'Test'
        datatest,labeltest = loaddataset(re.sub('train(?:set)?[0-9]*(?:bag)[0-9]*','test',data_key))

    alphas_file = path+f'alphas{data_key}_gamma={gamma}.npy'
    if not os.path.isfile(alphas_file):
        print('result '+alphas_file+' doesnt exist, exiting')
        sys.exit(-1)

    alphas = np.atleast_2d(np.load(alphas_file))
    nalphas = len(alphas)
    assert len(data) == alphas.shape[1], "alphas do not seem to be for the right data set?)"

    print('idx   \tsum_antn\ttrainacc\ttrainauroc\ttrainauprc\ttestacc  \ttestauroc\ttestauprc')

    trainacc_all=np.zeros([nalphas])
    trainauroc_all=np.zeros([nalphas])
    trainauprc_all=np.zeros([nalphas])

    testacc_all=np.zeros([nalphas])
    testauroc_all=np.zeros([nalphas])
    testauprc_all=np.zeros([nalphas])

    for i in range(nalphas):
        alphas_n = alphas[i]
        b = eval_offset_avg(alphas_n, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
        score = eval_classifier(data, alphas_n, data, label, gamma, b)
        scoretest = eval_classifier(datatest, alphas_n, data, label, gamma, b)
        trainacc,trainauroc,trainauprc = eval_acc_auroc_auprc(label,score)
        testacc,testauroc,testauprc = eval_acc_auroc_auprc(labeltest,scoretest)

        trainacc_all[i]=trainacc
        trainauroc_all[i]=trainauroc
        trainauprc_all[i]=trainauprc
        testacc_all[i]=testacc
        testauroc_all[i]=testauroc
        testauprc_all[i]=testauprc

        print(f'{i}\t{(label*alphas_n).sum():8.4f}\t{trainacc:8.4f}\t{trainauroc:8.4f}\t{trainauprc:8.4f}\t{testacc:8.4f}\t{testauroc:8.4f}\t{testauprc:8.4f}')

    # plot code starts here
    if plotoption != 'noplotsave':
        alphas_n = alphas[dwavesolutionidx] # plot only the requested
        b = eval_offset_avg(alphas_n, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
        score = eval_classifier(data, alphas_n, data, label, gamma, b)
        scoretest = eval_classifier(datatest, alphas_n, data, label, gamma, b)

        # roc curve
        plt.figure(figsize=(6.4,3.2))
        plt.subplot(1,2,1)
        plt.subplots_adjust(top=.95, right=.95, bottom=.15, wspace=.3)
        fpr, tpr, thresholds = roc_curve(labeltest, scoretest)
        auroc = roc_auc_score(labeltest, scoretest)
        plt.plot(fpr, tpr, label='AUROC=%0.3f' % auroc, color='g')
        plt.fill_between(fpr, tpr, alpha=0.2, color='g', step='post')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver Operating Curve')
        plt.legend(loc="lower right")
        # pr curve
        plt.subplot(1,2,2)
        precision, recall, _ = precision_recall_curve(labeltest, scoretest)
        auprc = auc(recall, precision)
        plt.step(recall, precision, color='g', where='post',
            label='AUPRC=%0.3f' % auprc)
        plt.fill_between(recall, precision, alpha=0.2, color='g', step='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        #plt.title('PR curve')
        plt.legend(loc="lower right")

        # save the data for gnuplot
        savename = f'{path.replace("/","_")}{dwavesolutionidx}'
        #with open('results/rocpr_curves/'+savename,'w') as out:
        with open(path_in+savename,'w') as out:
            out.write(f'AUROC\t{auroc:0.3f}\t# ROC:FPR,TPR\n')
            assert len(fpr) == len(tpr)
            for i in range(len(fpr)):
                out.write(f'{fpr[i]}\t{tpr[i]}\n')
            out.write(f'\n\nAUPRC\t{auprc:0.3f}\t# PRC:Recall,Precision\n')
            assert len(recall) == len(precision)
            for i in range(len(recall)):
                out.write(f'{recall[i]}\t{precision[i]}\n')
            print(f'saved data for {savename}')

        if plotoption == 'saveplot':
            savefigname = path_in+savename+'.pdf'
            plt.savefig(savefigname)
            print(f'saved as {savefigname}')
        else:
            plt.show()

    return np.average(trainacc_all), np.average(trainauroc_all), np.average(trainauprc_all) ,np.average(testacc_all), np.average(testauroc_all), np.average(testauprc_all)


# previously we used AUROC and AUPRC as metrics, now replaced by F1 score -> this function here is maybe not necessary anymore?
# returns the average of the results of all single classifiers and the result of the averaged classifer (for train and validation data)
# if validation = False, the "validation" results are all 0 (except energy which is only computed and returned for train data anyway)
# showplot option only works for toy model data, thus False by default
def eval_run_rocpr_curves_embedding_train(path_data_key,path_in,cs,ta,idx,max_alphas=0,validation=False,showplot=False):
    regex = '.*run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_E=(.*)_gamma=([^/]*)/result_couplers.*/?$'
    match = re.search(regex, path_in)

    path = path_in + ('/' if path_in[-1] != '/' else '')
    data_key = match.group(1)
    B = int(match.group(2))
    K = int(match.group(3))
    xi = float(match.group(4))
    gamma = float(match.group(6))
    E = int(match.group(5))
    data,label = loaddataset(path_data_key+data_key)
    if validation:
        data_val,label_val = loaddataset(path_data_key+data_key.replace('calibtrain','calibval'))

    dwavesolutionidx=0
    C=(B**np.arange(K-E)).sum()

    alphas_file = path+f'embedding{idx}_rcs{cs}_ta{ta}_alphas{data_key}_gamma={gamma}.npy'
    energy_file = path+f'embedding{idx}_rcs{cs}_ta{ta}_result.dat'
    energy_file_dat=np.genfromtxt(energy_file)
    energies=np.transpose(energy_file_dat[1:])[1]
    if not os.path.isfile(alphas_file):
        print('result '+alphas_file+' doesnt exist, exiting')
        sys.exit(-1)

    alphas = list(np.atleast_2d(np.load(alphas_file)))
    for i in reversed(range(len(alphas))):
        if all(np.logical_or(np.isclose(alphas[i],0),np.isclose(alphas[i],C))) or np.isclose(np.sum(alphas[i] * (C-alphas[i])),0): #remove cases where no support vectors are found (all alpha=0 or only slack variables all alpha=0 or alpha=C) -> numerical recipes Eq. 16.5.24
            print(f'Deleting alphas[{i}].')
            del alphas[i]
    alphas = np.array(alphas)
    if max_alphas == 0 or max_alphas > len(alphas):
        nalphas = len(alphas)
    else:
        nalphas = max_alphas
    assert len(data) == alphas.shape[1], "alphas do not seem to be for the right data set?)"

    trainacc_all=np.zeros([nalphas])
    trainauroc_all=np.zeros([nalphas])
    trainauprc_all=np.zeros([nalphas])
    testacc_all=np.zeros([nalphas])
    testauroc_all=np.zeros([nalphas])
    testauprc_all=np.zeros([nalphas])
    alphas_avg = np.zeros(len(alphas[0]))
    energies2 = np.zeros([nalphas])

    for i in range(nalphas):
        alphas_n = alphas[i]
        alphas_avg += alphas_n
        b = eval_offset_avg(alphas_n, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
        score = eval_classifier(data, alphas_n, data, label, gamma, b)
        trainacc,trainauroc,trainauprc = eval_acc_auroc_auprc(label,score)

        trainacc_all[i] = trainacc
        trainauroc_all[i] = trainauroc
        trainauprc_all[i] = trainauprc
        energies2[i] = compute_energy(alphas_n, data, label, gamma, xi)

        if validation:
            scoretest = eval_classifier(data_val, alphas_n, data, label, gamma, b)
            testacc,testauroc,testauprc = eval_acc_auroc_auprc(label_val,scoretest)

            testacc_all[i] = testacc
            testauroc_all[i] = testauroc
            testauprc_all[i] = testauprc

    alphas_avg = alphas_avg/nalphas
    b = eval_offset_avg(alphas_avg, data, label, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file
    score = eval_classifier(data, alphas_avg, data, label, gamma, b)
    trainacc,trainauroc,trainauprc = eval_acc_auroc_auprc(label,score)
    testacc = 0
    testauroc = 0
    testauprc = 0
    if validation:
        scoretest = eval_classifier(data_val, alphas_avg, data, label, gamma, b)
        testacc,testauroc,testauprc = eval_acc_auroc_auprc(label_val,scoretest)
    energy = compute_energy(alphas_avg, data, label, gamma,xi)
    print(alphas_avg)

    # if showplot:
    #     if validation:
    #         plot_result_val(alphas_avg, data, label, gamma, C, [-4,4], title=f'ta = {ta}, rcs = {cs:0.1f}, emb = {idx}, energy {energy:0.4f},\nacc = {trainacc:0.4f}, auroc = {trainauroc:0.4f},  auprc = {trainauprc:0.4f},\ntestacc = {testacc:0.4f}, testauroc = {testauroc:0.4f},  testauprc = {testauprc:0.4f}', filled=True, validation=True, data_val=data_val, label_val=label_val)
    #     else:
    #         plot_result_val(alphas_avg, data, label, gamma, C, [-4,4], title=f'ta = {ta}, rcs = {cs:0.1f}, emb = {idx}, energy {energy:0.4f},\nacc = {trainacc:0.4f}, auroc = {trainauroc:0.4f},  auprc = {trainauprc:0.4f}', filled=True)

    return np.average(trainacc_all), np.average(trainauroc_all), np.average(trainauprc_all), trainacc, trainauroc, trainauprc, np.average(energies2), energy, np.average(testacc_all), np.average(testauroc_all), np.average(testauprc_all), testacc, testauroc, testauprc


def compute_energy(alphas, data, label, gamma, xi):
    energy = 0
    sv = alphas*label
    for n in range(len(data)):
        for m in range(len(data)):
            energy += 0.5*sv[n]*sv[m]*kernel(data[n],data[m],gamma)
    energy -= np.sum(alphas)
    energy += 0.5*xi*(np.sum(sv))**2 # NOTE: the 1/2 here is only there because we have it in the svm paper this way
    return energy


# for plotting the toy model data with classifier and support vectors
def plot_result(alphas, data, label, gamma, C, xylim=[-2.0, 2.0], notitle=False, filled=False, title = "", save=""):
    b = eval_offset_avg(alphas, data, label, gamma, C)
    result = np.sign(eval_classifier(data, alphas, data, label, gamma, b))
    w = np.array([0.0,0.0])
    for l in range(0,len(label)):
        w += alphas[l] * label[l] * data[l]
    len_w = np.sqrt(w[0]**2+w[1]**2)
    w = w/len_w
    print(f'w = ( {w[0]} , {w[1]} )')
    w = -b/len_w*w
    print(f'b = {b}')

    xsample = np.arange(xylim[0], xylim[1]+.01, .05)
    x1grid, x2grid = np.meshgrid(xsample, xsample)
    X = np.vstack((x1grid.ravel(), x2grid.ravel())).T # ...xD for kernel contraction
    FX = eval_classifier(X, alphas, data, label, gamma, b).reshape(len(xsample), len(xsample))

    #plt.pcolor(x1grid, x2grid, FX, cmap='coolwarm')
    plt.contourf(x1grid, x2grid, FX, [-5.,-4.5,-4.,-3.5,-3.,-2.5,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.], cmap='seismic', extend='both',alpha=0.8)
    plt.contour(x1grid, x2grid, FX, [0.], linewidths=3, colors='black')
    plt.arrow(0,0,w[0],w[1], linewidth=2, length_includes_head=True)
    ax=plt.gca()
    ax.set_aspect('equal')
    if not title == "":
        plt.title(title)
    if not notitle:
        plt.title('acc = %g' % ((result == label).sum() / len(label)))

    if not filled:
        plt.scatter(data[result==1][:,0], data[result==1][:,1], c='r', marker=(8,2,0), linewidth=0.5)
        plt.scatter(data[result!=1][:,0], data[result!=1][:,1], c='b', marker='+', linewidth=1)
        plt.scatter(data[label==1][:,0], data[label==1][:,1], edgecolors='r', marker='s', linewidth=0.5, facecolors='none')
        plt.scatter(data[label!=1][:,0], data[label!=1][:,1], edgecolors='b', marker='o', linewidth=1, facecolors='none')
    else:
        plt.scatter(data[label==1][:,0], data[label==1][:,1], edgecolors=cols.CSS4_COLORS['darkorange'], marker='s', linewidth=1, facecolors=cols.CSS4_COLORS['red'])
        plt.scatter(data[label!=1][:,0], data[label!=1][:,1], edgecolors=cols.CSS4_COLORS['deepskyblue'], marker='D', linewidth=1, facecolors=cols.CSS4_COLORS['blue'])
        plt.scatter(data[alphas>0][:,0], data[alphas>0][:,1], edgecolors='k', marker='o', linewidth=2, facecolors='none')
        plt.scatter(data[alphas==C][:,0], data[alphas==C][:,1], edgecolors='w', marker='o', linewidth=2, facecolors='none')

    plt.xlim(*xylim)
    plt.ylim(*xylim)
    plt.show()

    if not save == "":
        plt.savefig(save+".svg")


def compute_alphas_avg_and_b(training_data, training_labels, alphas, gamma, actual_C, bias_strategy):
    # Compute the mean of the alphas
    alphas_avg = np.mean(alphas, axis=0)

    b = None
    if bias_strategy == 'zero':
        b = 0
    elif bias_strategy == 'paper':
        b = eval_offset_avg(alphas_avg, training_data, training_labels, gamma, actual_C)  # NOTE: this is NAN if no support vectors were found
    elif bias_strategy == 'post-selection':
        candidate_b_values = np.linspace(-10, 10, 201)
        accuracy_vals_on_training = []
        for b_value in candidate_b_values:
            predicted_labels = np.array(
                predict(training_data, training_data, training_labels, gamma, alphas_avg, b_value)
            )
            accuracy_vals_on_training.append((predicted_labels == training_labels).sum() / len(training_labels))
        b = candidate_b_values[np.argmax(accuracy_vals_on_training)]
    else:
        print(f'Unknown bias selection strategy: {bias_strategy}')

    return alphas_avg, b


# def predict(datatest,data,label,alphas,path_in):
def predict(test_data, training_data, training_labels, gamma, alphas_avg, b):
    # regex = 'run([^/]*)_B=(.*)_K=(.*)_xi=(.*)_E=(.*)_gamma=([^/]*)/result_couplers.*/?$'
    # match = re.search(regex, path_in)
    #
    # path = path_in + ('/' if path_in[-1] != '/' else '')
    # data_key = match.group(1)
    # B = int(match.group(2))
    # K = int(match.group(3))
    # xi = float(match.group(4))
    # gamma = float(match.group(6))
    # E = int(match.group(5))

    # C=(B**np.arange(K-E)).sum()
    #
    # # Compute the mean of the alphas
    # alphas_avg=np.mean(alphas,axis=0)
    #
    # b = eval_offset_avg(alphas_avg, training_data, training_labels, gamma, C) # NOTE: this is NAN if no support vectors were found, see TODO file

    scoretest = eval_classifier(test_data, alphas_avg, training_data, training_labels, gamma, b)

    return np.sign(scoretest), scoretest


def kernel(xn, xm, gamma=-1): # here (xn.shape: NxD, xm.shape: ...xD) -> Nx...
    if gamma == -1:
        return xn @ xm.T
    xn = np.atleast_2d(xn)
    xm = np.atleast_2d(xm)
    return np.exp(-gamma * np.sum((xn[:,None] - xm[None,:])**2, axis=-1))  # (N,1,D) - (1,...,D) -> (N,...,D) -> (N,...); see Hsu guide.pdf for formula


# B = base
# K = number of qubits per alpha
# E = shift of exponent
# decode binary -> alpha
def decode(binary, B=10, K=3, E=0):
    N = len(binary) // K
    Bvec = float(B) ** (np.arange(K)-E)
    return np.fromiter(binary,float).reshape(N,K) @ Bvec


# encode alpha -> binary with B and K (for each n, the binary coefficients an,k such that sum_k an,k B**k is closest to alphan)
def encode(alphas, B=10, K=3, E=0): # E allows for encodings with floating point numbers (limited precision of course)
    N = len(alphas)
    Bvec = float(B) ** (np.arange(K)-E) # B^(0-E) B^(1-E) B^(2-E) ... B^(K-1-E)
    allvals = np.array(list(map(lambda n : np.fromiter(bin(n)[2:].zfill(K),float,K), range(2**K)))) @ Bvec # [[0,0,0],[0,0,1],...] @ [1, 10, 100]
    return ''.join(list(map(lambda n : bin(n)[2:].zfill(K),np.argmin(np.abs(allvals[:,None] - alphas), axis=0))))


def encode_as_vec(alphas, B=10, K=3, E=0):
    return np.fromiter(encode(alphas,B,K,E), float)


def loaddataset(datakey):
    dataset = np.loadtxt(datakey, dtype=float, skiprows=1)
    return dataset[:,2:], dataset[:,1]  # data, labels


def save_json(filename, var):
    with open(filename, 'w') as f:
        f.write(str(json.dumps(var, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)))


def eval_classifier(x, alphas, data, label, gamma, b=0):  # evaluates the distance to the hyper plane according to 16.5.32 on p. 891 (Numerical Recipes); sign is the assigned class; x.shape = ...xD
    return np.sum((alphas * label)[:, None] * kernel(data, x, gamma), axis=0) + b


def eval_on_sv(x, alphas, data, label, gamma, C):
    return np.sum((alphas * (C-alphas) * label)[:,None] * kernel(data, x, gamma), axis=0)


def eval_offset_search(alphas, data, label, gamma, C, useavgforb=True): # search for the best offset
    maxacc=0
    b1=-9
    for i in np.linspace(-9,9,500):
        acc = accuracy_score(label,np.sign(eval_classifier(data, alphas, data, label, gamma, i)))
        if acc > maxacc:
            maxacc = acc
            b1=i
    maxacc=0
    b2=9
    reversed_space=np.linspace(-9,9,500)[::-1]
    for i in reversed_space:
        acc = accuracy_score(label,np.sign(eval_classifier(data, alphas, data, label, gamma, i)))
        if acc > maxacc:
            maxacc = acc
            b2=i
    return (b1+b2)/2


def eval_offset_MM(alphas, data, label, gamma, C, useavgforb=True): # evaluates offset b according to 16.5.37 (Mangasarian-Musicant variant) NOTE: does not seem to work with integer/very coarsely spaced alpha!
    return np.sum(alphas*label)


def eval_offset_avg(alphas, data, label, gamma, C, useavgforb=True): # evaluates offset b according to 16.5.33
    cross = eval_classifier(data, alphas, data, label, gamma) # cross[i] = sum_j aj yj K(xj, xi) (error in Numerical Recipes)
    if useavgforb:
        return np.sum(alphas * (C-alphas) * (label - cross)) / np.sum(alphas * (C-alphas))
        #return np.sum(label - cross) / num_sv
    else:  # this is actually not used, but we did a similar-in-spirit implementation in eval_finaltraining_avgscore.py
        if np.isclose(np.sum(alphas * (C-alphas)),0):
            print('no support vectors found, discarding this classifer')
            return np.nan
        bcandidates = [np.sum(alphas * (C-alphas) * (label - cross)) / np.sum(alphas * (C-alphas))]  # average according to NR should be the first candidate
        crosssorted = np.sort(cross)
        crosscandidates = -(crosssorted[1:] + crosssorted[:-1])/2  # each value between f(xi) and the next higher f(xj) is a candidate
        bcandidates += sorted(crosscandidates, key=lambda x:abs(x - bcandidates[0]))  # try candidates closest to the average first
        bnumcorrect = [(label == np.sign(cross + b)).sum() for b in bcandidates]
        return bcandidates[np.argmax(bnumcorrect)]


def eval_acc_auroc_auprc(label, score):  # score is the distance to the hyper plane (output from eval_classifier)
    precision,recall,_ = precision_recall_curve(label, score)
    return accuracy_score(label,np.sign(score)), roc_auc_score(label,score), auc(recall,precision)


################ This I/O functions are provided by http://hyperlabelme.uv.es/index.html ################
def dataread(filename):
    lasttag = 'description:'
    # Open file and locate lasttag
    f = open(filename, 'r')
    nl = 1
    for line in f:
        if line.startswith(lasttag): break
        nl += 1
    f.close()

    # Read data
    data = np.loadtxt(filename, delimiter=',', skiprows=nl)
    Y = data[:, 0]
    X = data[:, 1:]
    # Separate train/test
    Xtest = X[Y < 0, :]
    X = X[Y >= 0, :]
    Y = Y[Y >= 0, None]

    return X, Y, Xtest


def datawrite(path,method, dataset, Yp):
    filename = '{0}{1}_predictions.txt'.format(path, dataset)
    res = True
    try:
        with open(filename, mode='w') as f:
            f.write('{0} {1}'.format(method, dataset))
            for v in Yp:
                f.write(' {0}'.format(str(v)))
            f.write('\n')
    except Exception as e:
        print('Error', e)
        res = False
    return res
################


def write_samples(X, Y,path):
    f = open(path,"w")
    f.write("id label data \n")
    for i in range(0,X.shape[0]):
        f.write(str(i)+" ")
        if(Y[i]==1):
            f.write("-1 ")
        else:
            f.write("1 ")
        for j in range(0,X.shape[1]):
            f.write(str(X[i,j])+" ")
        f.write("\n")
    f.close()
