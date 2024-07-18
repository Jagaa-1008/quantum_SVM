import numpy as np

from quantum.quantum_binary_svm.QBSVM_lib import split_data_and_labels, gen_svm_qubos, dwave_run_embedding, \
    compute_alphas_avg_and_b, predict


class QBSVM:
    def __init__(self, config, seed=None, max_model_size=0):
        self.kernel_type = config['kernel_type']
        self.gamma = -1 if self.kernel_type == 0 else config['gamma']
        self.C = config['cost']
        self.B = config['base']
        self.K = config['binary_vars_num']
        self.E = 0
        self.xi = config['penalty_coef']
        self.embeddings_dir = config['embeddings_dir']
        self.actual_C = (self.B ** np.arange(self.K - self.E)).sum()

        self.seed = seed
        self.max_model_size = max_model_size

        self.split_training_data = []
        self.split_training_labels = []

        self.classes = []
        self.classes_mapping = {}

        self.annealing_times = [200]
        self.chain_strengths = [1]
        self.num_reads = 1000
        self.max_results = 100

        self.Q_list = []

        self.alphas_list = []
        self.alphas_avg_list = []
        self.b_list = []

        self.bias_strategy = 'post-selection'  # in ['zero', 'paper', 'post-selection']

    def train(self, data, labels, unique_labels):
        self.split_training_data, self.split_training_labels = \
            split_data_and_labels(self.max_model_size, data, labels, self.seed)

        assert len(unique_labels) == 2

        self.classes = unique_labels
        if not all(y in (-1, +1) for y in self.classes):
            self.split_training_labels = [np.where(labels == self.classes[0], -1, +1)
                                          for labels in self.split_training_labels]
            self.classes_mapping = {-1: self.classes[0], +1: self.classes[1]}

        for training_data, training_labels in zip(self.split_training_data, self.split_training_labels):
            Q, qubo_nodes, qubo_couplers = gen_svm_qubos(training_data, training_labels, self.B, self.K,
                                                         self.xi, self.gamma, self.E)
            self.Q_list.append(Q)

            alphas = dwave_run_embedding(qubo_nodes, qubo_couplers, self.B, self.K, self.E, self.embeddings_dir,
                                         annealing_times=self.annealing_times, chain_strengths=self.chain_strengths,
                                         num_reads=self.num_reads, profile='europe', max_results=self.max_results)[0]
            self.alphas_list.append(alphas)

            alphas_avg, b = compute_alphas_avg_and_b(training_data, training_labels, alphas, self.gamma, self.actual_C,
                                                     self.bias_strategy)
            self.alphas_avg_list.append(alphas_avg)
            self.b_list.append(b)

    def predict(self, test_instance):
        test_data = np.array([test_instance])

        scores = [predict(test_data, training_data, training_labels, self.gamma, alphas_avg, b)[1][0]
                  for training_data, training_labels, alphas_avg, b
                  in zip(self.split_training_data, self.split_training_labels, self.alphas_avg_list, self.b_list)]

        test_label = np.sign(np.average(scores))
        if len(self.classes_mapping) != 0:
            test_label = self.classes_mapping[test_label]

        return test_label
