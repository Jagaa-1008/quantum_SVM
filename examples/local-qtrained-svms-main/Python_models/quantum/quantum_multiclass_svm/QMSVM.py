import numpy as np

from quantum.quantum_multiclass_svm.QMSVM_lib import get_training_data_and_labels, gen_svm_qubos_multiclass, \
    dwave_run_embedding_multiclass, solutions_combination, eval_classifier_multiclass


class QMSVM:
    def __init__(self, config, seed=None, max_model_size=0):
        self.kernel_type = config['kernel_type']
        self.gamma = -1 if self.kernel_type == 0 else config['gamma']
        self.B = config['binary_vars_num']
        self.beta = config['multicl_reg']
        self.mu = config['penalty_coef']
        self.embeddings_dir = config['embeddings_dir']

        self.seed = seed
        self.max_model_size = max_model_size

        self.max_min_ratio = 1000

        self.training_data = None
        self.training_labels = None

        self.validation_data = None
        self.validation_labels = None

        self.classes = []
        self.C_classes_num = None
        self.training_labels_indices = []

        self.annealing_times = [200]
        self.chain_strengths = [1]
        self.num_reads = 1000
        self.max_results = 100

        self.Q = None

        self.taus = []
        self.taus_comb = []

    def train(self, data, labels, unique_labels):
        self.training_data, self.training_labels = \
            get_training_data_and_labels(self.max_model_size, data, labels, self.seed)
        self.validation_data, self.validation_labels = data, labels

        assert len(unique_labels) >= 2

        unique_training_labels = sorted(np.unique(self.training_labels))
        if len(unique_training_labels) != len(unique_labels):
            print(f'Warning: in QMSVM training, the labels in the training set ({unique_training_labels}) and '
                  f'the labels in the validation set ({unique_labels}) do not coincide')

        self.classes = unique_labels
        self.C_classes_num = len(self.classes)
        self.training_labels_indices = [
            self.classes.index(training_label) for training_label in self.training_labels
        ]

        self.Q, qubo_nodes, qubo_couplers = gen_svm_qubos_multiclass(self.training_data, self.training_labels_indices,
                                                                     self.C_classes_num, self.B, self.beta, self.mu,
                                                                     self.gamma, self.max_min_ratio)
        self.taus = dwave_run_embedding_multiclass(qubo_nodes, qubo_couplers, self.C_classes_num, self.B,
                                                   self.embeddings_dir, annealing_times=self.annealing_times,
                                                   chain_strengths=self.chain_strengths, num_reads=self.num_reads,
                                                   profile='europe', max_results=self.max_results)[0]
        self.taus_comb = solutions_combination(self.training_data, self.taus, self.validation_data,
                                               self.validation_labels, self.C_classes_num, self.gamma, self.classes,
                                               comb='softmax')

    def predict(self, test_instance):
        test_data = np.array([test_instance])

        decision_f = eval_classifier_multiclass(test_data, self.taus_comb, self.training_data, self.C_classes_num,
                                                self.gamma)
        decision_ind = np.argmax(decision_f, axis=0).tolist()
        test_label = self.classes[decision_ind[0]]

        return test_label
