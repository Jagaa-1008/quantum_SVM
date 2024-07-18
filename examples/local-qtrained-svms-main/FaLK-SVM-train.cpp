#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "FaLK-SVM/defs.h"
#include "FaLK-SVM/svm.h"
#include "FaLK-SVM/cover_tree.h"
#include "FaLK-SVM/par_est.h"
#include "FaLK-SVM/per_eval.h"
#include "FaLK-SVM/fast_lsvm.h"
#include "FaLK-SVM/knn.h"


void exit_with_help() {
    printf(
            "Usage: FaLK-SVM-train [options] input_dataset_file model_file\n"
            "options:\n"
            "-R relaxation parameter for cover trees (default 1)\n"
            "-K neighborhood size k\n"
            "-P assignment neighborhood size as fraction of K such that: K'=K*P (default 0.5)\n"
            "-q python_svm: whether to use the default C++ SVMs (0) or the Python ones (1)\n"
            "-t kernel_type: set the type of kernel function (default 2)\n"
            "\t0 -- linear: u'*v\n"
            "\t1 -- polynomial: (gamma*u'*v + coef0)^degree [not supported by the Python SVMs]\n"
            "\t2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
            "-d degree: set degree in kernel function (default 3)\n"
            "-g gamma: set gamma (locally) in kernel function, negative values for estimation based on histogram of distances (default = -0.5 for RBF, default = 1 for polynomial kernels)\n"
            "-r coef0: set coef0 in kernel function (default 0)\n"
            "-c cost: set the parameter C of the SVMs (default 1)\n"

            "-m cachesize: set cache memory size in MB for the C++ SVMs (default 100)\n"
            "-wi weight: set the parameter C of class i to weight*C, for the C++ SVMs (default 1)\n"
            "-p probability_estimates: whether to train the C++ local models for probability estimates, 0 or 1 (default 0)\n"

            "-z python_svm_type: type of Python SVMs:\n"
            "\t0 -- quantum for binary classification\n"
            "\t1 -- quantum for multiclass classification\n"
            "\t2 -- classical for multiclass classification (Crammer-Singer)\n"
            "-b python_svm_base: base used to encode the coefficients (Python quantum SVMs)\n"
            "-a python_svm_binary_vars_num: number of binary variables used to encode each coefficient (Python quantum SVMs)\n"
            "-i python_svm_penalty_coef: multiplier for penalty terms (ksi, mu) in Python quantum SVMs\n"
            "-e python_svm_multicl_reg: regularization parameter (beta) for multiclass Python quantum SVMs\n"
            "-n python_svm_embeddings_dir: directory where to store/load embeddings for the Python quantum SVMs\n"

            "-v n: n-fold cross validation modality\n"

            "-L local model selection (default 0):\n"
            "\t0 -- no local model selection \n"
            "\t1 -- local model selection on -M local models on (K, G, C) for the C++ SVMs, (K, B, A, I, E, G, C) for the Python SVMs [all the values related to the chosen SVMs (C++/Python) must be set, although not used in practice]\n"
            "-N K values for local grid model selection separated by ':' \n"
            "-G g values for local grid model selection separated by ':' \n"
            "-C c values for local grid model selection separated by ':' \n"

            "-B b values for local grid model selection separated by ':' \n"
            "-A a values for local grid model selection separated by ':' \n"
            "-I i values for local grid model selection separated by ':' \n"
            "-E e values for local grid model selection separated by ':' \n"

            "-M number of local models used for local model selection \n"
            "-o number of folds used in the cross validation for the local model selection (default 10) \n"
            "-l use all kp samples (1) or half of them (0) to evaluate the models in the cross validation for the local model selection (default 0) \n"

            "-s seed: (non-negative) seed for initialising the random numbers generator (default 27)\n"
            "-S silent mode (default 0)\n"
    );
    exit(1);
}

void parse_command_line(int argc, char **argv, struct svm_parameter *param, struct lkm_parameter *lkm_param,
                        unsigned int &seed, char *train_file_name, char *model_file_name);

int main(int argc, char **argv) {
    char train_file_name[1024];
    char model_file_name[1024];

    struct problem *train = Malloc(struct problem, 1);
    if (train == NULL) MallocError("train");
    struct svm_parameter *param = Malloc(struct svm_parameter, 1);
    if (param == NULL) MallocError("param");
    struct lkm_parameter *lkm_param = Malloc(struct lkm_parameter, 1);
    if (lkm_param == NULL) MallocError("lkm_param");
    unsigned int seed;

    parse_command_line(argc, argv, param, lkm_param, seed, train_file_name, model_file_name);
    srand(seed);

    read_problem(train_file_name, train);

    sFastLSVM *flm = new sFastLSVM(lkm_param, train, model_file_name);

    if (lkm_param->nr_fold > 1) {
        double *res = flm->cross_validation();
        PerEval *perEval = new PerEval(train->y, train->l);
        if (param->svm_type == C_SVC) {
            if (lkm_param->per_meas < 4)
                perEval->calcAll(res, lkm_param->per_meas);
            if (lkm_param->per_meas == 4)
                perEval->calcLift20P(res, true);
        } else if (param->svm_type == EPSILON_SVR)
            perEval->calcMSE(res, 1);
        delete perEval;
        free(res);
    } else {
        flm->train();
        flm->save();
    }

    delete flm;

    svm_destroy_param(param);
    free(param);

    destroy_lkm_param(lkm_param);
    free(lkm_param);

    free(train->y);
    free(train->x);
    free(train);

    return 0;
}


void parse_command_line(int argc, char **argv, struct svm_parameter *param, struct lkm_parameter *lkm_param,
                        unsigned int &seed, char *train_file_name, char *model_file_name) {
    int i;

    // default values
    param->python_svm = false;
    param->svm_type = C_SVC;
    param->kernel_type = RBF;
    param->degree = 3;
    param->gamma = -0.5;    // 1/k
    param->coef0 = 0;
    param->nu = 0.5;
    param->cache_size = 200;
    param->C = 100.0;
    param->eps = 1e-3;
    param->p = 0.1;
    param->shrinking = 1;
    param->probability = 0;
    param->h_warning = false;
    param->nr_weight = 0;
    param->weight_label = NULL;
    param->weight = NULL;
    param->python_svm_class = nullptr;
    param->python_svm_type = 0;
    param->python_svm_base = 2;
    param->python_svm_binary_vars_num = 2;
    param->python_svm_penalty_coef = 1.0;
    param->python_svm_multicl_reg = 1.0;
    param->python_svm_embeddings_dir = Malloc(char, 1);
    param->python_svm_embeddings_dir[0] = '\0';
    lkm_param->nr_fold = -1;
    lkm_param->k = 1000;
    lkm_param->kp = 250;
    lkm_param->base = 1;
    lkm_param->svm_par = param;
    lkm_param->local_model_selection = 0;
    lkm_param->conservative_mr = false;
    lkm_param->opt = 0;
    lkm_param->nr_folds_loc_models_selection = 10;
    lkm_param->kp_ratio = 0.5;
    lkm_param->eval_on_kp = false;
    lkm_param->unbalanced = 0;
    lkm_param->loc_prob_bil_th = 0.33;
    lkm_param->n_k_lms = 0;
    lkm_param->per_meas = 0;
    lkm_param->silent = false;

    seed = 27;

    // parse options
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-')
            break;

        if (++i >= argc)
            exit_with_help();
        switch (argv[i - 1][1]) {
            case 'R':
                lkm_param->base = atoi(argv[i]);
                break;
            case 'K':
                lkm_param->k = atoi(argv[i]);
                break;
            case 'P':
                lkm_param->kp_ratio = atof(argv[i]);
                break;
            case 'q':
                param->python_svm = (atoi(argv[i]) == 1);
                break;
            case 't':
                param->kernel_type = atoi(argv[i]);
                break;
            case 'd':
                param->degree = atoi(argv[i]);
                break;
            case 'g':
                param->gamma = atof(argv[i]);
                break;
            case 'r':
                param->coef0 = atof(argv[i]);
                break;
            case 'c':
                param->C = atof(argv[i]);
                break;
            case 'm':
                param->cache_size = atof(argv[i]);
                break;
            case 'w':
                ++param->nr_weight;
                param->weight_label = (int *) realloc(param->weight_label, sizeof(int) * param->nr_weight);
                param->weight = (double *) realloc(param->weight, sizeof(double) * param->nr_weight);
                param->weight_label[param->nr_weight - 1] = atoi(&argv[i - 1][2]);
                param->weight[param->nr_weight - 1] = atof(argv[i]);
                break;
            case 'p':
                param->probability = atoi(argv[i]);
                break;
            case 'z':
                param->python_svm_type = atoi(argv[i]);
                break;
            case 'b':
                param->python_svm_base = atoi(argv[i]);
                break;
            case 'a':
                param->python_svm_binary_vars_num = atoi(argv[i]);
                break;
            case 'i':
                param->python_svm_penalty_coef = atof(argv[i]);
                break;
            case 'e':
                param->python_svm_multicl_reg = atof(argv[i]);
                break;
            case 'n':
                param->python_svm_embeddings_dir = Realloc(param->python_svm_embeddings_dir, char, strlen(argv[i]) + 1);
                strcpy(param->python_svm_embeddings_dir, argv[i]);
                break;
            case 'v':
                lkm_param->nr_fold = atoi(argv[i]);
                if (lkm_param->nr_fold < 2) {
                    fprintf(stderr, "n-fold cross validation: n must >= 2\n");
                    exit_with_help();
                }
                break;
            case 'L':
                lkm_param->local_model_selection = atoi(argv[i]);
                break;
            case 'N': {
                int n_val = 1;
                int u = 0;

                for (u = 0; u < (int) strlen(argv[i]); u++)
                    if (argv[i][u] == ':')
                        n_val++;

                lkm_param->k_lms = Malloc(int, n_val);

                u = 0;
                char str[1024];
                unsigned int l = 0;
                int ll = 0;

                while (u < n_val) {
                    if (argv[i][l] == ':' || l == strlen(argv[i])) {
                        str[ll] = '\0';
                        lkm_param->k_lms[u++] = atoi(str);
                        l++;
                        ll = 0;
                    } else
                        str[ll++] = argv[i][l++];
                }
                lkm_param->n_k_lms = n_val;
                break;
            }
            case 'G': {
                int n_val = 1;
                int u = 0;

                for (u = 0; u < (int) strlen(argv[i]); u++)
                    if (argv[i][u] == ':')
                        n_val++;

                lkm_param->g_lms = Malloc(double, n_val);

                u = 0;
                char str[1024];
                unsigned int l = 0;
                int ll = 0;

                while (u < n_val) {
                    if (argv[i][l] == ':' || l == strlen(argv[i])) {
                        str[ll] = '\0';
                        lkm_param->g_lms[u++] = atof(str);
                        l++;
                        ll = 0;
                    } else
                        str[ll++] = argv[i][l++];
                }
                lkm_param->n_g_lms = n_val;
                break;
            }
            case 'C': {
                int n_val = 1;
                int u = 0;

                for (u = 0; u < (int) strlen(argv[i]); u++)
                    if (argv[i][u] == ':')
                        n_val++;

                lkm_param->c_lms = Malloc(double, n_val);

                u = 0;
                char str[1024];
                unsigned int l = 0;
                int ll = 0;

                while (u < n_val) {
                    if (argv[i][l] == ':' || l == strlen(argv[i])) {
                        str[ll] = '\0';
                        lkm_param->c_lms[u++] = atof(str);
                        l++;
                        ll = 0;
                    } else
                        str[ll++] = argv[i][l++];
                }
                lkm_param->n_c_lms = n_val;
                break;
            }
            case 'B': {
                int n_val = 1;
                int u = 0;

                for (u = 0; u < (int) strlen(argv[i]); u++)
                    if (argv[i][u] == ':')
                        n_val++;

                lkm_param->b_lms = Malloc(int, n_val);

                u = 0;
                char str[1024];
                unsigned int l = 0;
                int ll = 0;

                while (u < n_val) {
                    if (argv[i][l] == ':' || l == strlen(argv[i])) {
                        str[ll] = '\0';
                        lkm_param->b_lms[u++] = atoi(str);
                        l++;
                        ll = 0;
                    } else
                        str[ll++] = argv[i][l++];
                }
                lkm_param->n_b_lms = n_val;
                break;
            }
            case 'A': {
                int n_val = 1;
                int u = 0;

                for (u = 0; u < (int) strlen(argv[i]); u++)
                    if (argv[i][u] == ':')
                        n_val++;

                lkm_param->a_lms = Malloc(int, n_val);

                u = 0;
                char str[1024];
                unsigned int l = 0;
                int ll = 0;

                while (u < n_val) {
                    if (argv[i][l] == ':' || l == strlen(argv[i])) {
                        str[ll] = '\0';
                        lkm_param->a_lms[u++] = atoi(str);
                        l++;
                        ll = 0;
                    } else
                        str[ll++] = argv[i][l++];
                }
                lkm_param->n_a_lms = n_val;
                break;
            }
            case 'I': {
                int n_val = 1;
                int u = 0;

                for (u = 0; u < (int) strlen(argv[i]); u++)
                    if (argv[i][u] == ':')
                        n_val++;

                lkm_param->i_lms = Malloc(int, n_val);

                u = 0;
                char str[1024];
                unsigned int l = 0;
                int ll = 0;

                while (u < n_val) {
                    if (argv[i][l] == ':' || l == strlen(argv[i])) {
                        str[ll] = '\0';
                        lkm_param->i_lms[u++] = atoi(str);
                        l++;
                        ll = 0;
                    } else
                        str[ll++] = argv[i][l++];
                }
                lkm_param->n_i_lms = n_val;
                break;
            }
            case 'E': {
                int n_val = 1;
                int u = 0;

                for (u = 0; u < (int) strlen(argv[i]); u++)
                    if (argv[i][u] == ':')
                        n_val++;

                lkm_param->e_lms = Malloc(int, n_val);

                u = 0;
                char str[1024];
                unsigned int l = 0;
                int ll = 0;

                while (u < n_val) {
                    if (argv[i][l] == ':' || l == strlen(argv[i])) {
                        str[ll] = '\0';
                        lkm_param->e_lms[u++] = atoi(str);
                        l++;
                        ll = 0;
                    } else
                        str[ll++] = argv[i][l++];
                }
                lkm_param->n_e_lms = n_val;
                break;
            }
            case 'M':
                lkm_param->n_loc_models = atoi(argv[i]);
                break;
            case 'o':
                lkm_param->nr_folds_loc_models_selection = atoi(argv[i]);
                break;
            case 'l':
                lkm_param->eval_on_kp = (atoi(argv[i]) == 1);
                break;
            case 's':
                seed = (unsigned int) atoi(argv[i]);
                break;
            case 'S':
                lkm_param->silent = (atoi(argv[i]) == 1);
                if (lkm_param->silent)
                    svm_print_string = &print_null;
                break;
            default:
                fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
                exit_with_help();
        }
    }

    if (param->kernel_type == POLY && param->gamma < 0.0)
        param->gamma = 1.0;

    lkm_param->kp = MAX((int) (((double) lkm_param->k) * lkm_param->kp_ratio), 1);

    if (i >= argc)
        exit_with_help();

    strcpy(train_file_name, argv[i++]);

    // if (lkm_param->nr_fold < 2)
    strcpy(model_file_name, argv[i++]);
}
