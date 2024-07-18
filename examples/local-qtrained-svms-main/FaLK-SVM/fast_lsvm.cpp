#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <filesystem>
#include "defs.h"

#include "cover_tree.h"

#include "par_est.h"
#include "per_eval.h"
#include "fast_lsvm.h"
#include "svm.h"
#include "knn.h"

sFastLSVM::sFastLSVM(struct lkm_parameter *par, struct problem *prob, char *model_file, bool cross_validation_instance,
                     PyObject *python_svm_class) {

    this->par = par;
    this->model_file = model_file;
    this->cross_validation_instance = cross_validation_instance;

    coverTree = NULL;

    if (par->svm_par->python_svm) {
        if (!cross_validation_instance) {
            par->svm_par->python_svm_class = initialize_python();
        } else {
            par->svm_par->python_svm_class = python_svm_class;
        }

    }

    if (par->unbalanced == 1) {
        prob_pl = Malloc(struct problem, 1);

        prob_pl->l = 0;
        for (int i = 0; i < prob->l; i++)
            if (prob->y[i] == 1)
                prob_pl->l++;

        prob_pl->x = Malloc(struct node*, prob_pl->l);
        prob_pl->y = Malloc(double, prob_pl->l);

        int ppi = 0;
        for (int i = 0; i < prob->l; i++)
            if (prob->y[i] == 1) {
                prob_pl->x[ppi] = prob->x[i];
                prob_pl->y[ppi] = prob->y[i];
                ppi++;
            }
    }
    this->prob = prob;

    parEst = new ParEst(par->opt, par->nr_folds_loc_models_selection);
}

sFastLSVM::sFastLSVM(int pred_strategy, bool python_svm) {
    coverTree = NULL;
    if (python_svm) {
        tmp_python_svm_class = initialize_python();
    }
    parEst = new ParEst(0, 10);
    this->pred_strategy = pred_strategy;
}

sFastLSVM::~sFastLSVM() {
    delete parEst;
    if (coverTree != NULL)
        delete coverTree;
    if (par->svm_par->python_svm && !cross_validation_instance) {
        if (tmp_python_svm_class != nullptr) {
            Py_DECREF(tmp_python_svm_class);
        } else {
            Py_DECREF(par->svm_par->python_svm_class);
        }
        Py_Finalize();
    }
    if (loaded_from_file)
        svm_destroy_param(par->svm_par);
}

PyObject *sFastLSVM::initialize_python() {
    Py_Initialize();
    PyRun_SimpleString("import sys; sys.path.append(\"./Python_models\")");

    PyObject *py_module_name = PyUnicode_FromString("PythonSVM");
    PyObject *py_module = PyImport_Import(py_module_name);
    if (py_module == nullptr) {
        PyErr_Print();
        printf("Failed to import the PythonSVM module\n");
        exit(2);
    }

    char py_class_name[10] = "PythonSVM";
    PyObject *py_class = PyObject_GetAttrString(py_module, py_class_name);
    if (py_class == nullptr) {
        PyErr_Print();
        printf("Failed to import the PythonSVM class");
        exit(2);
    }

    Py_DECREF(py_module_name);
    Py_DECREF(py_module);

    return py_class;
}

void sFastLSVM::cross_validation_int(const problem *prob, int nr_fold, double *target, char *model_file_arg) {
    int i;
    int *fold_start = Malloc(int, nr_fold + 1);
    if (fold_start == NULL) MallocError("fold_start");
    int l = prob->l;
    int *perm = Malloc(int, l);
    if (perm == NULL) MallocError("perm");
    int nr_class;


    int *start = NULL;
    int *label = NULL;
    int *count = NULL;
    svm_group_classes(prob, &nr_class, &label, &start, &count, perm);

    // random shuffle and then data grouped by fold using the array perm
    int *fold_count = Malloc(int, nr_fold);
    if (fold_count == NULL) MallocError("fold_count");
    int c;
    int *index = Malloc(int, l);
    if (index == NULL) MallocError("index");
    for (i = 0; i < l; i++)
        index[i] = perm[i];
    for (c = 0; c < nr_class; c++)
        for (i = 0; i < count[c]; i++) {
            int j = i + rand() % (count[c] - i);
            swap(index[start[c] + j], index[start[c] + i]);
        }
    for (i = 0; i < nr_fold; i++) {
        fold_count[i] = 0;
        for (c = 0; c < nr_class; c++)
            fold_count[i] += (i + 1) * count[c] / nr_fold - i * count[c] / nr_fold;
    }
    fold_start[0] = 0;
    for (i = 1; i <= nr_fold; i++)
        fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
    for (c = 0; c < nr_class; c++)
        for (i = 0; i < nr_fold; i++) {
            int begin = start[c] + i * count[c] / nr_fold;
            int end = start[c] + (i + 1) * count[c] / nr_fold;
            for (int j = begin; j < end; j++) {
                perm[fold_start[i]] = index[j];
                fold_start[i]++;
            }
        }
    fold_start[0] = 0;
    for (i = 1; i <= nr_fold; i++)
        fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
    free(start);
    free(label);
    free(count);
    free(index);
    free(fold_count);

    int features_number = get_features_number(prob);
    double *means = Malloc(double, features_number);
    double *stds = Malloc(double, features_number);

    const char *model_file_extension_dot = strrchr(model_file_arg, '.');
    if (model_file_extension_dot == nullptr) {
        model_file_extension_dot = model_file_arg + strlen(model_file_arg);
    }
    char format_str[9] = "_fold_%d";
    char formatted_str[14];

    char *fold_i_train_file = Malloc(char, strlen(model_file_arg) + strlen(format_str) + 22);
    strncpy(fold_i_train_file, model_file_arg, model_file_extension_dot - model_file_arg);
    fold_i_train_file[model_file_extension_dot - model_file_arg] = '\0';

    char *fold_i_test_file = Malloc(char, strlen(model_file_arg) + strlen(format_str) + 18);
    strcpy(fold_i_test_file, fold_i_train_file);

    char *fold_i_model_file = Malloc(char, strlen(model_file_arg) + strlen(format_str) + 5);
    strcpy(fold_i_model_file, fold_i_train_file);

    char *fold_i_res_file = Malloc(char, strlen(model_file_arg) + strlen(format_str) + 13);
    strcpy(fold_i_res_file, fold_i_train_file);

    for (i = 0; i < nr_fold; i++) {
        if (!par->silent)
            printf("\n\n====== Fold %d ======================================================================\n", i);
        int begin = fold_start[i];
        int end = fold_start[i + 1];
        int j, k;
        struct problem training_subprob;
        struct problem test_subprob;

        training_subprob.l = l - (end - begin);
        training_subprob.x = Malloc(struct node*, training_subprob.l);
        if (training_subprob.x == NULL) MallocError("training_subprob.x");
        training_subprob.y = Malloc(double, training_subprob.l);
        if (training_subprob.y == NULL) MallocError("training_subprob.y");

        test_subprob.l = end - begin;
        test_subprob.x = Malloc(struct node*, test_subprob.l);
        if (test_subprob.x == NULL) MallocError("test_subprob.x");
        test_subprob.y = Malloc(double, test_subprob.l);
        if (test_subprob.y == NULL) MallocError("test_subprob.y");

        k = 0;
        for (j = 0; j < begin; j++) {
            training_subprob.x[k] = prob->x[perm[j]];
            training_subprob.y[k] = prob->y[perm[j]];
            struct node *n = training_subprob.x[k];
            while (n->index > 0) n++;
            n->index = -k;
            ++k;
        }
        for (j = end; j < l; j++) {
            training_subprob.x[k] = prob->x[perm[j]];
            training_subprob.y[k] = prob->y[perm[j]];
            struct node *n = training_subprob.x[k];
            while (n->index > 0) n++;
            n->index = -k;
            ++k;
        }

        k = 0;
        for (j = begin; j < end; j++) {
            test_subprob.x[k] = prob->x[perm[j]];
            test_subprob.y[k] = prob->y[perm[j]];
            struct node *n = test_subprob.x[k];
            while (n->index > 0) n++;
            n->index = -k;
            ++k;
        }

        compute_means_and_stds(&training_subprob, features_number, means, stds);

        for (k = 0; k < training_subprob.l; k++) {
            for (int f = 0; f < features_number; f++) {
                training_subprob.x[k][f].value = (training_subprob.x[k][f].value - means[f]) / stds[f];
            }
        }

        for (k = 0; k < test_subprob.l; k++) {
            for (int f = 0; f < features_number; f++) {
                test_subprob.x[k][f].value = (test_subprob.x[k][f].value - means[f]) / stds[f];
            }
        }

        fold_i_train_file[model_file_extension_dot - model_file_arg] = '\0';
        sprintf(formatted_str, format_str, i);
        strcat(fold_i_train_file, formatted_str);
        strcat(fold_i_train_file, "_training_set.txt");

        save_problem(fold_i_train_file, &training_subprob);

        fold_i_test_file[model_file_extension_dot - model_file_arg] = '\0';
        strcat(fold_i_test_file, formatted_str);
        strcat(fold_i_test_file, "_test_set.txt");

        save_problem(fold_i_test_file, &test_subprob);

        fold_i_model_file[model_file_extension_dot - model_file_arg] = '\0';
        strcat(fold_i_model_file, formatted_str);
        strcat(fold_i_model_file, model_file_extension_dot);

        sFastLSVM *flmf = new sFastLSVM(par, &training_subprob, fold_i_model_file, true, par->svm_par->python_svm_class);
        flmf->train();
        flmf->save();

        fold_i_res_file[model_file_extension_dot - model_file_arg] = '\0';
        strcat(fold_i_res_file, formatted_str);
        strcat(fold_i_res_file, "_res.csv");

        FILE *out = fopen(fold_i_res_file, "w");
        fprintf(out, "sample_index,predicted,correct\n");

        double *tmp_res = Malloc(double, end - begin);
        double *tmp_c = Malloc(double, end - begin);

        for (j = begin; j < end; j++) {
            if (par->svm_par->probability == 0)
                target[perm[j]] = flmf->predict(prob->x[perm[j]], features_number);
            else
                target[perm[j]] = flmf->predict_p(prob->x[perm[j]]);
            tmp_res[j - begin] = target[perm[j]];
            tmp_c[j - begin] = prob->y[perm[j]];
            fprintf(out, "%d,%.6f,%.6f\n", perm[j], target[perm[j]], prob->y[perm[j]]);
        }

        fclose(out);

        PerEval *perEval = new PerEval(tmp_c, end - begin);
        bool s = !par->silent;

        if (par->svm_par->svm_type == C_SVC) {
            if (par->per_meas < 4) perEval->calcAll(tmp_res, par->per_meas, s);
            if (par->per_meas == 4) perEval->calcLift20P(tmp_res, true);
        } else if (par->svm_par->svm_type == EPSILON_SVR) perEval->calcMSE(tmp_res, 1);

        delete perEval;

        free(tmp_res);
        free(tmp_c);
        delete flmf;

        free(training_subprob.x);
        free(training_subprob.y);
        free(test_subprob.x);
        free(test_subprob.y);
        if (!par->silent)
            printf("\n====== END Fold %d ==================================================================\n", i);
    }

    free(fold_start);
    free(perm);
    free(means);
    free(stds);
    free(fold_i_train_file);
    free(fold_i_test_file);
    free(fold_i_model_file);
    free(fold_i_res_file);
}

double *sFastLSVM::cross_validation() {
    double *target = Malloc(double, prob->l);
    cross_validation_int(prob, par->nr_fold, target, model_file);
    return target;
}


void sFastLSVM::train() {
    coverTree = new CoverTree(par->svm_par, par->base, false, 0, par->silent);

    int n_kps = par->n_k_lms;
    int *KPs = Malloc(int, n_kps);
    for (int i = 0; i < n_kps; i++)
        KPs[i] = (int) (((double) par->k_lms[i]) * par->kp_ratio);

    ct_prob = coverTree->addPoints(prob);

    const char *model_file_extension_dot = strrchr(model_file, '.');
    if (model_file_extension_dot == nullptr) {
        model_file_extension_dot = model_file + strlen(model_file);
    }
    auto model_file_extension_dot_index = (unsigned long) (model_file_extension_dot - model_file);

    if (par->local_model_selection != NO_LMS) {
        struct problem **nlp = NULL;

        int kkk = par->k_lms[par->n_k_lms - 1];
        int kkp = (int) (((double) par->k_lms[0]) * par->kp_ratio);
        nlp = coverTree->createNLocalProbs(par->n_loc_models, kkk, kkp, par->loc_prob_bil_th);

        char *local_model_selection_log_file = Malloc(char, strlen(model_file) + 9);
        strncpy(local_model_selection_log_file, model_file, model_file_extension_dot_index);
        local_model_selection_log_file[model_file_extension_dot_index] = '\0';
        strcat(local_model_selection_log_file, "_lms.txt");

        FILE *log_fp = nullptr;
        if (std::filesystem::exists(local_model_selection_log_file)) {
            log_fp = fopen(local_model_selection_log_file, "r+");
        } else {
            log_fp = fopen(local_model_selection_log_file, "w+");
        }

        if (par->svm_par->kernel_type == LINEAR) {
            double G_fake[1] = {1.0};
            if (par->svm_par->python_svm) {
                parEst->estCGEIABK(nlp, par->n_loc_models, par, par->c_lms, par->n_c_lms, G_fake, 1,
                                   par->e_lms, par->n_e_lms, par->i_lms, par->n_i_lms, par->a_lms, par->n_a_lms,
                                   par->b_lms, par->n_b_lms, par->k_lms, par->n_k_lms, KPs, n_kps, log_fp);
            } else {
                parEst->estCGK(nlp, par->n_loc_models, par, par->c_lms, par->n_c_lms, G_fake, 1, par->k_lms, par->n_k_lms,
                               KPs, n_kps, log_fp);
            }
        }
        if (par->svm_par->kernel_type == RBF) {
            if (par->svm_par->python_svm) {
                parEst->estCGEIABK(nlp, par->n_loc_models, par, par->c_lms, par->n_c_lms, par->g_lms, par->n_g_lms,
                                   par->e_lms, par->n_e_lms, par->i_lms, par->n_i_lms, par->a_lms, par->n_a_lms,
                                   par->b_lms, par->n_b_lms, par->k_lms, par->n_k_lms, KPs, n_kps, log_fp);
            } else {
                parEst->estCGK(nlp, par->n_loc_models, par, par->c_lms, par->n_c_lms, par->g_lms, par->n_g_lms,
                               par->k_lms,
                               par->n_k_lms, KPs, n_kps, log_fp);
            }
        }

        for (int i = 0; i < par->n_loc_models; i++)
            free(nlp[i]);
        free(nlp);
        free(local_model_selection_log_file);
        fclose(log_fp);
    }

    free(KPs);

    coverTree->setKs(par->k, par->kp);
    coverTree->createLocalProbs();

    int n = 0;

    bool est_gamma = false;
    double quant = -par->svm_par->gamma;
    if (par->svm_par->kernel_type == RBF && par->svm_par->gamma < 0.0) est_gamma = true;

    int h_warning = 0;
    int shrinked = 0;
    double h_warning_th = 0.5;

    struct problem **probs = coverTree->get_trained_probs();

    int features_number = -1;
    unsigned long python_svm_filepath_dir_index = 0;
    char *python_svm_filepath = nullptr;
    char python_svm_filename_format_str [14] = "py_svm_%d.pkl";
    char *python_svm_filename = nullptr;
    if (par->svm_par->python_svm && (coverTree->get_n_local_probs() > 0)) {
        features_number = get_features_number(probs[0]);

        char python_svm_dir_suffix [12] = "_py_models/";
        python_svm_filepath_dir_index = model_file_extension_dot_index + strlen(python_svm_dir_suffix);

        python_svm_filepath = Malloc(char, python_svm_filepath_dir_index + strlen(python_svm_filename_format_str) + 5);
        strncpy(python_svm_filepath, model_file, model_file_extension_dot_index);
        python_svm_filepath[model_file_extension_dot_index] = '\0';
        strcat(python_svm_filepath, python_svm_dir_suffix);

        python_svm_filename = Malloc(char, strlen(python_svm_filename_format_str) + 5);
    }

    printf("\n");
    for (int i = 0; i < coverTree->get_n_local_probs(); i++) {
        struct problem *lp = probs[i];

        if (par->unbalanced == 1) {
            lp = Malloc(struct problem, 1);
            lp->l = probs[i]->l;
            lp->x = Malloc(struct node*, lp->l);
            lp->y = Malloc(double, lp->l);

            for (int h = 0; h < lp->l; h++) {
                lp->x[h] = probs[i]->x[h];
                lp->y[h] = probs[i]->y[h];
            }

            for (int h = 0; h < prob_pl->l; h++) {
                struct node *nod = prob_pl->x[h];

                bool found = false;
                for (int s = 0; s < lp->l; s++)
                    if (lp->y[s] == 1 && lp->x[s] == nod)
                        found = true;

                if (!found) {
                    lp->l++;
                    lp->x = Realloc(lp->x, struct node*, lp->l);
                    lp->y = Realloc(lp->y, double, lp->l);
                    lp->x[lp->l - 1] = nod;
                    lp->y[lp->l - 1] = prob_pl->y[h];
                }
            }
        }

        struct svm_parameter *parameter = copy_svm_parameter(par->svm_par);

        if (est_gamma)
            parameter->gamma = parEst->estG_quant(lp, quant);
        if (par->local_model_selection == LMS_ALL) {
            parEst->estCG(lp, parameter, par->c_lms, par->n_c_lms, par->g_lms, par->n_g_lms, par->kp / 2, par->k);
            parameter->gamma = parEst->estG_quant(lp, -parameter->gamma);
        }

        int www = 0;
        if (parameter->weight != NULL && parameter->weight[0] < 0.0 && parameter->weight_label[0] == 1.0) {
            www = 1;
            int np = 0;
            for (int u = 0; u < lp->l; u++)
                if (lp->y[u] == 1.0)
                    np++;
            double cp = 1.0;
            if (np > 1) cp = -((double) lp->l) / ((double) np) * parameter->weight[0];
            if (cp > 256.0) cp = 256.0;
            parameter->weight[0] = cp;
        }

        if (i > 15 && (double) h_warning / (double) shrinked > h_warning_th) {
            parameter->shrinking = 0;
            if (!par->silent) {
                if (www)
                    printf("train the %d local SVM model [c=%5.2f, cw1=%.2f, cw-1=%.2f, g:%5.2f --> %5.2f, tot=%d, size=%d - shrinking deactivated]: \n",
                           (n++) + 1, par->svm_par->C, parameter->weight[0], parameter->weight[1], par->svm_par->gamma,
                           parameter->gamma, coverTree->get_n_local_probs(), lp->l);
                else {
                    if (par->svm_par->python_svm) {
                        printf("train the %d local SVM model [c=%5.2f, g:%5.2f --> %5.2f, b=%d, a=%d, i=%.2f, e=%.2f, n=%s, tot=%d, size=%d - shrinking deactivated] \n",
                               (n++) + 1, par->svm_par->C, par->svm_par->gamma, parameter->gamma, par->svm_par->python_svm_base,
                               par->svm_par->python_svm_binary_vars_num, par->svm_par->python_svm_penalty_coef,
                               par->svm_par->python_svm_multicl_reg, par->svm_par->python_svm_embeddings_dir,
                               coverTree->get_n_local_probs(), lp->l);
                    } else {
                        printf("train the %d local SVM model [c=%5.2f, g:%5.2f --> %5.2f, tot=%d, size=%d - shrinking deactivated]: \n",
                               (n++) + 1, par->svm_par->C, par->svm_par->gamma, parameter->gamma,
                               coverTree->get_n_local_probs(), lp->l);
                    }
                }
            }
        } else {
            if (!par->silent) {
                if (www)
                    printf("train the %d local SVM model [c=%5.2f, cw1=%.2f, cw-1=%.2f, g:%5.2f --> %5.2f, tot=%d, size=%d]: \n",
                           (n++) + 1, par->svm_par->C, parameter->weight[0], parameter->weight[1], par->svm_par->gamma,
                           parameter->gamma, coverTree->get_n_local_probs(), lp->l);
                else {
                    if (par->svm_par->python_svm) {
                        printf("train the %d local SVM model [c=%5.2f, g:%5.2f --> %5.2f, b=%d, a=%d, i=%.2f, e=%.2f, n=%s, tot=%d, size=%d] \n",
                               (n++) + 1, par->svm_par->C, par->svm_par->gamma, parameter->gamma, par->svm_par->python_svm_base,
                               par->svm_par->python_svm_binary_vars_num, par->svm_par->python_svm_penalty_coef,
                               par->svm_par->python_svm_multicl_reg, par->svm_par->python_svm_embeddings_dir,
                               coverTree->get_n_local_probs(), lp->l);
                    } else {
                        printf("train the %d local SVM model [c=%5.2f, g:%5.2f --> %5.2f, tot=%d, size=%d]: \n", (n++) + 1,
                               par->svm_par->C, par->svm_par->gamma, parameter->gamma, coverTree->get_n_local_probs(),
                               lp->l);
                    }
                }
            }
        }

        struct svm_model *m;
        if (par->svm_par->python_svm) {
            python_svm_filepath[python_svm_filepath_dir_index] = '\0';

            sprintf(python_svm_filename, python_svm_filename_format_str, i);
            strcat(python_svm_filepath, python_svm_filename);

            m = python_svm_train(lp, parameter, features_number, python_svm_filepath);
        } else {
            m = svm_train(lp, parameter);
        }

        if (par->unbalanced == 1) {
            free(lp->x);
            free(lp->y);
            free(lp);
        }


        if (!par->silent) printf("\n");

        if (parameter->shrinking) {
            shrinked++;
            if (parameter->h_warning) h_warning++;
        }
        free(parameter);

        coverTree->set_model(m, i);
    }

    if (par->svm_par->python_svm) {
        free(python_svm_filepath);
        free(python_svm_filename);
    }
}

void sFastLSVM::save() {
    coverTree->save(model_file);
}

void sFastLSVM::load(char *file) {
    coverTree = new CoverTree(NULL, -1, false, pred_strategy, 1);
    coverTree->load(file, tmp_python_svm_class);
    tmp_python_svm_class = nullptr;
    par = Malloc(struct lkm_parameter, 1);
    par->k = coverTree->getK();
    par->kp = coverTree->getKP();
    par->local_model_selection = 0;
    par->svm_par = coverTree->getParam();
    loaded_from_file = true;
}

int sFastLSVM::get_svm_type() {
    return par->svm_par->svm_type;
}

double *sFastLSVM::predict(struct problem *test) {
    double *res = Malloc(double, test->l);
    if (res == NULL) MallocError("res");

    int features_number = -1;
    if (par->svm_par->python_svm)
        features_number = get_features_number(test);

    for (int i = 0; i < test->l; i++) {
        struct svm_model *m = coverTree->nn_model(test->x[i]);
        if (par->svm_par->python_svm) {
            res[i] = python_svm_predict(m, test->x[i], features_number);
        } else {
            res[i] = svm_predict(m, test->x[i]);
        }
    }
    return res;
}

double *sFastLSVM::predict_p(struct problem *test) {
    double *res = Malloc(double, test->l);
    if (res == NULL) MallocError("res");
    for (int i = 0; i < test->l; i++)
        res[i] = predict_p(test->x[i]);
    return res;
}


double sFastLSVM::predict_p(struct node *test) {
    struct svm_model *m = coverTree->nn_model(test);

    int nr_class = svm_get_nr_class(m);

    int *labels = (int *) malloc(nr_class * sizeof(int));
    svm_get_labels(m, labels);
    double *prob_estimates = (double *) malloc(nr_class * sizeof(double));

    int ind_an = -1;
    for (int j = 0; j < nr_class; j++)
        if (labels[j] == 1)
            ind_an = j;//printf(" %d",labels[j]);

    free(labels);

    svm_predict_probability(m, test, prob_estimates);
    double res = prob_estimates[ind_an];

    free(prob_estimates);

    return res;
}

double sFastLSVM::predict(struct node *test, int features_number) {
    struct svm_model *m = coverTree->nn_model(test);
    if (par->svm_par->python_svm) {
        return python_svm_predict(m, test, features_number);
    } else {
        return svm_predict(m, test);
    }
}


double *sFastLSVM::predict(struct problem *test, int k) {
    double *res = Malloc(double, test->l);
    if (res == NULL) MallocError("res");

    int pc1 = test->l / 1000;
    int nn = 0;

    int features_number = -1;
    if (par->svm_par->python_svm)
        features_number = get_features_number(test);

    double *loc = Malloc(double, k);

    KNN *knn = new KNN(NULL, NULL);

    for (int i = 0; i < test->l; i++) {
        struct svm_model **ms = coverTree->knn_models(test->x[i], k);

        for (int j = 0; j < k; j++) {
            if (par->svm_par->python_svm) {
                loc[j] = python_svm_predict(ms[j], test->x[i], features_number);
            } else {
                loc[j] = svm_predict(ms[j], test->x[i]);
            }
        }

        res[i] = knn->majorityRuleBin(loc, k);

        if (i == pc1 * nn) {
            printf("%d point tested [%.1f%%]\n", i, ((double) i) / (double(test->l)) * 100.0);
            nn++;
        }

        free(ms);
    }

    // free(knn);
    delete knn;
    return res;
}

void destroy_lkm_param(struct lkm_parameter *par){
    if (par->c_lms != nullptr) {
        free(par->c_lms);
    }
    if (par->g_lms != nullptr) {
        free(par->g_lms);
    }
    if (par->k_lms != nullptr) {
        free(par->k_lms);
    }
    if (par->b_lms != nullptr) {
        free(par->b_lms);
    }
    if (par->a_lms != nullptr) {
        free(par->a_lms);
    }
    if (par->i_lms != nullptr) {
        free(par->i_lms);
    }
    if (par->e_lms != nullptr) {
        free(par->e_lms);
    }
}
