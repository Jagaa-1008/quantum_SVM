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
#include "FaLK-SVM/fast_lsvm.h"
#include "FaLK-SVM/per_eval.h"

void exit_with_help() {
    printf(
            "Usage: FaLK-SVM-predict [options] test_file model_file output_label_file\n"
            "options:\n"
            "-T local model retrieval strategy (default 0)\n"
            "	0 -- nn with all points\n"
            "	1 -- nn with centers only\n"
            "	2 -- knn with centers only\n"
            "-K k for knn with centers only (default 3)\n"
            "-p probability_estimates: whether to predict probability estimates, 0 or 1 (default 0) [not supported by the Python SVMs]\n"
            "-q python_svm: whether the model consists of Python (1) SVMs instead of the default C++ ones (0)\n"
            "-M performance measure \n"
            "-S silent mode (default 0)\n"
    );
    exit(1);
}

void parse_command_line(int argc, char **argv, char *test_file_name, char *model_file_name, char *output_file_name);

//void read_problem(const char *filename, struct problem *prob, struct svm_parameter *param);
void do_cross_validation();

int pred_strategy;
int k;
int probability;
bool python_svm;
int per_meas;
int silent;

int main(int argc, char **argv) {
    char test_file_name[1024];
    char model_file_name[1024];
    char output_file_name[1024];

    struct problem *test = Malloc(struct problem, 1);
    if (test == NULL) MallocError("test");

    parse_command_line(argc, argv, test_file_name, model_file_name, output_file_name);

    read_problem(test_file_name, test);

    sFastLSVM *flm = new sFastLSVM(pred_strategy, python_svm);
    flm->load(model_file_name);
    if (!python_svm) fprintf(stdout, "\n");

    //if( probability ) printf("probability not implemented yet\n");
    double *res;
    if (pred_strategy != PRED_K_CENTERS) {
        if (probability == 1)
            res = flm->predict_p(test);
        else
            res = flm->predict(test);
    } else {
        //if( probability == 1 ) res = flm->predict_p(test);
        res = flm->predict(test, k);
    }

    FILE *out = fopen(output_file_name, "w");

    for (int i = 0; i < test->l; i++)
        fprintf(out, "%.6f\n", res[i]);

    fclose(out);

    PerEval *perEval = new PerEval(test->y, test->l);
    if (flm->get_svm_type() == C_SVC) {
        if (per_meas < 4)
            perEval->calcAll(res, per_meas);
        if (per_meas == 4)
            perEval->calcLift20P(res, true);
    } else if (flm->get_svm_type() == EPSILON_SVR)
        perEval->calcMSE(res, 1);
    delete perEval;

    delete flm;
    free(res);

    free(test->y);
    free(test->x);
    free(test);

    return 0;
}


void parse_command_line(int argc, char **argv, char *test_file_name, char *model_file_name, char *output_file_name) {
    pred_strategy = PRED_ALL;
    k = 3;
    probability = 0;
    python_svm = false;
    per_meas = 0;
    silent = 0;
    int i;
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-')
            break;

        if (++i >= argc)
            exit_with_help();
        switch (argv[i - 1][1]) {
            case 'T':
                pred_strategy = atoi(argv[i]);
                break;
            case 'K':
                k = atoi(argv[i]);
                break;
            case 'p':
                probability = atoi(argv[i]);
                break;
            case 'q':
                python_svm = (atoi(argv[i]) == 1);
                break;
            case 'M':
                per_meas = atoi(argv[i]);
                break;
            case 'S':
                silent = (atoi(argv[i]) == 1);
                if (silent)
                    svm_print_string = &print_null;
                break;
            default:
                fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
                exit_with_help();
        }
    }

    if (i >= argc)
        exit_with_help();

    strcpy(test_file_name, argv[i++]);
    strcpy(model_file_name, argv[i++]);
    strcpy(output_file_name, argv[i++]);
}

// read in a problem (in svmlight format)

