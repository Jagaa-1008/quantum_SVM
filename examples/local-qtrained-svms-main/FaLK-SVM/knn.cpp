#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "defs.h"
#include "cover_tree.h"
#include "knn.h"
#include "per_eval.h"
#include "svm.h"

#define MAX_CLASSES 1000

KNN::KNN(struct lkm_parameter *par, struct problem *prob) {
    this->par = par;
    this->prob = prob;
    this->lab_a = Malloc(double, MAX_CLASSES);
    this->lab_c = Malloc(int, MAX_CLASSES);
    coverTree = NULL;
}

KNN::~KNN() {
    if (coverTree != NULL)
        delete coverTree;
    free(lab_a);
    free(lab_c);
}

void KNN::train() {
    coverTree = new CoverTree(par->svm_par, par->base, false, 0, par->silent);
    coverTree->setKs(par->k, par->kp);
    coverTree->addPoints(prob);
}

double *KNN::predict(struct problem *test) {
    double *pred = Malloc(double, test->l);

    if (par->type == CLASSIFICATION) {
        for (int i = 0; i < test->l; i++) {
            if (i % 1000 == 0)
                printf("Tested %d\n", i);
            pred[i] = coverTree->nn_label(test->x[i]);
            //struct node *nn = coverTree->nn(test->x[i]);
            //while(nn->index > 0)nn++;
            //printf("%d\n",-nn->index);
        }
    } else if (par->type == REGRESSION) {
        for (int i = 0; i < test->l; i++) {
            if (i % 1000 == 0)
                printf("Reg Tested %d\n", i);
            pred[i] = coverTree->nn_label(test->x[i]);
        }
    } else {
        printf("ERROR: CLASSIFICATION OR REGERSSION??\n");
        exit(0);
    }

    return pred;
}

double *KNN::predict(struct problem *test, int k) {
    double *pred = Malloc(double, test->l);

    if (par->type == CLASSIFICATION) {
        if (par->knn_strategy == 0)
            for (int i = 0; i < test->l; i++) {
                if (i % 1000 == 0)
                    printf("Tested %d\n", i);
                double *pr = coverTree->knn_label(test->x[i], k);
                pred[i] = majorityRuleBin(pr, k);
                free(pr);
            }
        else if (par->knn_strategy == 1)
            for (int i = 0; i < test->l; i++) {
                if (i % 1000 == 0)
                    printf("Tested %d\n", i);
                double *pr = coverTree->knn_label(test->x[i], k);
                pred[i] = count(pr, k, 1.0);
                free(pr);
            }
    } else if (par->type == REGRESSION) {
        for (int i = 0; i < test->l; i++) {
            if (i % 1000 == 0)
                printf("Reg Tested %d\n", i);
            double *pr = coverTree->knn_label(test->x[i], k);
            pred[i] = averageRuleBin(pr, k);
            free(pr);
        }
    } else {
        printf("ERROR: CLASSIFICATION OR REGERSSION??\n");
        exit(0);
    }

    return pred;
}

double KNN::predict_one(struct node *test) {
    if (par->type == CLASSIFICATION) {
        if (par->knn_strategy == 0)
            return coverTree->nn_label(test);
        else if (par->knn_strategy == 1)
            return coverTree->nn_label(test) == 1.0;
    } else if (par->type == REGRESSION) {
        return coverTree->nn_label(test);
    } else {
        printf("ERROR: CLASSIFICATION OR REGERSSION??\n");
        exit(0);
    }

    return -1.0;
}

double KNN::predict_one(struct node *test, int k) {
    double *pr = coverTree->knn_label(test, k);

    if (par->type == CLASSIFICATION) {
        double prod = -1.0;
        if (par->knn_strategy == 0)
            prod = majorityRuleBin(pr, k);
        else if (par->knn_strategy == 1)
            prod = count(pr, k, 1.0);
        free(pr);
        return prod;
    } else if (par->type == REGRESSION) {
        double prod = -1.0;
        prod = averageRuleBin(pr, k);
        free(pr);
        return prod;
    } else {
        printf("ERROR: CLASSIFICATION OR REGERSSION??\n");
        exit(0);
    }
}


double KNN::majorityRuleBin(double *ls, int k) {
    lab_a[0] = ls[0];
    lab_c[0] = 1;
    int n_l = 1;

    for (int i = 1; i < k; i++) {
        int j = 0;
        for (j = 0; j < i; j++)
            if (ls[i] == lab_a[j]) {
                lab_c[j]++;
                break;
            }
        if (j == i) {
            lab_a[n_l] = ls[i];
            lab_c[n_l] = 1;
            n_l++;
        }
    }

    int max = 0;
    double max_l = -1.0;
    for (int i = 0; i < n_l; i++) {
        if (lab_c[i] > max) {
            max = lab_c[i];
            max_l = lab_a[i];
        }
    }

    return max_l;

    /*
    double l0 = ls[0];
    double lm1 = -1.0;
    int count_l0 = 0;

    for( int i = 0; i < k; i++ )
        if( ls[i] == l0 )
            {
            count_l0++;
            if( count_l0 >= (k+1) / 2 ) return l0;
            }
        else lm1 = ls[i];
    return lm1;
    */
}


double KNN::averageRuleBin(double *ls, int k) {
    double sum = 0.0;

    for (int i = 0; i < k; i++)
        sum += ls[i];

    return sum / (double) k;
}


double KNN::count(double *ls, int k, double lab) {
    double count_l = 0.0;

    for (int i = 0; i < k; i++)
        if (ls[i] == lab)
            count_l++;

    return count_l;
}


void KNN::cross_validation(int nr_fold, double *target) {
    int i;
    int *fold_start = Malloc(int, nr_fold + 1);
    if (fold_start == NULL) MallocError("fold_start");
    int l = prob->l;
    int *perm = Malloc(int, l);
    if (perm == NULL) MallocError("perm");
    int nr_class;

    // stratified cv may not give leave-one-out rate
    // Each class to l folds -> some folds may have zero elements
    {
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
    }


    for (i = 0; i < nr_fold; i++) {
        printf("\n\n====== Fold %d ======================================================================\n", i);
        int begin = fold_start[i];
        int end = fold_start[i + 1];
        int j, k;
        struct problem subprob;

        subprob.l = l - (end - begin);
        subprob.x = Malloc(struct node*, subprob.l);
        if (subprob.x == NULL) MallocError("subprob.x");
        subprob.y = Malloc(double, subprob.l);
        if (subprob.y == NULL) MallocError("subprob.y");

        k = 0;
        for (j = 0; j < begin; j++) {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }
        for (j = end; j < l; j++) {
            subprob.x[k] = prob->x[perm[j]];
            subprob.y[k] = prob->y[perm[j]];
            ++k;
        }

        KNN *sub_knn = new KNN(this->par, &subprob);
        sub_knn->train();

        double *tmp_res = Malloc(double, end - begin);
        double *tmp_c = Malloc(double, end - begin);

        for (j = begin; j < end; j++) {
            if (par->k == 1)
                target[perm[j]] = sub_knn->predict_one(prob->x[perm[j]]);
            else
                target[perm[j]] = sub_knn->predict_one(prob->x[perm[j]], par->k);
            tmp_res[j - begin] = target[perm[j]];
            tmp_c[j - begin] = prob->y[perm[j]];
        }

        PerEval *perEval = new PerEval(tmp_c, end - begin);
        if (par->type == CLASSIFICATION) {
            if (par->per_meas < 4)
                perEval->calcAll(tmp_res, par->per_meas);
            if (par->per_meas == 4)
                perEval->calcLift20P(tmp_res, true);
        } else if (par->type == REGRESSION) {
            perEval->calcMSE(tmp_res, true);
        }
        delete perEval;

        free(tmp_res);
        free(tmp_c);
        delete sub_knn;

        free(subprob.x);
        free(subprob.y);

        printf("====== END Fold %d ==================================================================\n", i);
    }
    free(fold_start);
    free(perm);
}
