#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include "per_eval.h"
#include "defs.h"


PerEval::PerEval(double *true_labels, int n) {
    this->n = n;
    this->true_labels = true_labels;

    distinct_labels = Malloc(double, n);
    distinct_labels_num = 0;
    for (int i = 0; i < n; i++) {
        bool unseen_label = true;
        for (int j = 0; j < distinct_labels_num && unseen_label; j++) {
            if (true_labels[i] == distinct_labels[j]) {
                unseen_label = false;
            }
        }
        if (unseen_label){
            distinct_labels[distinct_labels_num] = true_labels[i];
            distinct_labels_num++;
        }
    }
    distinct_labels = Realloc(distinct_labels, double, distinct_labels_num);
    if (distinct_labels_num != 0 && distinct_labels == NULL) MallocError("distinct_labels in PerEval");

    accuracy = -1.0;

    precision = Malloc(double, distinct_labels_num);
    recall = Malloc(double, distinct_labels_num);
    FMeasure = Malloc(double, distinct_labels_num);

    for (int j = 0; j < distinct_labels_num; j++) {
        precision[j] = -1.0;
        recall[j] = -1.0;
        FMeasure[j] = -1.0;
    }
}

PerEval::~PerEval() {
    free(distinct_labels);
    free(precision);
    free(recall);
    free(FMeasure);
}

double PerEval::calcAccuracy(double *pred_labels, bool print) {
    double right = 0.0;

    for (int i = 0; i < n; i++)
        if (pred_labels[i] == true_labels[i])
            right++;

    accuracy = right / ((double) n);

    if (print) {
        printf("\n########################### RESULT ##########################\n");
        printf(" Accuracy = %f [%d/%d]\n", accuracy, (int) right, n);
        printf("#############################################################\n");
    }
    return accuracy;
}


double PerEval::calcMSE(double *pred_labels, bool print) {
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

    for (int i = 0; i < n; i++)
        //if( pred_labels[i] == true_labels[i] )
    {
        //printf("%f %f\n",pred_labels[i],true_labels[i]);
        error += (pred_labels[i] - true_labels[i]) * (pred_labels[i] - true_labels[i]);
        sump += pred_labels[i];
        sumt += true_labels[i];
        sumpp += pred_labels[i] * pred_labels[i];
        sumtt += true_labels[i] * true_labels[i];
        sumpt += pred_labels[i] * true_labels[i];
        ++total;
    }

    if (print) {
        printf("\n########################### RESULT ##########################\n");
        printf("Mean squared error = %g (regression)\n", error / total);
        printf("Squared correlation coefficient = %g (regression)\n",
               ((total * sumpt - sump * sumt) * (total * sumpt - sump * sumt)) /
               ((total * sumpp - sump * sump) * (total * sumtt - sumt * sumt))
        );
        printf("#############################################################\n");
    }
    return accuracy;
}


double PerEval::calcFMeasure(double *pred_labels, bool print) {
    return calcAll(pred_labels, FMEASURE, print);
}


double PerEval::calcPrecision(double *pred_labels, bool print) {
    return calcAll(pred_labels, PRECISION, print);
}


double PerEval::calcRecall(double *pred_labels, bool print) {
    return calcAll(pred_labels, RECALL, print);
}

inline int comp_lift(const void *e1, const void *e2) {
    /* register */ double **pa1 = (double **) e1;
    /* register */ double **pa2 = (double **) e2;
    if (*pa1[0] > *pa2[0]) return -1;
    if (*pa1[0] < *pa2[0]) return 1;
    return 0;
}


double PerEval::calcLift20P(double *prob, bool print) {
    double **a = Malloc(double*, n);
    for (int i = 0; i < n; i++) {
        a[i] = Malloc(double, 2);
        a[i][1] = true_labels[i];
        a[i][0] = prob[i];
    }

    qsort(a, n, sizeof(double *), comp_lift);

    int n20p = n / 5;

    double lift20p = 0.0;
    double na = 0.0;

    for (int i = 0; i < n; i++) {
        if (i < n20p && a[i][1] == 1.0)
            lift20p++;
        if (a[i][1] == 1.0)
            na++;
    }

    lift20p = lift20p / na * 5.0;

    if (print) {
        printf("\n########################### RESULT ##########################\n");
        printf(" Lift 20 perc = %f \n", lift20p);
        printf("#############################################################\n");
    }


    //for( int i = 0; i < n; i++ )
    //	{
    //	printf("%f -- %f\n",a[i][0],a[i][1]);
    //	free(a[i]);
    //	}

    free(a);
    return lift20p;
}


double PerEval::calcAll(double *pred_labels, int opt, bool print) {
    double *tp = Malloc(double, distinct_labels_num);
    double *fn = Malloc(double, distinct_labels_num);
    double *fp = Malloc(double, distinct_labels_num);
    double *tn = Malloc(double, distinct_labels_num);
    for (int j = 0; j < distinct_labels_num; j++) {
        tp[j] = 0.0;
        fn[j] = 0.0;
        fp[j] = 0.0;
        tn[j] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < distinct_labels_num; j++) {
            if (true_labels[i] == distinct_labels[j]) {
                if (pred_labels[i] == true_labels[i]) {
                    tp[j]++;
                } else {
                    fn[j]++;
                }
            } else {
                if (pred_labels[i] == distinct_labels[j]) {
                    fp[j]++;
                } else {
                    tn[j]++;
                }
            }
        }
    }

    accuracy = 0.0;
    double avg_precision = 0.0, avg_recall = 0.0, avg_FMeasure = 0.0;
    for (int j = 0; j < distinct_labels_num; j++) {
        accuracy += tp[j];

        precision[j] = tp[j] / (tp[j] + fp[j]);
        avg_precision += precision[j];

        recall[j] = tp[j] / (tp[j] + fn[j]);
        avg_recall += recall[j];

        FMeasure[j] = 2.0 * precision[j] * recall[j] / (precision[j] + recall[j]);
        avg_FMeasure += FMeasure[j];
    }
    accuracy /= ((double) n);
    avg_precision /= ((double) distinct_labels_num);
    avg_recall /= ((double) distinct_labels_num);
    avg_FMeasure /= ((double) distinct_labels_num);

    if (print) {
        printf("\n########################### RESULT ##########################\n");
        printf(" Samples number = %d \n", n);
        printf(" Classes        = [ ");
        for (int j = 0 ; j < distinct_labels_num; j++) {
            if (j != 0)
                printf(", ");
            printf("%5.0f", distinct_labels[j]);
        }
        printf(" ]\n");
        printf(" Accuracy       = %f \n", accuracy);
        printf(" Precision      = [ ");
        for (int j = 0 ; j < distinct_labels_num; j++) {
            if (j != 0)
                printf(", ");
            printf("%5.3f", precision[j]);
        }
        printf(" ]\n");
        printf(" Avg. Precision = %f \n", avg_precision);
        printf(" Recall         = [ ");
        for (int j = 0 ; j < distinct_labels_num; j++) {
            if (j != 0)
                printf(", ");
            printf("%5.3f", recall[j]);
        }
        printf(" ]\n");
        printf(" Avg. Recall    = %f \n", avg_recall);
        printf(" FMeasure       = [ ");
        for (int j = 0 ; j < distinct_labels_num; j++) {
            if (j != 0)
                printf(", ");
            printf("%5.3f", FMeasure[j]);
        }
        printf(" ]\n");
        printf(" Avg. FMeasure  = %f \n", avg_FMeasure);
        printf("#############################################################\n");
    }

    free(tp);
    free(fn);
    free(fp);
    free(tn);

    if (opt == ACCURACY)
        return accuracy;
    if (opt == PRECISION)
        return avg_precision;
    if (opt == RECALL)
        return avg_recall;
    if (opt == FMEASURE) {
        if (!(avg_FMeasure >= 0.0 || avg_FMeasure <= 1.0))
            return 0.0;
        return avg_FMeasure;
    }
    return -1.0;
}
