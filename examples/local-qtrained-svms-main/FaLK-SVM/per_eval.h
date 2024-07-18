#ifndef _PAR_EVAL_H
#define _PAR_EVAL_H

class PerEval {

public:

    PerEval(double *true_labels, int n);

    ~PerEval();

    double calcAccuracy(double *pred_labels, bool print = true);

    double calcFMeasure(double *pred_labels, bool print = true);

    double calcPrecision(double *pred_labels, bool print = true);

    double calcRecall(double *pred_labels, bool print = true);

    double calcAll(double *pred_labels, int opt, bool print = true);

    double calcLift20P(double *prob, bool print = true);

    double calcMSE(double *pred_labels, bool print = true);


private:

    int n;
    double *true_labels;

    int distinct_labels_num;
    double* distinct_labels;

    double accuracy;
    double* precision;
    double* recall;
    double* FMeasure;
};


#endif
