#ifndef _KNN_H
#define _KNN_H


class KNN {

public:

    KNN(struct lkm_parameter *par, struct problem *prob);

    ~KNN();

    void train();

    double *predict(struct problem *test);

    double *predict(struct problem *test, int k);

    double predict_one(struct node *test);

    double predict_one(struct node *test, int k);

    double majorityRuleBin(double *ls, int k);

    double averageRuleBin(double *ls, int k);

    double count(double *ls, int k, double lab);

    void cross_validation(int nr_fold, double *target);

private:

    struct lkm_parameter *par;
    struct problem *prob;
    double *lab_a;
    int *lab_c;
    CoverTree *coverTree;
};


#endif
