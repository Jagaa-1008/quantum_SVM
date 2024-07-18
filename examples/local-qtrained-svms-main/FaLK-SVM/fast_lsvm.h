#ifndef _SFASTLSVM_H
#define _SFASTLSVM_H


class sFastLSVM {
public:

    sFastLSVM(struct lkm_parameter *par, struct problem *prob, char *model_file, bool cross_validation_instance = false,
              PyObject *python_svm_class = nullptr);

    sFastLSVM(int pred_strategy = 0, bool python_svm = false);

    ~sFastLSVM();

    void train();

    double *cross_validation();

    void save();

    void load(char *file);

    double *predict(struct problem *test);

    double predict(struct node *test, int features_number = -1);

    double *predict_p(struct problem *test);

    double *predict(struct problem *test, int k);

    double predict_p(struct node *test);

    int get_svm_type();

    CoverTree *get_cover_tree() { return coverTree; };

private:

    struct lkm_parameter *par;
    struct problem *prob;
    struct problem *prob_pl;
    struct problem *ct_prob;
    CoverTree *coverTree;
    ParEst *parEst;
    int pred_strategy;

    char *model_file = nullptr;

    bool cross_validation_instance = false;
    bool loaded_from_file = false;
    PyObject *tmp_python_svm_class = nullptr;

    //bool too_unbalanced(struct problem *p);
    //bool contains(int el, int *arr, int n);
    void cross_validation_int(const problem *prob, int nr_fold, double *target, char *model_file_arg);

    PyObject *initialize_python();
};

void destroy_lkm_param(struct lkm_parameter *par);

#endif
