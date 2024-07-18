#ifndef _DEFS_H
#define _DEFS_H

#define VERSION 102


// added model selection to FkNNSVM


#define MallocError(p) {fprintf(stderr,"MallocError: %s\n",p); exit(0);}
#define ReallocError(p) {fprintf(stderr,"ReallocError: %s\n",p); exit(0);}
#define LoadError(f, l) {fprintf(stderr,"%s file: Wrong input format at line: %s\n", f, l); exit(0);}
#define LoadErrorP(f, l, l2) {fprintf(stderr,"%s file: %s (line: %s\n)", f, l2, l); exit(0);}

template<class T>
inline void swap(T &x, T &y) {
    T t = x;
    x = y;
    y = t;
}


#define Malloc(type, n) (type*)malloc((n)*sizeof(type))
#define Realloc(ptr, type, n) (type *)realloc(ptr,(n)*sizeof(type))

#define bot(d) ((d) < (0.0) ? (0.0) : (d))
#define ELEM_SWAP(a, b) { /* register */ double t=(a);(a)=(b);(b)=t; }
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


struct lkm_parameter {
    int nr_fold;
    int k;
    int kp;

    struct svm_parameter *svm_par;

    char out_put_file[1024];

    int base;

    int type;

    int nr_strategy;
    bool conservative_mr;

    double *c_lms;
    int n_c_lms;

    double *g_lms;
    int n_g_lms;

    int *k_lms;
    int n_k_lms;

    int *b_lms;
    int n_b_lms;

    int *a_lms;
    int n_a_lms;

    int *i_lms;
    int n_i_lms;

    int *e_lms;
    int n_e_lms;

    int pred_strategy;
    bool silent;

    int knn_strategy;

    int unbalanced;

    double kp_ratio;

    int local_model_selection;
    int n_loc_models;
    double loc_prob_bil_th;

    int opt;
    int per_meas;

    int nr_folds_loc_models_selection;
    bool eval_on_kp;
};

struct node {
    int index;
    double value;
};

#define FREE_INCR 1000
struct problem {
    int l;
    double *y;
    struct node **x;

    int *k_rank;
    int *lpm_ind;
};

enum {
    C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
};    /* svm_type */
enum {
    LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
}; /* kernel_type */

enum {
    LSVM_NR = 0, FALKNR = 1, ENN = 2, ALLKNN = 3, RR1 = 11, RR2 = 12, RR3 = 13, RR11 = 21, RR12 = 22, RR13 = 23
};
enum {
    CLASSIFICATION, REGRESSION
};

enum {
    ACCURACY = 0, PRECISION = 1, RECALL = 2, FMEASURE = 3, LIFT_20P = 4
};

enum {
    PRED_ALL, PRED_CENTERS, PRED_K_CENTERS
};

enum {
    NO_LMS = 0, LMS_INIT = 1, LMS_ALL = 2
};

enum {
    MAJ_RULE = 0, COUNT_P = 1
};


struct svm_parameter {
    bool python_svm;

    int svm_type;
    int kernel_type;
    int degree;    /* for poly */
    double gamma;    /* for poly/rbf/sigmoid */
    double coef0;    /* for poly/sigmoid */

    /* these are for C++ SVMs training */
    double cache_size; /* in MB */
    double eps;    /* stopping criteria */
    double C;    /* for C_SVC, EPSILON_SVR and NU_SVR */
    int nr_weight;        /* for C_SVC */
    int *weight_label;    /* for C_SVC */
    double *weight;        /* for C_SVC */
    double nu;    /* for NU_SVC, ONE_CLASS, and NU_SVR */
    double p;    /* for EPSILON_SVR */
    int shrinking;    /* use the shrinking heuristics */
    int probability; /* do probability estimates */

    bool h_warning;

    PyObject *python_svm_class;
    int python_svm_type;
    int python_svm_base;
    int python_svm_binary_vars_num;
    double python_svm_penalty_coef;
    double python_svm_multicl_reg;
    char* python_svm_embeddings_dir;
};

struct svm_model {
    svm_parameter param;    // parameters

    PyObject *python_svm_instance;

    int nr_class;            // number of classes, = 2 in regression/one class svm
    int l;                    // total #SV
    node **SV;                // SVs (SV[l])
    double **sv_coef;        // coefficients for SVs in decision functions (sv_coef[k-1][l])
    double *rho;            // constants in decision functions (rho[k*(k-1)/2])
    double *probA;          // pairwise probability information
    double *probB;

    // for classification only
    int *label;        // label of each class (label[k])
    int *nSV;        // number of SVs for each class (nSV[k])
    // nSV[0] + nSV[1] + ... + nSV[k-1] = l
    // XXX
    int free_sv;        // 1 if svm_model is created by svm_load_model
    // 0 if svm_model is created by svm_train
};

#endif /* _DEFS_H */
