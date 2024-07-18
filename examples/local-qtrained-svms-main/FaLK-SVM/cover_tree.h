#ifndef _COVER_TREE_H
#define _COVER_TREE_H

#define CT_VERSION 0.5

struct ct {
    struct ct_node_set *root;
    struct ct_node_set *tmp_root;
    struct problem *prob;
    struct svm_parameter *param;

    int rt_min_lev;
    int rt_max_lev;
};

#define LCACHE 1
struct ct_node {
    struct node **x;
    int *ind;
    struct ct_node **children;
    int n_children;
    int card;

    double *tmp_dist_c;
    struct node **tmp_dist_node_c;
    int last;

    double dist_from_f;
};

struct ct_node_set {
    struct ct_node **s;
    int n;
};

#define LEV_MAXX 50
#define LEV_MINN 250

#define DEF_SON_N 10
#define DEF_INCR_NODES 20000
#define MIN_BATCH_CONSTRUCT 0


#define FREED_NOD_INCR 10000


class CoverTree {
public:
    CoverTree(struct svm_parameter *par, int base, bool brute = false, int red_tree = 0, bool silent = 0);

    ~CoverTree();

    void setKs(int k, int kp);

    int getK() {
        return k;
    };

    int getKP() {
        return kp;
    };

    struct svm_parameter *getParam() {
        return param;
    };

    struct problem *addPoints(struct problem *pro);

    void removePoints(struct problem *pro);

    void removePoint(struct node *node);

    struct node *nn(struct node *node);

    double nn_label(struct node *node);

    struct node **knn(struct node *node, int k);

    double *knn_label(struct node *node, int k);

    struct problem *knn_p(struct node *node, int k, bool esc_n = false);

    struct svm_model *nn_model(struct node *node);

    struct svm_model **knn_models(struct node *node, int k);

    int get_n_local_probs() {
        return n_local_probs;
    };

    struct problem **get_trained_probs() {
        return l_probs;
    };

    struct svm_model **get_trained_models() {
        return l_models;
    };

    void set_model(struct svm_model *m, int i) {
        l_models[i] = m;
    };

    struct problem **createNLocalProbs(int nlp, int kk, int k_val, double loc_prob_bil_th);

    void createLocalProbs(bool inc_first = true);

    void save(char *filename);

    void load(char *filename, PyObject *python_svm_class);

private:
    struct ct *tree;
    struct problem *pr_prob;
    struct svm_parameter *param;

    unsigned int init_rand;

    struct ct_node **all_nn(struct problem *test);

    double ct_node_maj(struct ct_node *n);

    bool too_unbalanced(struct problem *p, double r, int k_val, const double *distinct_labels, int distinct_labels_num);

    struct problem **l_probs;
    struct svm_model **l_models;

    bool silent;

    int k, kp;
    int red_tree;

    double BASE;
    int PEN;

    double *ddda;

    int n_local_probs, n_nodes, n_calc_dists, n_reps;

    double *pows, **neighbors;
    struct ct_node **nodes;
    void **nodes2;
    struct ct_node_set *Q, *Qim1;
    int *allocQim1q;

    int alloc_nodes, used_nodes, incr_n_nodes;

    void ***arr;
    int arr_l;
    double *ddd;

    void set_pows(int base);


    struct svm_model *get_model(struct node *px);

    // Internal Core functions
    struct ct_node *nn_int(struct node *node, struct ct *cover_tree);

    struct ct_node_set *knn_int(struct node *node, int kn);

    struct ct_node *
    construct(struct ct_node *p, struct ct_node_set *near, struct ct_node_set *far, int level, struct ct *t);

    void build_local_problem(struct ct_node *p, bool inc_first = true);

    // Supporting functions
    void
    split(struct ct_node *nn, double dist, struct ct_node_set *in1, struct ct_node_set *in2, struct ct_node_set *o1,
          struct ct_node_set *o2);

    void assign_distances(struct ct_node *f, struct ct_node *s);

    struct node **knn_brute(struct node *node, int k);

    struct node **brute_knn_Q(struct node *node, struct ct_node_set *Q, int k);

    struct ct_node **brute_knn_Q_ct_node(struct node *node, struct ct_node_set *Q, int k);

    struct ct_node *brute_nn_Q(struct node *node, struct ct_node_set *Q);

    bool isImplicit(struct ct_node *node);

    bool containsNode(struct ct_node *c, const struct node *n);

    void addRepeatedPoint(struct ct_node *nn, const struct node *n, int index);

    void addChildren(struct ct_node *f, struct ct_node *s);

    // Distance functions
    double dist_n(struct ct_node *ch, struct node *n);

    double dist_dot(const node *px, const node *py);

    double dist_pol(const node *px, const node *py);

    double dist(const struct node *px, const struct node *py);

    void dot_sqr(node *px);

    void dot_sqr_dot(node *px);

    void dot_sqr_pol(node *px);

    // ct_node_set internal functions
    struct ct_node_set *new_ct_node_set(struct ct_node_set *o);

    void add_to_ct_node_set(struct ct_node_set *s, struct ct_node *n);

    void free_ct_node_set(struct ct_node_set *o);

    void rem_from_ct_node_set(struct ct_node_set *s, int i);

    // Memory-related functions
    void allocNodes();

    struct ct_node *clone_node(struct ct_node *node);

    struct ct_node *get_node(struct node *node, int index);

    struct ct_node *get_node(struct node *node, int index, int n_children, double dist_from_f, bool impl);

    struct ct_node *get_node(struct node **x, int card, int *indx, int n_children, double dist_from_f, bool impl);
};


#endif
