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
#include "svm.h"
#include "cover_tree.h"

inline int ddcompare(const void *e1, const void *e2);

inline int dcompare(const void *e1, const void *e2);

inline int mycomp(const void *e1, const void *e2);

inline double powi(double base, int times);

inline double kth_smallest(double *a[2], int n, int k);

void CoverTree::set_pows(int base) {
    PEN = base;
    int tot_lev = LEV_MAXX + LEV_MINN;
    if (PEN > 0) BASE = pow(2.0, (1.0) / ((double) (PEN + 1))) * 1.00001;
    else BASE = 2.0;

    pows = Realloc(pows, double, tot_lev + 1 + PEN);
    if (pows == NULL) MallocError("pows");

    for (int i = 0; i <= tot_lev + PEN; i++)
        pows[i] = pow(BASE, i - LEV_MINN);
    pows[0] = pow(BASE, -LEV_MINN - 1);

    if (!silent) printf("BASE %f, PEN %d\n", BASE, PEN);
    if (!silent) printf("MIN DIST %g \n", pows[0]);
}

CoverTree::CoverTree(struct svm_parameter *par, int base, bool brute, int red_tree, bool silent) {
    if (!silent) printf("\n--- INIT CT          ---\n");
    this->param = par;
    this->tree = NULL;
    this->silent = silent;
    this->n_reps = 0;
    this->red_tree = red_tree;

    n_local_probs = 0;
    pows = NULL;

    init_rand = 1982; // (unsigned)time(NULL)

    incr_n_nodes = DEF_INCR_NODES;

    this->l_probs = NULL;
    this->l_models = NULL;

    n_nodes = 0;

    set_pows(base);

    used_nodes = 0;
    alloc_nodes = 0;

    n_calc_dists = 0;

    arr = NULL;
    arr_l = 0;
    ddd = NULL;

    Q = Malloc(struct ct_node_set, 1);
    if (Q == NULL) MallocError("Q");
    Q->s = NULL;
    Qim1 = Malloc(struct ct_node_set, 1);
    if (Qim1 == NULL) MallocError("Qim1");
    Qim1->s = NULL;

    neighbors = NULL;
    nodes2 = NULL;
    nodes = NULL;

    ddda = NULL;

    if (!silent) printf("--- ---------------  ---\n");
}

CoverTree::~CoverTree() {
    if (Q != NULL) {
        if (Q->s != NULL) free(Q->s);
        free(Q);
    }

    if (Qim1 != NULL) {
        if (Qim1->s != NULL) free(Qim1->s);
        free(Qim1);
    }

    if (neighbors != NULL) {
        for (int i = 0; i < tree->prob->l; i++)
            free(neighbors[i]);
        free(neighbors);
    }

    if (nodes2 != NULL) free(nodes2);
    if (tree != NULL) {
        if (tree->tmp_root != NULL) {
            free(tree->tmp_root->s);
            free(tree->tmp_root);
        }

        tree->tmp_root = NULL;

        if (tree->root != NULL) {
            if (tree->root->s != NULL) free(tree->root->s);
            free(tree->root);
        }

        if (tree->prob != NULL) {
            if (tree->prob->x != NULL) free(tree->prob->x);
            if (tree->prob->y != NULL) free(tree->prob->y);
            if (tree->prob->k_rank != NULL) free(tree->prob->k_rank);
            if (tree->prob->lpm_ind != NULL) free(tree->prob->lpm_ind);
            free(tree->prob);
        }
        free(tree);
    }

    if (used_nodes > 0) {
        for (int i = 0; i < used_nodes; i++) {
            free(nodes[i]->ind);
            free(nodes[i]->children);
            free(nodes[i]->x);
            free(nodes[i]->tmp_dist_c);
            free(nodes[i]->tmp_dist_node_c);
        }
        free(nodes);
    }
    free(pows);


    if (arr != NULL) {
        for (int i = 0; i < arr_l; i++)
            free(arr[i]);
        free(arr);
    }


    if (ddd != NULL) free(ddd);
    if (ddda != NULL) free(ddda);
    if (l_probs != NULL) {
        for (int i = 0; i < n_local_probs; i++) {
            free(l_probs[i]->x);
            free(l_probs[i]->y);
        }
        free(l_probs);
    }

    if (l_models != NULL) {
        for (int i = 0; i < n_local_probs; i++) {
            if (param->python_svm) {
                python_svm_destroy_model(l_models[i], true);
            } else {
                svm_destroy_model(l_models[i], true);
            }
        }
        free(l_models);
    }
}

struct node **CoverTree::knn(struct node *node, int k) {
    dot_sqr(node);
    dist_n(tree->tmp_root->s[0], node);
    tree->tmp_root->s[0]->dist_from_f = 0.0;
    struct ct_node_set *ns = knn_int(node, k);
    return brute_knn_Q(node, ns, k);
}

struct problem *CoverTree::knn_p(struct node *node, int k, bool esc_n) {
    dot_sqr(node);
    dist_n(tree->tmp_root->s[0], node);
    tree->tmp_root->s[0]->dist_from_f = 0.0;

    if (esc_n) k++;
    struct ct_node_set *ns = knn_int(node, k);
    struct ct_node **knn = brute_knn_Q_ct_node(node, ns, k);

    int nc = 0;
    for (int i = 0; i < k; i++) {
        nc += knn[i]->card;
        if (nc >= k) break;
    }

    struct problem *p = Malloc(struct problem, 1);
    if (p == NULL) MallocError("p");
    p->l = nc;
    if (esc_n) p->l--;
    p->x = Malloc(struct node*, p->l);
    if (p->x == NULL) MallocError("p->x");
    p->y = Malloc(double, p->l);
    if (p->y == NULL) MallocError("p->y");

    int nnn = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < knn[i]->card; j++) {
            if (nnn >= p->l) break;
            struct node *noden = knn[i]->x[j];

            if (esc_n && i == 0 && node == noden) continue;

            p->x[nnn] = noden;
            p->y[nnn] = tree->prob->y[knn[i]->ind[j]];
            nnn++;
        }
        if (nnn >= p->l) break;
    }

    free(knn);

    if (esc_n) return p;

    if (p->x[0] == node) return p;

    for (int i = 1; i < nnn; i++)
        if (p->x[i] == node) {
            struct node *tmp = p->x[i];
            double tmpd = p->y[i];
            p->x[i] = p->x[0];
            p->y[i] = p->y[0];
            p->x[0] = tmp;
            p->y[0] = tmpd;
            return p;
        }

    printf(" BIG ERROR %.20f\n", dist(p->x[0], node));
    exit(0);

    return p;
}

double CoverTree::ct_node_maj(struct ct_node *n) {
    if (n->card == 1) return tree->prob->y[n->ind[0]];

    double prev = -1001.0;

    bool diff = false;

    double *classes = Malloc(double, n->card);
    if (classes == NULL) MallocError("classes");
    for (int i = 0; i < n->card; i++)
        classes[i] = -1001.0;

    int n_classes = 0;
    double *labs = Malloc(double, n->card);
    if (labs == NULL) MallocError("labs");
    int count = 0;

    for (int i = 0; i < n->card; i++) {
        if (tree->prob->y[n->ind[i]] != prev) {
            diff = true;
            double lab = tree->prob->y[n->ind[i]];

            bool found = false;
            for (int j = 0; j < i; j++)
                if (lab == classes[j])
                    found = true;

            if (!found) {
                classes[i] = lab;
                n_classes++;
                labs[count] = lab;
                count++;
            }
        }

        prev = tree->prob->y[n->ind[i]];
    }

    if (n_classes == 1) {
        free(classes);
        free(labs);
        return tree->prob->y[n->ind[0]];
    }


    int *nlabs = Malloc(int, n_classes);
    if (nlabs == NULL) MallocError("nlabs");
    for (int j = 0; j < n_classes; j++)
        nlabs[j] = 0;

    for (int i = 0; i < n->card; i++) {
        double lab = tree->prob->y[n->ind[i]];
        for (int j = 0; j < n_classes; j++)
            if (lab == labs[j])
                nlabs[j]++;
    }

    int max_ind = -1;
    int max = -1;

    for (int j = 0; j < n_classes; j++)
        if (max < nlabs[j]) {
            max = nlabs[j];
            max_ind = j;
        }

    double lll = labs[max_ind];

    free(classes);
    free(labs);
    free(nlabs);

    return lll;
}

double CoverTree::nn_label(struct node *node) {
    dot_sqr(node);
    dist_n(tree->tmp_root->s[0], node);
    tree->tmp_root->s[0]->dist_from_f = 0.0;
    struct ct_node *n = nn_int(node, tree);
    return ct_node_maj(n);
}

void print_ct_node_set(struct ct_node_set *n, struct ct *tree) {
    for (int i = 0; i < n->n; i++) {
        struct ct_node *nod = n->s[i];
        for (int j = 0; j < nod->card; j++) {
            struct node *nodd = nod->x[j];
            while (nodd->index > 0) nodd++;
            printf("%g ", tree->prob->y[-nodd->index]);

            nodd = nod->x[j];
            while (nodd->index > 0) {
                printf("%d:%g ", nodd->index, nodd->value);
                nodd++;
            }
            printf("\n");
        }
    }
}

/*
double *CoverTree::knn_label(struct node *node, int k)
{
	dot_sqr(node);
	dist_n(tree->tmp_root->s[0],node);
	tree->tmp_root->s[0]->dist_from_f = 0.0;
	struct ct_node_set *n = knn_int(node,k);


	//print_ct_node_set(n,tree);


	double *ret = Malloc(double,k);

	int count = 0;
	for( int i = 0; i < n->n; i++ )
		for( int j = 0; j < n->s[i]->card; j++ )
			if( count < k ) ret[count++] = tree->prob->y[n->s[i]->ind[j]];

	return ret;
}
*/


double *CoverTree::knn_label(struct node *node, int k) {
    dot_sqr(node);
    dist_n(tree->tmp_root->s[0], node);
    tree->tmp_root->s[0]->dist_from_f = 0.0;
    struct ct_node_set *n = knn_int(node, k);

    struct ct_node **knn = brute_knn_Q_ct_node(node, n, k);
    //print_ct_node_set(n,tree);


    double *ret = Malloc(double, k);

    int count = 0;
    for (int i = 0; i < n->n; i++)
        for (int j = 0; j < n->s[i]->card; j++)
            if (count < k) ret[count++] = tree->prob->y[knn[i]->ind[j]];

    free(knn);

    return ret;
}

struct node *CoverTree::nn(struct node *node) {
    dot_sqr(node);
    dist_n(tree->tmp_root->s[0], node);
    tree->tmp_root->s[0]->dist_from_f = 0.0;
    struct ct_node *n = nn_int(node, tree);
    return n->x[0];
}


struct problem *CoverTree::addPoints(struct problem *pro) {
    if (!silent) printf("\n--- ADD POINTS       ---\n");
    if (pro == NULL || pro->l < 1) return NULL;
    int prec_dists = this->n_calc_dists;

    bool batch_construct = false;

    if (tree == NULL) {
        tree = Malloc(struct ct, 1);
        if (this->tree == NULL) MallocError("tree");
        tree->prob = Malloc(struct problem, 1);
        if (this->tree->prob == NULL) MallocError("tree->prob");
        tree->root = NULL;
        tree->tmp_root = NULL;
        tree->prob->x = NULL;
        tree->prob->y = NULL;
        tree->prob->k_rank = NULL;
        tree->prob->lpm_ind = NULL;
        tree->prob->l = 0;
        tree->param = this->param;
        tree->rt_min_lev = LEV_MAXX;
        tree->rt_max_lev = -LEV_MINN;
        if (pro->l >= MIN_BATCH_CONSTRUCT) batch_construct = true;
    }

    int first_added = false;
    int pro_added = 0;

    if (tree->root == NULL) {
        if (pro->l >= MIN_BATCH_CONSTRUCT) batch_construct = true;
        struct ct_node *svt = get_node(NULL, -1);
        svt->n_children = 1;
        svt->card = 1;

        if (tree->prob->l < 1) {
            tree->prob->x = Malloc(struct node*, 1);
            if (tree->prob->x == NULL) MallocError("tree->prob->x");
            tree->prob->y = Malloc(double, 1);
            if (tree->prob->y == NULL) MallocError("tree->prob->y");
            tree->prob->k_rank = Malloc(int, 1);
            if (tree->prob->k_rank == NULL) MallocError("tree->prob->k_rank");
            tree->prob->lpm_ind = Malloc(int, 1);
            if (tree->prob->lpm_ind == NULL) MallocError("tree->prob->lpm_ind");
        }

        int index;

        {
            index = 0;
            tree->prob->x[index] = pro->x[index];
            tree->prob->y[index] = pro->y[index];
            tree->prob->k_rank[index] = -1;
            tree->prob->lpm_ind[index] = -1;
            first_added = true;
        }

        tree->root = Malloc(struct ct_node_set, 1);
        if (tree->root == NULL) MallocError("tree->root");
        tree->root->n = 1;
        tree->root->s = Malloc(struct ct_node *, tree->root->n);
        if (tree->root->s == NULL) MallocError("tree->root->s");

        tree->tmp_root = Malloc(struct ct_node_set, 1);
        if (tree->tmp_root == NULL) MallocError("tree->tmp_root");
        tree->tmp_root->n = 1;
        tree->tmp_root->s = Malloc(struct ct_node *, tree->tmp_root->n);
        if (tree->tmp_root->s == NULL) MallocError("tree->tmp_root->s");

        if (!batch_construct) {
            svt->x[0] = tree->prob->x[index];
            svt->dist_from_f = 0.0;

            svt->ind[0] = index;
            svt->children = Malloc(struct ct_node*, DEF_SON_N);
            if (svt->children == NULL) MallocError("svt->children");
            svt->children[0] = svt;

            tree->root = Malloc(struct ct_node_set, 1);
            if (tree->root == NULL) MallocError("tree->root");
            tree->root->n = 1;
            tree->root->s = Malloc(struct ct_node *, tree->root->n);
            if (tree->root->s == NULL) MallocError("tree->root->s");
            tree->root->s[0] = svt;

            tree->tmp_root = Malloc(struct ct_node_set, 1);
            if (tree->tmp_root == NULL) MallocError("tree->tmp_root");
            tree->tmp_root->n = 1;
            tree->tmp_root->s = Malloc(struct ct_node *, tree->tmp_root->n);
            if (tree->tmp_root->s == NULL) MallocError("tree->tmp_root->s");
            tree->tmp_root->s[0] = svt;

            n_nodes++;
        }
    }

    int to_add = pro->l - pro_added;
    int old_l = tree->prob->l;
    tree->prob->l = tree->prob->l + to_add;

    if (tree->prob->l > old_l) {
        tree->prob->x = Realloc(tree->prob->x, struct node*, tree->prob->l);
        if (tree->prob->x == NULL) ReallocError("tree->prob->x");
        tree->prob->y = Realloc(tree->prob->y, double, tree->prob->l);
        if (tree->prob->y == NULL) ReallocError("tree->prob->y");
        tree->prob->k_rank = Realloc(tree->prob->k_rank, int, tree->prob->l);
        if (tree->prob->k_rank == NULL) ReallocError("tree->prob->k_rank");
        tree->prob->lpm_ind = Realloc(tree->prob->lpm_ind, int, tree->prob->l);
        if (tree->prob->lpm_ind == NULL) ReallocError("tree->prob->lpm_ind");
    }


    for (int i = pro_added; i < pro->l; i++) // memcopy!
    {
        tree->prob->x[old_l + i - pro_added] = pro->x[i];
        tree->prob->y[old_l + i - pro_added] = pro->y[i];
        tree->prob->k_rank[old_l + i - pro_added] = -1;
        tree->prob->lpm_ind[old_l + i - pro_added] = -1;
        dot_sqr(tree->prob->x[old_l + i - pro_added]);
    }

    if (tree->prob->l > old_l) {
        Q->s = Realloc(Q->s, struct ct_node*, tree->prob->l);
        if (Q->s == NULL) ReallocError("Q->s");
        ddda = Realloc(ddda, double, tree->prob->l);
        if (ddda == NULL) ReallocError("Q->s");
        Qim1->s = Realloc(Qim1->s, struct ct_node*, tree->prob->l);
        if (Qim1->s == NULL) ReallocError("Qim1->s");
        neighbors = Realloc(neighbors, double*, tree->prob->l);
        if (neighbors == NULL) ReallocError("neighbors");
        nodes2 = Realloc(nodes2, void*, tree->prob->l);
        if (nodes2 == NULL) ReallocError("nodes2");
        for (int i = old_l; i < tree->prob->l; i++) {
            neighbors[i] = Malloc(double, 2);
            if (neighbors[i] == NULL) MallocError("neighbors[i]");
        }
    }

    allocNodes();


    struct ct_node_set *all = NULL;
    struct ct_node_set *v = NULL;

    all = new_ct_node_set(all);
    v = new_ct_node_set(v);

    struct ct_node *first = get_node(tree->prob->x[old_l], old_l);

    for (int i = 1; i < to_add; i++) {
        struct ct_node *n = get_node(tree->prob->x[old_l + i], old_l + i);
        add_to_ct_node_set(all, n);
    }

    n_nodes++;

    tree->root->s[0] = construct(first, all, v, LEV_MAXX, tree);
    if (!silent) printf("allocated nodes %d\n", alloc_nodes);


    if (!silent) printf("Added %d points [reps = %d]. Batch construct %d\n", pro->l, n_reps, batch_construct);
    if (!silent) printf("Calculated dists %d [total %d]\n", n_calc_dists - prec_dists, n_calc_dists);
    if (!silent) printf("CT Max LEV %d, min LEV %d\n", tree->rt_max_lev, tree->rt_min_lev);
    if (!silent) printf("--- ---------------- ---\n");

    return tree->prob;
}


void CoverTree::build_local_problem(struct ct_node *p, bool inc_first) {
    bool all_done = true;
    for (int i = 0; i < p->card; i++)
        if (tree->prob->k_rank[p->ind[i]] < 0) {
            all_done = false;
            break;
        }
    if (all_done) return;

    struct ct_node_set *near = knn_int(p->x[0], k);

    int ln = 0;

    for (int i = 0; i < near->n; i++) ln += near->s[i]->card;

    int nn = 0;
    if (arr_l < ln) {
        int old_arr_l = arr_l;
        arr = Realloc(arr, void **, ln);
        if (arr == NULL) ReallocError("arr");
        ddd = Realloc(ddd, double, ln);
        if (ddd == NULL) ReallocError("ddd");
        arr_l = ln;

        for (int i = old_arr_l; i < arr_l; i++) {
            arr[i] = Malloc(void *, 4);
            if (arr[i] == NULL) MallocError("arr[i]");
        }
    }

    int st = 0;
    if (!inc_first) st++;

    for (int i = st; i < near->n; i++)
        for (int j = 0; j < near->s[i]->card; j++) {
            ddd[nn] = dist_n(near->s[i], p->x[0]);
            int indl = near->s[i]->ind[j];
            arr[nn][0] = (void *) &(ddd[nn]);
            arr[nn][1] = (void *) tree->prob->x[indl];
            arr[nn][2] = (void *) &tree->prob->y[indl];
            arr[nn][3] = (void *) &near->s[i]->ind[j];
            nn++;
        }

    qsort(arr, nn, sizeof(void *), mycomp);

    n_local_probs++;

    l_probs = Realloc(l_probs, struct problem *, n_local_probs);
    l_probs[n_local_probs - 1] = Malloc(struct problem, 1);
    if (l_probs[n_local_probs - 1] == NULL) MallocError("l_probs[n_local_probs-1]");
    l_models = Realloc(l_models, struct svm_model *, n_local_probs);

    struct problem *pr = l_probs[n_local_probs - 1];

    pr->l = k;
    pr->x = Malloc(struct node *, pr->l);
    if (pr->x == NULL) MallocError("pr->x");
    pr->y = Malloc(double, pr->l);
    if (pr->y == NULL) MallocError("pr->y");


    for (int i = 0; i < pr->l; i++) {
        pr->x[i] = (struct node *) arr[i][1];
        pr->y[i] = *((double *) arr[i][2]);
        int inddd = *((int *) arr[i][3]);

        if (i < kp && (tree->prob->k_rank[inddd] < 0 || i < tree->prob->k_rank[inddd])) {
            tree->prob->k_rank[inddd] = i;
            tree->prob->lpm_ind[inddd] = n_local_probs - 1;
        }
    }

}


void CoverTree::setKs(int k, int kp) {
    this->k = k;
    this->kp = kp;
}

struct ct_node *CoverTree::construct(struct ct_node *p,
                                     struct ct_node_set *near, struct ct_node_set *far,
                                     int level,
                                     struct ct *t
) {
    double plev = pows[level + LEV_MINN];
    double plevm1 = pows[level - 1 + LEV_MINN];

    if (near->n == 0) {
        if (level < t->rt_min_lev) t->rt_min_lev = level;
        return p;
    }

    struct ct_node *ret = NULL;
    struct ct_node_set *near_s_n = NULL;
    struct ct_node_set *near_s_f = NULL;
    near_s_n = new_ct_node_set(near_s_n);
    near_s_f = new_ct_node_set(near_s_f);

    split(p, plevm1, near, NULL, near_s_n, near_s_f);

    struct ct_node *self = construct(p, near_s_n, near_s_f, level - 1, t);
    free_ct_node_set(near_s_n);

    struct ct_node *n_p = clone_node(self);
    n_p->n_children = 1;
    n_p->children[0] = self;
    self->dist_from_f = 0.0;
    ret = n_p;
    if (self->n_children > 1 && level == t->rt_max_lev + 1) t->tmp_root->s[0] = self;

    struct ct_node_set *o11_s = NULL;
    struct ct_node_set *o22_s = NULL;
    o11_s = new_ct_node_set(o11_s);
    o22_s = new_ct_node_set(o22_s);

    while (near_s_f->n > 0) {
        struct ct_node *q = near_s_f->s[--near_s_f->n];

        split(q, plevm1, near_s_f, far, o11_s, o22_s);
        struct ct_node *child = construct(q, o11_s, o22_s, level - 1, t);

        child->dist_from_f = dist_n(child, ret->x[0]);
        addChildren(ret, child);
        n_nodes++;

        if (level > t->rt_max_lev) t->rt_max_lev = level;

        split(ret, plev, o22_s, NULL, near_s_f, far);

        o11_s->n = 0;
        o22_s->n = 0;
    }

    free_ct_node_set(o11_s);
    free_ct_node_set(o22_s);
    free_ct_node_set(near_s_f);

    return ret;
}

void CoverTree::createLocalProbs(bool inc_first) {
    int todo = 0;
    int old_lp = this->n_local_probs;
    int old_dists = this->n_calc_dists;

    if (!silent) printf("\n--- START CALC LOCAL PROBS ---\n");

    for (int i = 0; i < this->tree->prob->l; i++)
        if (this->tree->prob->k_rank[i] < 0) todo++;

    struct ct_node_set *csi = Malloc(struct ct_node_set, 1);
    if (csi == NULL) MallocError("csi");
    struct ct_node_set *csmi;

    csi->n = tree->tmp_root->n;
    if (csi->n != 1) return;

    csi->s = Malloc(struct ct_node*, csi->n);
    if (csi->s == NULL) MallocError("csi->s");
    csi->s[0] = tree->tmp_root->s[0];

    for (int i = tree->rt_max_lev; i >= tree->rt_min_lev; i--) {
        int ch = 0;
        for (int j = 0; j < csi->n; j++) {
            build_local_problem(csi->s[j], inc_first);
            if (!isImplicit(csi->s[j]))
                ch += csi->s[j]->n_children;
        }

        csmi = Malloc(struct ct_node_set, 1);
        if (csmi == NULL) MallocError("csmi");
        csmi->n = ch;
        csmi->s = Malloc(struct ct_node*, csmi->n);
        if (csmi->s == NULL) MallocError("csmi->s");

        ch = 0;
        for (int j = 0; j < csi->n; j++)
            for (int c = 0; c < csi->s[j]->n_children; c++)
                if (!isImplicit(csi->s[j]))
                    csmi->s[ch++] = csi->s[j]->children[c];

        free(csi->s);
        free(csi);
        csi = csmi;

        if (!silent)
            printf("createLocalProbs: level %*d [max %d min %*d] (%d nodes %d local probs) ------------- \n", 2, i,
                   tree->rt_max_lev, 2, tree->rt_min_lev, csi->n, n_local_probs);
    }

    free(csi->s);
    free(csi);

    todo = 0;
    for (int i = 0; i < this->tree->prob->l; i++)
        if (this->tree->prob->k_rank[i] < 0) todo++;

    if (!silent) printf("\n--- CALC LOCAL PROBS ---\n");
    if (!silent)
        printf("Built %d local probs [total %d] [k = %d, kp = %d]\n", this->n_local_probs - old_lp, n_local_probs, k,
               kp);
    if (!silent) printf("Calculated dists %d [total %d]\n", n_calc_dists - old_dists, n_calc_dists);
    if (!silent) printf("--- ---------------- ---\n");
}


bool CoverTree::too_unbalanced(struct problem *p, double r, int k_val, const double *distinct_labels, int distinct_labels_num) {
    double labels_frequencies [distinct_labels_num];
    for (int j = 0; j < distinct_labels_num; j++) {
        labels_frequencies[j] = 0.0;
    }

    for (int i = 0; i < k_val; i++) {
        bool label_found = false;
        for (int j = 0; j < distinct_labels_num && !label_found; j++) {
            if (p->y[i] == distinct_labels[j]) {
                labels_frequencies[j]++;
                label_found = true;
            }
        }
        if (!label_found) {
            printf("Error, label not found!");
            exit(1);
        }
    }

    for (int j = 0; j < distinct_labels_num; j++) {
        labels_frequencies[j] /= (double) k_val;
        if (!silent)
            printf("%2.0f perc = %6.2f [r=%5.3f] --> return %d\n", distinct_labels[j], labels_frequencies[j] * 100,
                   r, labels_frequencies[j] < r || labels_frequencies[j] > 1.0 - r);

        if (labels_frequencies[j] < r || labels_frequencies[j] > 1.0 - r)
            return true;
    }

    return false;

//    double fl = p->y[0];
//    double fln = 0.0;
//
//    for (int i = 0; i < k_val; i++)
//        if (fl == p->y[i])
//            fln++;
//
//    double frac = fln / (double) k_val;
//
//    if (!silent) printf("%f perc = %f [r=%f] --> return %d\n", fl, frac * 100, r, frac < r || frac > 1.0 - r);
//
//    if (frac < r || frac > 1.0 - r) return true;
//    return false;
}

struct problem **CoverTree::createNLocalProbs(int nlp, int kk, int k_val, double loc_prob_bil_th) {
    struct problem **ps = Malloc(struct problem *, nlp);
    if (ps == NULL) MallocError("ps");

    // srand ( init_rand );

    double *distinct_labels = Malloc(double, tree->prob->l);
    int distinct_labels_num = 0;
    for (int i = 0; i < tree->prob->l; i++) {
        bool unseen_label = true;
        for (int j = 0; j < distinct_labels_num && unseen_label; j++) {
            if (tree->prob->y[i] == distinct_labels[j]) {
                unseen_label = false;
            }
        }
        if (unseen_label){
            distinct_labels[distinct_labels_num] = tree->prob->y[i];
            distinct_labels_num++;
        }
    }
    distinct_labels = Realloc(distinct_labels, double, distinct_labels_num);
    if (distinct_labels_num != 0 && distinct_labels == NULL) MallocError("distinct_labels");

    printf("\n");
    for (int m = 0; m < nlp; m++) {
        int rrr = rand() % tree->prob->l;

        struct problem *pp = knn_p(tree->prob->x[rrr], kk);
        double sb = 1.0 / (distinct_labels_num + 1); // loc_prob_bil_th;
        while (too_unbalanced(pp, sb, k_val, distinct_labels, distinct_labels_num)) {
            rrr = rand() % tree->prob->l;
            pp = knn_p(tree->prob->x[rrr], kk);
            sb -= 0.002;
        }

        ps[m] = pp;
        if (m != nlp-1)
            printf("\n");
    }

    free(distinct_labels);

    return ps;
}


void CoverTree::addRepeatedPoint(struct ct_node *nn, const struct node *n, int index) {
    n_reps++;

    nn->card++;
    nn->x = Realloc(nn->x, struct node *, nn->card);
    if (nn->x == NULL) ReallocError("nn->x");
    nn->x[nn->card - 1] = (struct node *) n;
    nn->ind = Realloc(nn->ind, int, nn->card);
    if (nn->ind == NULL) ReallocError("nn->ind");
    nn->ind[nn->card - 1] = index;
}

struct svm_model *CoverTree::nn_model(struct node *node) {
    dot_sqr(node);
    dist_n(tree->tmp_root->s[0], node);
    tree->tmp_root->s[0]->dist_from_f = 0.0;
    struct ct_node *nn = nn_int(node, tree);
    return get_model(nn->x[0]);
}

struct svm_model **CoverTree::knn_models(struct node *node, int k) {
    struct svm_model **ret = Malloc(struct svm_model *, k);
    dot_sqr(node);
    struct ct_node_set *knn = knn_int(node, k);
    for (int i = 0; i < k; i++)
        ret[i] = get_model(knn->s[i]->x[0]);
    free(knn);
    return ret;
}

struct svm_model *CoverTree::get_model(struct node *px) {
    while (px->index > 0) ++px;
    return l_models[tree->prob->lpm_ind[-px->index]];
}

struct ct_node *CoverTree::nn_int(struct node *node, struct ct *cover_tree) {
    /* register */ int i, j, h, k;
    /* register */ struct ct_node_set *Q_l;
    /* register */ double m_d = FLT_MAX, cmp = 0.0, d;

    if (cover_tree == NULL) return NULL;

    Q_l = cover_tree->tmp_root;
    struct ct_node *nearest = NULL;
    struct ct_node *tmp = NULL;

    for (i = cover_tree->rt_max_lev; i >= cover_tree->rt_min_lev; i--) {
        k = 0;
        m_d = FLT_MAX;

        if (Q_l->n == 1 && isImplicit(Q_l->s[0])) {
            nearest = Q_l->s[0];
            break;
        }

        for (j = 0; j < Q_l->n; j++) {
            struct ct_node *Qisj = Q_l->s[j];
            for (h = 0; h < Qisj->n_children; h++) {
                tmp = Qisj->children[h];

                if (tmp->dist_from_f == 0.0) assign_distances(Qisj, tmp);
                //if( dist_n(Qisj,node) - tmp->dist_from_f > m_d + pows[i+LEV_MINN + PEN] )
                //	continue;

                d = dist_n(tmp, node);
                if (d < m_d) {
                    m_d = d;
                    nearest = tmp;
                } else if (isImplicit(tmp)) continue;
                Q->s[k] = tmp;
                ddda[k] = d;
                k++;
            }
        }

        Q->n = k;

        if (k > 0) cmp = m_d + pows[i + LEV_MINN + PEN];

        for (j = 0, k = 0; j < Q->n; j++)
            if (ddda[j] <= cmp)
                Qim1->s[k++] = Q->s[j];

        Qim1->n = k;
        Q_l = Qim1;
    }

    if (nearest == NULL) {
        nearest = cover_tree->tmp_root->s[0];
        if (!silent) printf("ERROR\n");
    }
    return nearest;
}

void CoverTree::assign_distances(struct ct_node *f, struct ct_node *s) {
    for (int i = 0; i < LCACHE; i++) {
        s->tmp_dist_c[i] = f->tmp_dist_c[i];
        s->tmp_dist_node_c[i] = f->tmp_dist_node_c[i];
    }
    s->last = f->last;
}

struct ct_node_set *CoverTree::knn_int(struct node *node, int kn) {
    /* register */ int i, j, h, kk = -1;
    /* register */ struct ct_node_set *Q_l;
    /* register */ double cmp;

    if (tree == NULL || tree->root->n == 0) return NULL;

    Q_l = tree->tmp_root;

    for (i = tree->rt_max_lev; i >= tree->rt_min_lev; i--) {
        kk = 0;
        for (j = 0; j < Q_l->n; j++) {
            struct ct_node *Qisj = Q_l->s[j];
            for (h = 0; h < Qisj->n_children; h++) {
                Q->s[kk] = Qisj->children[h];
                if (Q->s[kk]->dist_from_f == 0.0) {
                    assign_distances(Qisj, Q->s[kk]);
                    ddda[kk] = dist_n(Qisj, node);
                } else ddda[kk] = dist_n(Q->s[kk], node);
                neighbors[kk][0] = ddda[kk];
                kk++;
            }
        }

        Q->n = kk;

        if (kn > kk)
            for (j = 0, kk = 0; j < Q->n; j++)
                Qim1->s[kk++] = Q->s[j];
        else {
            cmp = kth_smallest(neighbors, kk, kn - 1) + pows[i + LEV_MINN + PEN];
            for (j = 0, kk = 0; j < Q->n; j++)
                if (ddda[j] <= cmp)
                    Qim1->s[kk++] = Q->s[j];
        }

        Qim1->n = kk;
        Q_l = Qim1;
    }

    Qim1->n = kk;
    Q_l = Qim1;

    return Q_l;
}

/*
 * DISTANCE FUNCTIONS
 */
inline double CoverTree::dist_n(struct ct_node *ch, struct node *n) {
    if (ch->tmp_dist_node_c[ch->last] == NULL) {
        ch->tmp_dist_c[ch->last] = dist(n, ch->x[0]);
        ch->tmp_dist_node_c[ch->last] = n;
        return ch->tmp_dist_c[ch->last];
    }

    for (int i = 0; i < LCACHE; i++)
        if (ch->tmp_dist_node_c[i] == n) {
            ch->last = i;
            return ch->tmp_dist_c[i];
        }

    int ind = (ch->last + 1) % LCACHE;

    ch->tmp_dist_c[ind] = dist(n, ch->x[0]);
    ch->tmp_dist_node_c[ind] = n;

    ch->last = ind;

    return ch->tmp_dist_c[ind];
}


inline void CoverTree::dot_sqr(node *px) {
    if (param->kernel_type == LINEAR || param->kernel_type == RBF)
        dot_sqr_dot(px);
    else if (param->kernel_type == POLY)
        dot_sqr_pol(px);
}


inline void CoverTree::dot_sqr_dot(node *px) {
    double sum = 0.0;
    while (px->index > 0) {
        sum += px->value * px->value;
        ++px;
    }
    px->value = sum;
}


inline void CoverTree::dot_sqr_pol(node *px) {
    double sum = 0.0;
    while (px->index > 0) {
        sum += px->value * px->value;
        ++px;
    }
    px->value = powi(param->gamma * sum + param->coef0, param->degree);
}

inline double CoverTree::dist_dot(const node *px, const node *py) {

    double sum = 0;
    while (px->index > 0 && py->index > 0) {
        if (px->index == py->index) {
            sum += px->value * py->value;
            ++px;
            ++py;
        } else {
            if (px->index > py->index)
                ++py;
            else
                ++px;
        }
    }
    while (px->index > 0) ++px;
    while (py->index > 0) ++py;
    return sqrt(MAX(px->value + py->value - 2.0 * sum, 0.0));
}

inline double powi(double base, int times) {
    double tmp = base, ret = 1.0;

    for (int t = times; t > 0; t /= 2) {
        if (t % 2 == 1) ret *= tmp;
        tmp = tmp * tmp;
    }
    return ret;
}

inline double CoverTree::dist_pol(const node *px, const node *py) {

    double sum = 0.0;
    while (px->index > 0 && py->index > 0) {
        if (px->index == py->index) {
            sum += px->value * py->value;
            ++px;
            ++py;
        } else {
            if (px->index > py->index)
                ++py;
            else
                ++px;
        }
    }
    while (px->index > 0) ++px;
    while (py->index > 0) ++py;

    sum = powi(param->gamma * sum + param->coef0, param->degree);
    return sqrt(MAX(px->value + py->value - 2.0 * sum, 0.0));
}

inline double CoverTree::dist(const struct node *px, const struct node *py) // upper?
{
    n_calc_dists++;
    if (param->kernel_type == LINEAR || param->kernel_type == RBF)
        return dist_dot(px, py);
    else if (param->kernel_type == POLY)
        return dist_pol(px, py);
    else return -1;
}

inline bool CoverTree::isImplicit(struct ct_node *node) {
    return node->n_children == 1 && node->children[0] == node;
}

inline bool CoverTree::containsNode(struct ct_node *c, const struct node *n) {
    for (int i = 0; i < c->card; i++)
        if (c->x[i] == n)
            return true;
    return false;
}

/*
 * SUPPORTING BRUTE FUNCTIONS
 */
struct node **CoverTree::knn_brute(struct node *node, int k) {
    int i;
    struct node **knn;

    int j = 0;
    for (i = 0; i < tree->prob->l; i++)
        if (tree->prob->x[i] != NULL) {
            neighbors[j][0] = dist(node, tree->prob->x[i]);
            neighbors[j][1] = (double) i;
            j++;
        }

    if (k > j) return NULL;

    qsort(neighbors, j, sizeof(double *), ddcompare);

    knn = Malloc(struct node *, k);
    if (knn == NULL) MallocError("knn");

    for (i = 0; i < k; i++)
        knn[i] = tree->prob->x[(int) neighbors[i][1]];

    return knn;
}

struct ct_node *CoverTree::brute_nn_Q(struct node *node, struct ct_node_set *Q) {
    int i;
    struct ct_node *nn_neighbor = NULL;
    double min_dist = DBL_MAX;

    for (i = 0; i < Q->n; i++) {
        struct ct_node *Qsi = Q->s[i];
        double dd = dist_n(Qsi, node);
        if (dd < min_dist) {
            min_dist = dd;
            nn_neighbor = Qsi;
        }
    }
    return nn_neighbor;
}


void CoverTree::split(struct ct_node *nn, double dist,
                      struct ct_node_set *in1, struct ct_node_set *in2,
                      struct ct_node_set *o1, struct ct_node_set *o2) {
    struct node *n = nn->x[0];
    double dist_p_2 = 2.0 * dist;

    for (int i = in1->n - 1; i >= 0; i--) {
        struct ct_node *el = in1->s[i];
        double d = dist_n(el, n);

        if (d <= dist) {
            if (d < pows[0]) addRepeatedPoint(nn, el->x[0], el->ind[0]);
            else add_to_ct_node_set(o1, el);
        } else if (d <= dist_p_2)
            add_to_ct_node_set(o2, el);
        else continue;

        rem_from_ct_node_set(in1, i);
    }

    if (in2 == NULL || in2->n == 0) return;

    for (int i = in2->n - 1; i >= 0; i--) {
        struct ct_node *el = in2->s[i];
        double d = dist_n(el, n);

        if (d <= dist) {
            if (d < pows[0]) addRepeatedPoint(nn, el->x[0], el->ind[0]);
            else add_to_ct_node_set(o1, el);
        } else if (d <= dist_p_2) add_to_ct_node_set(o2, el);
        else continue;

        rem_from_ct_node_set(in2, i);
    }
}


struct node **CoverTree::brute_knn_Q(struct node *node, struct ct_node_set *Q, int k) {
    int i, j, count = 0;

    count = 0;

    struct node **knn = Malloc(struct node *, k);
    if (knn == NULL) MallocError("knn");

    for (i = 0; i < Q->n; i++) {
        struct ct_node *Qsi = Q->s[i];

        for (j = 0; j < Qsi->card; j++) {
            neighbors[count][0] = dist_n(Qsi, node);
            neighbors[count][1] = (double) count;
            nodes2[count] = Qsi->x[j];
            count++;
        }
    }

    qsort(neighbors, count, sizeof(double *), ddcompare);

    for (i = 0; i < k; i++)
        knn[i] = (struct node *) nodes2[(int) neighbors[i][1]];

    return knn;
}

struct ct_node **CoverTree::brute_knn_Q_ct_node(struct node *node, struct ct_node_set *Q, int k) {
    int i, count = 0;

    count = 0;

    struct ct_node **knn = Malloc(struct ct_node *, k);
    if (knn == NULL) MallocError("knn");

    for (i = 0; i < Q->n; i++) {
        struct ct_node *Qsi = Q->s[i];

        neighbors[count][0] = dist_n(Qsi, node);
        neighbors[count][1] = (double) count;
        nodes2[count] = Qsi;
        count++;
    }

    qsort(neighbors, count, sizeof(double *), ddcompare);

    for (i = 0; i < k; i++)
        knn[i] = (struct ct_node *) nodes2[(int) neighbors[i][1]];

    return knn;
}


/*
 * MEMORY-RELATED FUNCTIONS
 */
void CoverTree::free_ct_node_set(struct ct_node_set *o) {
    free(o->s);
    free(o);
}

struct ct_node_set *CoverTree::new_ct_node_set(struct ct_node_set *o) {
    o = Malloc(struct ct_node_set, 1);
    if (o == NULL) MallocError("o");
    o->s = Malloc(struct ct_node*, FREED_NOD_INCR);
    if (o->s == NULL) MallocError("o->s");
    o->n = 0;
    return o;
}

void CoverTree::add_to_ct_node_set(struct ct_node_set *s, struct ct_node *n) {
    if (s == NULL) printf("ERROR\n");

    if (s->n % FREED_NOD_INCR == 0) {
        s->s = Realloc(s->s, struct ct_node*, s->n + FREED_NOD_INCR);
        if (s->s == NULL) ReallocError("s->s");
    }

    s->s[s->n] = n;
    s->n++;
}

void CoverTree::rem_from_ct_node_set(struct ct_node_set *s, int i) {
    if (s->n <= 0) return;

    s->s[i] = s->s[s->n - 1];
    s->n--;


    if (s->n % FREED_NOD_INCR == 0 && s->n > 0) {
        s->s = Realloc(s->s, struct ct_node*, s->n);
        if (s->s == NULL) ReallocError("s->s");
    }

}

void CoverTree::allocNodes() {
    if (alloc_nodes - used_nodes > 0) return;

    nodes = Realloc(nodes, struct ct_node*, alloc_nodes + incr_n_nodes);
    if (nodes == NULL) ReallocError("nodes");

    for (int i = alloc_nodes; i < alloc_nodes + incr_n_nodes; i++) {
        nodes[i] = Malloc(struct ct_node, 1);
        if (nodes[i] == NULL) MallocError("nodes[i]");
        nodes[i]->x = Malloc(struct node *, 1);
        if (nodes[i]->x == NULL) MallocError("nodes[i]->x");
        nodes[i]->x[0] = NULL;
        nodes[i]->ind = Malloc(int, 1);
        if (nodes[i]->ind == NULL) MallocError("nodes[i]->ind");
        nodes[i]->ind[0] = -1;
        nodes[i]->children = NULL;
        nodes[i]->card = 1;
        nodes[i]->n_children = 0;
        nodes[i]->tmp_dist_node_c = Malloc(struct node*, LCACHE);
        if (nodes[i]->tmp_dist_node_c == NULL) MallocError("nodes[i]->tmp_dist_node_c");
        for (int j = 0; j < LCACHE; j++)
            nodes[i]->tmp_dist_node_c[j] = NULL;

        nodes[i]->tmp_dist_c = Malloc(double, LCACHE);
        if (nodes[i]->tmp_dist_c == NULL) MallocError("nodes[i]->tmp_dist_c");
        nodes[i]->last = LCACHE - 1;
    }

    alloc_nodes = alloc_nodes + incr_n_nodes;
}

struct ct_node *CoverTree::clone_node(struct ct_node *node) {
    struct ct_node *svt;

    if (used_nodes == alloc_nodes) allocNodes();
    svt = nodes[used_nodes++];

    svt->card = node->card;
    if (svt->card > 1) {
        svt->x = Realloc(svt->x, struct node*, svt->card);
        if (svt->x == NULL) ReallocError("svt->x");
        svt->ind = Realloc(svt->ind, int, svt->card);
        if (svt->ind == NULL) ReallocError("svt->ind");
    }

    for (int in = 0; in < svt->card; in++) {
        svt->x[in] = node->x[in];
        svt->ind[in] = node->ind[in];
    }
    svt->n_children = 1;
    svt->children = Malloc(struct ct_node*, DEF_SON_N);
    if (svt->children == NULL) MallocError("svt->children"); // ??????????????
    svt->children[0] = svt;
    return svt;
}

struct ct_node *CoverTree::get_node(struct node *node, int index) {
    struct ct_node *svt;

    if (used_nodes == alloc_nodes) allocNodes();
    svt = nodes[used_nodes++];


    svt->card = 1;
    svt->x[0] = (struct node *) node;
    svt->ind[0] = index;
    svt->n_children = 1;
    svt->dist_from_f = -1.0;
    svt->children = Malloc(struct ct_node*, DEF_SON_N);
    if (svt->children == NULL) MallocError("svt->children"); // allocare i figli prima??????
    svt->children[0] = svt;
    return svt;
}

struct ct_node *CoverTree::get_node(struct node *node, int index, int n_children, double dist_from_f, bool impl) {
    struct ct_node *svt;

    if (used_nodes == alloc_nodes) allocNodes();
    svt = nodes[used_nodes++];


    svt->card = 1;
    svt->x[0] = (struct node *) node;
    svt->ind[0] = index;
    svt->n_children = n_children;
    svt->dist_from_f = dist_from_f;
    svt->children = Malloc(struct ct_node*, svt->n_children);
    if (svt->children == NULL) MallocError("svt->children"); // allocare i figli prima??????
    if (impl) svt->children[0] = svt;
    else svt->children[0] = NULL;
    return svt;
}


struct ct_node *
CoverTree::get_node(struct node **x, int card, int *indx, int n_children, double dist_from_f, bool impl) {
    struct ct_node *svt;

    if (used_nodes == alloc_nodes) allocNodes();
    svt = nodes[used_nodes++];

    svt->card = card;

    svt->x = Realloc(svt->x, struct node *, card);
    svt->ind = Realloc(svt->ind, int, card);

    for (int i = 0; i < card; i++) {
        svt->ind[i] = indx[i];
        svt->x[i] = (struct node *) x[indx[i]];
    }

    svt->n_children = n_children;
    svt->dist_from_f = dist_from_f;
    svt->children = Malloc(struct ct_node*, svt->n_children);
    if (svt->children == NULL) MallocError("svt->children"); // allocare i figli prima??????
    if (impl) svt->children[0] = svt;
    else svt->children[0] = NULL;
    return svt;
}


void CoverTree::addChildren(struct ct_node *f, struct ct_node *s) {
    if (f->n_children % DEF_SON_N == 0) {
        f->children = Realloc(f->children, struct ct_node*, ((f->n_children / DEF_SON_N) + 1) * DEF_SON_N);
        if (f->children == NULL) ReallocError("f->children");
    }
    f->n_children++;

    f->children[f->n_children - 1] = s;
}


/*
 * GENERAL PURPOSE SUPPORTING FUNCTIONS
 */
inline int ddcompare(const void *e1, const void *e2) {
    /* register */ double **pa1 = (double **) e1;
    /* register */ double **pa2 = (double **) e2;
    if (*pa1[0] < *pa2[0]) return -1;
    if (*pa1[0] > *pa2[0]) return 1;
    return 0;
}

inline int dcompare(const void *e1, const void *e2) {
    /* register */ double pa1 = *((double *) e1);
    /* register */ double pa2 = *((double *) e2);
    if (pa1 < pa2) return -1;
    if (pa1 > pa2) return 1;
    return 0;
}

inline int mycomp(const void *e1, const void *e2) {
    /* register */ double pa1 = (double) ***(double ***) e1;
    /* register */ double pa2 = (double) ***(double ***) e2;

    if (pa1 < pa2) return -1;
    if (pa1 > pa2) return 1;

    return 0;
}


double kth_smallest(double *a[2], int n, int k) {
    /* register */ int i, j, l, m;
    /* register */ double x;

    l = 0;
    m = n - 1;

    while (l < m) {
        x = a[k][0];
        i = l;
        j = m;
        do {
            while (a[i][0] < x) i++;
            while (x < a[j][0]) j--;
            if (i <= j) {
                ELEM_SWAP(a[i][0], a[j][0]);
                i++;
                j--;
            }
        } while (i <= j);
        if (j < k) l = i;
        if (k < i) m = j;
    }
    return a[k][0];
}

#define MAX_LIN 1024

void CoverTree::load(char *filename, PyObject *python_svm_class) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) return;

    char *line = Malloc(char, MAX_LIN);

    int min_l, max_l;
    int pl;
    double dou;

    if (sscanf(line = fgets(line, MAX_LIN, fp), "COVER TREE version %lf", &dou) < 1) LoadError(filename, line);
    if (CT_VERSION != dou) LoadErrorP(filename, line, "Wrong cover type version");

    if (sscanf(line = fgets(line, MAX_LIN, fp), "PEN %d", &PEN) < 1) LoadError(filename, line);
    set_pows(PEN);
    if (sscanf(line = fgets(line, MAX_LIN, fp), "n_points %d", &pl) < 1) LoadError(filename, line);
    if (sscanf(line = fgets(line, MAX_LIN, fp), "rt_min_lev %d", &min_l) < 1) LoadError(filename, line);
    if (sscanf(line = fgets(line, MAX_LIN, fp), "rt_max_lev %d", &max_l) < 1) LoadError(filename, line);
    if (sscanf(line = fgets(line, MAX_LIN, fp), "n_local_probs %d", &n_local_probs) < 1) LoadError(filename, line);

    param = Malloc(struct svm_parameter, 1);

    if (sscanf(line = fgets(line, MAX_LIN, fp), "K %d", &k) < 1) LoadError(filename, line);
    if (sscanf(line = fgets(line, MAX_LIN, fp), "KP %d", &kp) < 1) LoadError(filename, line);

    int tmp_python_svm;
    if (sscanf(line = fgets(line, MAX_LIN, fp), "python_svm %d", &tmp_python_svm) < 1) LoadError(filename, line);
    param->python_svm = (tmp_python_svm == 1);

    if (sscanf(line = fgets(line, MAX_LIN, fp), "svm_type %d", &param->svm_type) < 1) LoadError(filename, line);
    if (sscanf(line = fgets(line, MAX_LIN, fp), "kernel_type %d", &param->kernel_type) < 1) LoadError(filename, line);

    if (param->kernel_type == POLY)
        if (sscanf(line = fgets(line, MAX_LIN, fp), "degree %d", &param->degree) < 1) LoadError(filename, line);

    if (param->kernel_type == POLY || param->kernel_type == RBF || param->kernel_type == SIGMOID)
        if (sscanf(line = fgets(line, MAX_LIN, fp), "gamma %lf", &param->gamma) < 1) LoadError(filename, line);

    if (param->kernel_type == POLY || param->kernel_type == SIGMOID)
        if (sscanf(line = fgets(line, MAX_LIN, fp), "coef0 %lf", &param->coef0) < 1) LoadError(filename, line);

    param->cache_size = -1;
    param->eps = -1;
    param->C = -1;
    param->nr_weight = -1;
    param->weight_label = NULL;
    param->weight = NULL;
    param->nu = -1;
    param->p = -1;
    param->shrinking = -1;
    param->probability = -1;
    param->h_warning = false;

    param->python_svm_class = python_svm_class;
    if (sscanf(line = fgets(line, MAX_LIN, fp), "python_svm_type %d", &param->python_svm_type) < 1) LoadError(filename, line);
    if (sscanf(line = fgets(line, MAX_LIN, fp), "python_svm_base %d", &param->python_svm_base) < 1) LoadError(filename, line);
    if (sscanf(line = fgets(line, MAX_LIN, fp), "python_svm_binary_vars_num %d", &param->python_svm_binary_vars_num) < 1) LoadError(filename, line);
    if (sscanf(line = fgets(line, MAX_LIN, fp), "python_svm_penalty_coef %lf", &param->python_svm_penalty_coef) < 1) LoadError(filename, line);
    if (sscanf(line = fgets(line, MAX_LIN, fp), "python_svm_multicl_reg %lf", &param->python_svm_multicl_reg) < 1) LoadError(filename, line);

    line = fgets(line, MAX_LIN, fp);
    param->python_svm_embeddings_dir = Malloc(char, MAX_LIN);
    if (sscanf(line, "python_svm_embeddings_dir %s", param->python_svm_embeddings_dir) < 1) {
        free(param->python_svm_embeddings_dir);
        param->python_svm_embeddings_dir = nullptr;
    } else {
        param->python_svm_embeddings_dir = Realloc(param->python_svm_embeddings_dir, char, strlen(param->python_svm_embeddings_dir) + 1);
    }

    l_models = Malloc(struct svm_model *, n_local_probs);
    for (int i = 0; i < n_local_probs; i++)
        l_models[i] = NULL;

    if (tree == NULL) {
        tree = Malloc(struct ct, 1);
        if (this->tree == NULL) MallocError("tree");
        tree->prob = Malloc(struct problem, 1);
        if (tree->prob == NULL) MallocError("tree->prob");
        tree->root = NULL;
        tree->tmp_root = NULL;
        tree->prob->l = pl;
        if (red_tree) tree->prob->l = n_local_probs;
        tree->prob->x = Malloc(struct node *, tree->prob->l);
        if (tree->prob->x == NULL) MallocError("prob->x");
        tree->prob->y = Malloc(double, tree->prob->l);
        if (tree->prob->y == NULL) MallocError("prob->y");
        tree->prob->k_rank = Malloc(int, tree->prob->l);
        if (tree->prob->k_rank == NULL) MallocError("prob->y");
        tree->prob->lpm_ind = Malloc(int, pl);
        if (tree->prob->lpm_ind == NULL) MallocError("tree->prob->lpm_ind");
        tree->param = this->param;
        tree->rt_min_lev = min_l;
        tree->rt_max_lev = max_l;
    }

    int n_c = 0;

    pr_prob = Malloc(struct problem, 1);
    if (pr_prob == NULL) MallocError("pr_prob");
    pr_prob->l = pl;
    pr_prob->x = Malloc(struct node *, pr_prob->l);
    if (pr_prob->x == NULL) MallocError("pr_prob->x");
    pr_prob->y = Malloc(double, pr_prob->l);
    if (pr_prob->y == NULL) MallocError("pr_prob->y");
    pr_prob->k_rank = Malloc(int, pr_prob->l);
    if (pr_prob->k_rank == NULL) MallocError("pr_prob->y");
    pr_prob->lpm_ind = Malloc(int, pr_prob->l);
    if (pr_prob->lpm_ind == NULL) MallocError("pr_prob->lpm_ind");


    for (int i = 0; i < pl; i++) {
        int nn_nod;
        if (fscanf(fp, "%lf", &pr_prob->y[i]) < 1) {
            printf("Error in loading Cover Tree\n");
            exit(0);
        }
        if (fscanf(fp, " %d", &pr_prob->k_rank[i]) < 1) {
            printf("Error in loading Cover Tree\n");
            exit(0);
        }
        if (fscanf(fp, " %d", &pr_prob->lpm_ind[i]) < 1) {
            printf("Error in loading Cover Tree\n");
            exit(0);
        }
        if (fscanf(fp, " %d", &nn_nod) < 1) {
            printf("Error in loading Cover Tree\n");
            exit(0);
        }

        pr_prob->x[i] = Malloc(struct node, nn_nod);
        if (pr_prob->x[i] == NULL) MallocError("pr_prob->x[i]");

        int j = 0;
        while (1) {
            int c;
            do {
                c = getc(fp);
                if (c == '\n') goto out2;
            } while (isspace(c));
            ungetc(c, fp);
            if (fscanf(fp, "%d:%lf", &(pr_prob->x[i][j].index), &(pr_prob->x[i][j].value)) < 2) {
                fprintf(stderr, "Wrong input format at line %d\n", i + 1);
                exit(1);
            }
            ++j;
        }
        out2:;

        if (red_tree) {
            if (pr_prob->k_rank[i] == 0) {
                tree->prob->x[n_c] = pr_prob->x[i];
                tree->prob->y[n_c] = pr_prob->y[i];
                tree->prob->k_rank[n_c] = 0;
                n_c++;
            }
        } else {
            tree->prob->x[i] = pr_prob->x[i];
            tree->prob->y[i] = pr_prob->y[i];
            tree->prob->k_rank[i] = pr_prob->k_rank[i];
            tree->prob->lpm_ind[i] = pr_prob->lpm_ind[i];
        }
        tree->prob->lpm_ind[i] = pr_prob->lpm_ind[i];
    }

    if (!silent) printf("N_LOCAL_PROBS %d, NC %d\n", n_local_probs, n_c);
    if (red_tree && n_local_probs != n_c) {
        tree->prob->l = n_c;
        tree->prob->x = Realloc(tree->prob->x, struct node *, tree->prob->l);
        if (tree->prob->x == NULL) ReallocError("prob->x");
        tree->prob->y = Realloc(tree->prob->y, double, tree->prob->l);
        if (tree->prob->y == NULL) ReallocError("prob->y");
        tree->prob->k_rank = Realloc(tree->prob->k_rank, int, tree->prob->l);
        if (tree->prob->k_rank == NULL) ReallocError("prob->y");
        tree->prob->lpm_ind = Realloc(tree->prob->lpm_ind, int, pl);
        if (tree->prob->lpm_ind == NULL) ReallocError("tree->prob->lpm_ind");
    }

    int level, np_lev;
    int card, ind, *inds = NULL, n_ch;
    double dff;

    struct ct_node_set *old = NULL;
    old = new_ct_node_set(old);

    if (!red_tree)
        for (int l = tree->rt_max_lev; l >= tree->rt_min_lev; l--) {
            struct ct_node_set *cur = NULL;
            cur = new_ct_node_set(cur);

            while (sscanf(line = fgets(line, MAX_LIN, fp), " LEVEL %d - %d", &level, &np_lev) < 2);

            for (int i = 0; i < np_lev; i++) {
                if (fscanf(fp, "%d ", &card) < 1) LoadError(filename, line);

                if (card == 1) {
                    if (fscanf(fp, "%d ", &ind) < 1) {
                        printf("Error in loading Cover Tree\n");
                        exit(0);
                    }
                } else {
                    inds = Realloc(inds, int, card);
                    for (int h = 0; h < card; h++)
                        if (fscanf(fp, "%d ", &inds[h]) < 1) LoadError(filename, line);
                }

                if (fscanf(fp, "%d %lf\n", &n_ch, &dff) < 2) LoadError(filename, line);

                bool impl = false;
                if (n_ch == -1) {
                    impl = true;
                    n_ch = 1;
                }

                struct ct_node *tnode = NULL;

                if (card == 1) tnode = get_node(tree->prob->x[ind], ind, n_ch, dff, impl);
                else tnode = get_node(tree->prob->x, card, inds, n_ch, dff, impl);
                add_to_ct_node_set(cur, tnode);
            }


            int n_c = 0;
            for (int i = 0; i < old->n; i++) {
                struct ct_node *node = old->s[i];

                if (isImplicit(node)) continue;

                for (int j = 0; j < node->n_children; j++)
                    node->children[j] = cur->s[n_c++];
            }

            if (l == tree->rt_max_lev) {
                tree->tmp_root = new_ct_node_set(tree->tmp_root);
                add_to_ct_node_set(tree->tmp_root, cur->s[0]);
            }

            free_ct_node_set(old);
            old = cur;
        }
    else {
        struct ct_node_set *all = NULL;
        struct ct_node_set *v = NULL;

        all = new_ct_node_set(all);
        v = new_ct_node_set(v);

        tree->tmp_root = Malloc(struct ct_node_set, 1);
        if (tree->tmp_root == NULL) MallocError("tree->tmp_root");
        tree->root = Malloc(struct ct_node_set, 1);
        if (tree->root == NULL) MallocError("tree->tmp_root");
        tree->tmp_root->n = 1;
        tree->tmp_root->s = Malloc(struct ct_node *, tree->tmp_root->n);
        if (tree->tmp_root->s == NULL) MallocError("tree->tmp_root->s");
        tree->root->n = 1;
        tree->root->s = Malloc(struct ct_node *, tree->root->n);
        if (tree->root->s == NULL) MallocError("tree->root->s");

        struct ct_node *first = get_node(tree->prob->x[0], 0);

        for (int i = 1; i < tree->prob->l; i++) {
            struct ct_node *n = get_node(tree->prob->x[i], i);
            add_to_ct_node_set(all, n);
        }

        n_nodes++;

        tree->rt_min_lev = LEV_MAXX;
        tree->rt_max_lev = -LEV_MINN;

        tree->root->s[0] = construct(first, all, v, LEV_MAXX, tree);
        if (!silent) printf("allocated nodes %d\n", alloc_nodes);
    }

    free_ct_node_set(old);

    while (strcmp(line = fgets(line, MAX_LIN, fp), "MODELS\n") != 0);

    long filename_last_slash_index = -1;
    char *python_svm_filepath = nullptr;
    char *python_svm_filename = nullptr;
    if (param->python_svm) {
        const char *filename_last_slash = strrchr(filename, '/');
        if (filename_last_slash != nullptr) {
            filename_last_slash_index = filename_last_slash - filename;
        }

        python_svm_filepath = Malloc(char, strlen(filename) + MAX_LIN);
        strncpy(python_svm_filepath, filename, filename_last_slash_index + 1);

        python_svm_filename = Malloc(char, MAX_LIN);
    }

    for (int i = 0; i < n_local_probs; i++) {
        if (param->python_svm) {
            python_svm_filepath[filename_last_slash_index + 1] = '\0';
            if (sscanf(line = fgets(line, MAX_LIN, fp), "%s", python_svm_filename) < 1) LoadError(filename, line);
            strcat(python_svm_filepath, python_svm_filename);

            l_models[i] = python_svm_load_model(python_svm_filepath, param);

            line = fgets(line, MAX_LIN, fp);
        } else {
            l_models[i] = svm_load_model2(fp, pr_prob, param);
        }
    }

    if (param->python_svm) {
        free(python_svm_filepath);
        free(python_svm_filename);
    }


    if (inds != NULL) free(inds);


    fclose(fp);

    Q->s = Realloc(Q->s, struct ct_node*, tree->prob->l);
    if (Q->s == NULL) ReallocError("Q->s"); // non ottimizzato per aggiunta punti !!!!!!!
    Qim1->s = Realloc(Qim1->s, struct ct_node*, tree->prob->l);
    if (Qim1->s == NULL) ReallocError("Qim1->s"); // non ottimizzato per aggiunta punti !!!!!!!
    neighbors = Realloc(neighbors, double*, tree->prob->l);
    if (neighbors == NULL) ReallocError("neighbors");
    nodes2 = Realloc(nodes2, void*, tree->prob->l);
    if (nodes2 == NULL) ReallocError("nodes2");
    for (int i = 0; i < tree->prob->l; i++) {
        neighbors[i] = Malloc(double, 2);
        if (neighbors[i] == NULL) MallocError("neighbors[i]");
    }

    ddda = NULL;
    ddda = Realloc(ddda, double, tree->prob->l);
    if (ddda == NULL) ReallocError("Q->s");
    free(line);

}

void CoverTree::save(char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) return;

    fprintf(fp, "COVER TREE version %g\n", CT_VERSION);
    fprintf(fp, "PEN %d\n", PEN);
    fprintf(fp, "n_points %d\n", tree->prob->l);
    fprintf(fp, "rt_min_lev %d\n", tree->rt_min_lev);
    fprintf(fp, "rt_max_lev %d\n", tree->rt_max_lev);
    fprintf(fp, "n_local_probs %d\n", n_local_probs);

    fprintf(fp, "K %d\n", k);
    fprintf(fp, "KP %d\n", kp);

    fprintf(fp, "python_svm %d\n", param->python_svm);

    fprintf(fp, "svm_type %d\n", param->svm_type);
    fprintf(fp, "kernel_type %d\n", param->kernel_type);

    if (param->kernel_type == POLY)
        fprintf(fp, "degree %d\n", param->degree);

    if (param->kernel_type == POLY || param->kernel_type == RBF || param->kernel_type == SIGMOID)
        fprintf(fp, "gamma %g\n", param->gamma);

    if (param->kernel_type == POLY || param->kernel_type == SIGMOID)
        fprintf(fp, "coef0 %g\n", param->coef0);

    fprintf(fp, "python_svm_type %d\n", param->python_svm_type);
    fprintf(fp, "python_svm_base %d\n", param->python_svm_base);
    fprintf(fp, "python_svm_binary_vars_num %d\n", param->python_svm_binary_vars_num);
    fprintf(fp, "python_svm_penalty_coef %.2f\n", param->python_svm_penalty_coef);
    fprintf(fp, "python_svm_multicl_reg %.2f\n", param->python_svm_multicl_reg);
    fprintf(fp, "python_svm_embeddings_dir %s\n", param->python_svm_embeddings_dir);

    char *line = Malloc(char, 100000);
    char *line2 = Malloc(char, 100000);

    int zeros = 0;
    for (int i = 0; i < tree->prob->l; i++) {
        //if ( tree->prob->k_rank[i] == 0 ) zeros++;
        fprintf(fp, "%g %d %d ", tree->prob->y[i], tree->prob->k_rank[i], tree->prob->lpm_ind[i]);
        struct node *px = tree->prob->x[i];
        int nn_nod = 0;
        //sprintf(line,"");
        line[0] = '\0';
        line2[0] = '\0';
        while (px->index > 0) {
            sprintf(line2, "%s%d:%.8g ", line, px->index, px->value);
            sprintf(line, "%s", line2);
            nn_nod++;
            ++px;
        }
        sprintf(line2, "%s%d:%.8g ", line, px->index, px->value);
        sprintf(line, "%s", line2);
        fprintf(fp, "%d %s\n", nn_nod + 1, line);

    }

    //printf("ZEROS %d\n",zeros);

    fprintf(fp, "\n");

    free(line);
    free(line2);

    struct ct_node_set *o = NULL;
    o = new_ct_node_set(o);

    add_to_ct_node_set(o, tree->tmp_root->s[0]);

    struct ct_node_set *l = NULL;


    for (int i = tree->rt_max_lev; i >= tree->rt_min_lev; i--) {
        l = new_ct_node_set(l);

        fprintf(fp, "\n LEVEL %d - %d\n", i, o->n);
        for (int j = 0; j < o->n; j++) {
            struct ct_node *Qisj = o->s[j];

            fprintf(fp, "%d ", Qisj->card);

            for (int h = 0; h < Qisj->card; h++)
                fprintf(fp, "%d ", Qisj->ind[h]);

            if (isImplicit(Qisj)) fprintf(fp, "-1 ");
            else {
                fprintf(fp, "%d ", Qisj->n_children);
                for (int h = 0; h < Qisj->n_children; h++)
                    add_to_ct_node_set(l, Qisj->children[h]);
            }

            fprintf(fp, "%.8g\n", Qisj->dist_from_f);
        }

        fprintf(fp, "\n");


        free_ct_node_set(o);
        o = l;
    }

    free_ct_node_set(l);

    fprintf(fp, "MODELS\n");

    for (int i = 0; i < n_local_probs; i++) {
        if (param->python_svm) {
            const char *relative_filepath = python_svm_get_relative_filepath(l_models[i]);
            fprintf(fp, "%s\n", relative_filepath);
        } else {
            svm_save_model2(fp, l_models[i]);
        }
        if (i != n_local_probs - 1) fprintf(fp, "\n");
    }

    fclose(fp);
}




