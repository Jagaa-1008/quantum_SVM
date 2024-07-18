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

#include "par_est.h"

#include "svm.h"
#include "per_eval.h"


#define MAX_DISTS 70

inline double ParEst::dist_dot(const node *px, const node *py) {

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
    while (px->index > 0)
        ++px;
    while (py->index > 0)
        ++py;
    return sqrt(px->value + py->value - 2.0 * sum);
}

inline int dcompare(const void *e1, const void *e2) {
    /* register */ double pa1 = *((double *) e1);
    /* register */ double pa2 = *((double *) e2);
    if (pa1 < pa2)
        return -1;
    if (pa1 > pa2)
        return 1;
    return 0;
}

ParEst::ParEst(int opt, int n_folds) {
    dists = Malloc(double, MAX_DISTS * MAX_DISTS);
    this->opt = opt;
    this->n_folds = n_folds;
}

ParEst::~ParEst() {
    if (dists != NULL)
        free(dists);
}

#define MAX_G_VALUE 1048576.0
#define CORR_MIN_D_VALUE 0.000695

double ParEst::estimateGammaQ(struct problem *p, double quant) {
    int max_dists = MAX_DISTS;
    if (p->l < max_dists)
        max_dists = p->l;

    int incr = p->l / max_dists;

    int n = 0;
    for (int i = 0; i < p->l; i += incr)
        for (int j = i; j < p->l; j += incr)
            if (i != j && n < MAX_DISTS * MAX_DISTS)
                dists[n++] = dist_dot(p->x[i], p->x[j]);

    qsort(dists, n, sizeof(double), dcompare);

    int q = (int) ((double) n * quant);

    if (dists[q] < CORR_MIN_D_VALUE)
        return MAX_G_VALUE;
    return 1.0 / (2.0 * dists[q] * dists[q]);
}

double ParEst::estimateGamma(struct problem *p) {
    return estimateGammaQ(p, 0.5);
}

double ParEst::estG_quant(struct problem *p, double quant) {
    return estimateGammaQ(p, quant);
}

double ParEst::estC_int(struct problem *pb,
                        struct svm_parameter *par,
                        double *Cs, int n_cs,
                        double *res,
                        int kp, int k, int val_k,
                        int model_index, FILE *log_fp) {
    double *target = Malloc(double, kp);
    struct problem *lp_int = get_lp_int(pb, kp);
    struct problem *lp_ext = get_lp_ext(pb, kp, k);
;
    char *line = nullptr;
    if (log_fp != nullptr) {
        line = Malloc(char, 1024);
        line = fgets(line, 1024, log_fp);
    }

    PerEval *perEval = new PerEval(lp_int->y, val_k);
    for (int i = 0; i < n_cs; i++) {
        par->C = Cs[i];
        bool rip = true;
        if (svm_print_string != &print_null) {
            svm_print_string = &print_null;
        } else
            rip = false;

        double performance_measure = 0.0;
        if (line == nullptr) {
            svm_cross_validation_int(lp_int, lp_ext, par, n_folds, target);
            if (rip)
                svm_print_string = &print_string_stdout;
            performance_measure = perEval->calcAll(target, opt, false);
            res[i] += performance_measure;

            if (log_fp != nullptr) {
                fprintf(log_fp, "%.5f K=%d idx=%d b=%d a=%d i=%.2f e=%.2f g=%.2f c=%.2f\n",
                        performance_measure, pb->l, model_index, par->python_svm_base, par->python_svm_binary_vars_num,
                        par->python_svm_penalty_coef, par->python_svm_multicl_reg, par->gamma, par->C);
            }
        } else {
            for (int j = 0; j < lp_int->l; j++) {
                rand();
            }

            if (sscanf(line, "%lf ", &performance_measure) < 1) LoadError("Local model selection log file", line);
            res[i] += performance_measure;

            if (i != (n_cs -1)) {
                line = fgets(line, 1024, log_fp);
            }
        }
    }

    delete perEval;
    free(line);

    free(target);

    free(lp_int->x);
    free(lp_int->y);
    free(lp_int);

    free(lp_ext->x);
    free(lp_ext->y);
    free(lp_ext);

    return 0.0;
}

double ParEst::estCG_int(struct problem *pb,
                         struct lkm_parameter *par,
                         double *Cs, int n_cs,
                         double *Gs, int n_gs,
                         double **res,
                         int kp, int k, int val_k) {
    for (int j = 0; j < n_gs; j++) {
        par->svm_par->gamma = Gs[j];
        estC_int(pb, par->svm_par, Cs, n_cs, res[j], kp, k, val_k, -1, nullptr);
    }

    return -1.0;
}

double ParEst::estCGK_int(struct problem *pb,
                          struct lkm_parameter *par,
                          double *Cs, int n_cs,
                          double *Gs, int n_gs,
                          int *Ks, int n_ks, int *KPs, int n_kps, int val_k,
                          double ***res) {
    for (int k = 0; k < n_ks; k++) {
        pb->l = Ks[k];
        for (int j = 0; j < n_gs; j++)
            if (Gs[j] < 0.0)
                Gs[j] = estimateGammaQ(pb, -par->svm_par->gamma);
        estCG_int(pb, par, Cs, n_cs, Gs, n_gs, res[k], KPs[k], Ks[k], val_k);
    }
    return 1.0;
}

double ParEst::estC(struct problem **pb, int nms, struct lkm_parameter *par, double *Cs, int n_cs, int kp, int k) {
    printf("*********************************************************\n");
    printf("*** L1 MODEL SELECTION ON C *****************************\n");

    double *res = Malloc(double, n_cs);
    for (int i = 0; i < n_cs; i++)
        res[i] = 0.0;

    double gs = par->svm_par->gamma;

    for (int i = 0; i < nms; i++) {
        if (gs < 0.0 && par->svm_par->kernel_type == RBF)
            par->svm_par->gamma = estimateGammaQ(pb[i], -gs);
        estC_int(pb[i], par->svm_par, Cs, n_cs, res, kp, k, kp, -1, nullptr);
    }

    for (int i = 0; i < n_cs; i++)
        res[i] = res[i] / (double) nms * 100.0;

    select_best_C(res, Cs, n_cs, par);

    free(res);

    par->svm_par->gamma = gs;

    printf("*** BEST C %f\n", par->svm_par->C);
    printf("*** END L1 MODEL SELECTION*******************************\n\n");

    return par->svm_par->C;
}

double ParEst::estCG(struct problem **pb, int nms, struct lkm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs,
              int kp, int k) {
    printf("*********************************************************\n");
    printf("*** L7 MODEL SELECTION ON C and G ***********************\n");

    //bool var = false;

    //gs = par->svm_par->gamma;

    double **res = Malloc(double*, n_gs);
    for (int g = 0; g < n_gs; g++) {
        res[g] = Malloc(double, n_cs);
        for (int c = 0; c < n_cs; c++)
            res[g][c] = 0.0;
    }


    for (int i = 0; i < nms; i++) {
        printf("%d local model for model selection [tot = %d] +++++++++++++++++++++++++++++++++++++\n", i + 1, nms);
        int KK = k;
        int KP = kp;
        int KV = kp;

        for (int g = 0; g < n_gs; g++) {
            pb[i]->l = KK;
            if (Gs[g] < 0.0)
                par->svm_par->gamma = estimateGammaQ(pb[i], -Gs[g]);
            else
                par->svm_par->gamma = Gs[g];
            estC_int(pb[i], par->svm_par, Cs, n_cs, res[g], KP, KK, KV, -1, nullptr);
        }

    }

    for (int g = 0; g < n_gs; g++)
        for (int c = 0; c < n_cs; c++)
            res[g][c] /= (double) (nms);

    select_best_CG(res, Cs, n_cs, Gs, n_gs, par->svm_par);

    for (int g = 0; g < n_gs; g++)
        free(res[g]);
    free(res);


    printf("*** BEST C %f\n", par->svm_par->C);
    printf("*** BEST G %f\n", par->svm_par->gamma);
    printf("*** END L7 MODEL SELECTION *******************************\n\n");

    return par->svm_par->C;
}

double ParEst::estCG(struct problem *pb, struct svm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs, int kp,
                     int k) {
    printf("*********************************************************\n");
    printf("*** L7 MODEL SELECTION ON C and G ***********************\n");


    double **res = Malloc(double*, n_gs);
    for (int g = 0; g < n_gs; g++) {
        res[g] = Malloc(double, n_cs);
        for (int c = 0; c < n_cs; c++)
            res[g][c] = 0.0;
    }

    int KK = k;
    int KP = kp;
    int KV = kp / 3 * 2;

    for (int g = 0; g < n_gs; g++) {
        pb->l = KK;
        if (Gs[g] < 0.0)
            par->gamma = estimateGammaQ(pb, -Gs[g]);
        else
            par->gamma = Gs[g];
        estC_int(pb, par, Cs, n_cs, res[g], KP, KK, KV, -1, nullptr);
    }


    select_best_CG(res, Cs, n_cs, Gs, n_gs, par);

    for (int g = 0; g < n_gs; g++)
        free(res[g]);
    free(res);


    printf("*** BEST C %f\n", par->C);
    printf("*** BEST G %f\n", par->gamma);
    printf("*** END L7 MODEL SELECTION *******************************\n\n");

    return par->C;

}

double ParEst::estCGK(struct problem **pb, int nms, struct lkm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs,
               int *Ks, int n_ks, int *KPs, int n_kps, FILE *log_fp) {
    if (!par->silent)
        printf("\n*********************************************************\n");
    if (!par->silent)
        printf("*** LOCAL MODEL SELECTION ON C and G and K **************");

    double ***res = Malloc(double**, n_ks);
    for (int k = 0; k < n_ks; k++) {
        res[k] = Malloc(double*, n_gs);
        for (int g = 0; g < n_gs; g++) {
            res[k][g] = Malloc(double, n_cs);
            for (int c = 0; c < n_cs; c++)
                res[k][g][c] = 0.0;
        }
    }

    double last_best_k = -1.0;
    for (int k = 0; k < n_ks; k++) {
        int KK = Ks[k];
        int KP = KPs[k];
        int KV = KPs[k] / 2;
        if (par->nr_strategy == FALKNR || par->eval_on_kp)
            KV = KPs[k];

        for (int i = 0; i < nms; i++)
            for (int g = 0; g < n_gs; g++) {
                pb[i]->l = KK;
                if (Gs[g] < 0.0)
                    par->svm_par->gamma = estimateGammaQ(pb[i], -Gs[g]);
                else
                    par->svm_par->gamma = Gs[g];
                estC_int(pb[i], par->svm_par, Cs, n_cs, res[k][g], KP, KK, KV, i, log_fp);
            }

        double best_k = -1.0;
        for (int g = 0; g < n_gs; g++)
            for (int c = 0; c < n_cs; c++) {
                res[k][g][c] /= (double) (nms);
                if (best_k < res[k][g][c])
                    best_k = res[k][g][c];
            }
        if (last_best_k > best_k) {
            n_ks = MIN(k + 1, n_ks);
            break;
        }
        last_best_k = best_k;
    }

    select_best_CGK(res, Cs, n_cs, Gs, n_gs, Ks, n_ks, KPs, n_kps, par);

    for (int k = 0; k < n_ks; k++) {
        for (int g = 0; g < n_gs; g++)
            free(res[k][g]);
        free(res[k]);
    }
    free(res);


    if (!par->silent)
        printf("*** BEST C %.2f\n", par->svm_par->C);
    if (!par->silent)
        printf("*** BEST G %.2f\n", par->svm_par->gamma);
    if (!par->silent)
        printf("*** BEST K %d (KP %d)\n", par->k, par->kp);
    if (!par->silent) {
        printf("*** LOCAL MODEL SELECTION ON C and G and K **************\n");
        printf("*********************************************************\n");
    }

    return par->svm_par->C;
}

double ParEst::estCGEIABK(struct problem **pb, int nms, struct lkm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs,
                          int *Es, int n_es, int *Is, int n_is, int *As, int n_as, int *Bs, int n_bs, int *Ks, int n_ks, int *KPs, int n_kps,
                          FILE *log_fp) {
    if (!par->silent)
        printf("\n*********************************************************\n");
    if (!par->silent)
        printf("*** LOCAL MODEL SELECTION ON K, B, A, I, E, G, C ********");

    double *******res = Malloc(double******, n_ks);
    for (int k = 0; k < n_ks; k++) {
        res[k] = Malloc(double*****, n_bs);
        for (int b = 0; b < n_bs; b++) {
            res[k][b] = Malloc(double****, n_as);
            for (int a = 0; a < n_as; a++) {
                res[k][b][a] = Malloc(double***, n_is);
                for (int i = 0; i < n_is; i++) {
                    res[k][b][a][i] = Malloc(double**, n_es);
                    for (int e = 0; e < n_es; e++) {
                        res[k][b][a][i][e] = Malloc(double*, n_gs);
                        for (int g = 0; g < n_gs; g++) {
                            res[k][b][a][i][e][g] = Malloc(double, n_cs);
                            for (int c = 0; c < n_cs; c++)
                                res[k][b][a][i][e][g][c] = 0.0;
                        }
                    }
                }
            }
        }
    }

    double last_best_k = -1.0;
    for (int k = 0; k < n_ks; k++) {
        int KK = Ks[k];
        int KP = KPs[k];
        int KV = KPs[k] / 2;
        if (par->nr_strategy == FALKNR || par->eval_on_kp)
            KV = KPs[k];

        for (int m = 0; m < nms; m++) {
            pb[m]->l = KK;
            for (int b = 0; b < n_bs; b++) {
                par->svm_par->python_svm_base = Bs[b];
                for (int a = 0; a < n_as; a++) {
                    par->svm_par->python_svm_binary_vars_num = As[a];
                    for (int i = 0; i < n_is; i++) {
                        par->svm_par->python_svm_penalty_coef = Is[i];
                        for (int e = 0; e < n_es; e++) {
                            par->svm_par->python_svm_multicl_reg = Es[e];
                            for (int g = 0; g < n_gs; g++) {
                                if (Gs[g] < 0.0)
                                    par->svm_par->gamma = estimateGammaQ(pb[m], -Gs[g]);
                                else
                                    par->svm_par->gamma = Gs[g];
                                estC_int(pb[m], par->svm_par, Cs, n_cs, res[k][b][a][i][e][g], KP, KK, KV, m, log_fp);
                            }
                        }
                    }
                }
            }
        }

        double best_k = -1.0;
        for (int b = 0; b < n_bs; b++) {
            for (int a = 0; a < n_as; a++) {
                for (int i = 0; i < n_is; i++) {
                    for (int e = 0; e < n_es; e++) {
                        for (int g = 0; g < n_gs; g++) {
                            for (int c = 0; c < n_cs; c++) {
                                res[k][b][a][i][e][g][c] /= (double) (nms);
                                if (best_k < res[k][b][a][i][e][g][c])
                                    best_k = res[k][b][a][i][e][g][c];
                            }
                        }
                    }
                }
            }
        }
        if (last_best_k > best_k) {
            n_ks = MIN(k + 1, n_ks);
            break;
        }
        last_best_k = best_k;
    }

    select_best_CGEIABK(res, Cs, n_cs, Gs, n_gs, Es, n_es, Is, n_is, As, n_as, Bs, n_bs, Ks, n_ks, KPs, n_kps, par);

    for (int k = 0; k < n_ks; k++) {
        for (int b = 0; b < n_bs; b++) {
            for (int a = 0; a < n_as; a++) {
                for (int i = 0; i < n_is; i++) {
                    for (int e = 0; e < n_es; e++) {
                        for (int g = 0; g < n_gs; g++) {
                            free(res[k][b][a][i][e][g]);
                        }
                        free(res[k][b][a][i][e]);
                    }
                    free(res[k][b][a][i]);
                }
                free(res[k][b][a]);
            }
            free(res[k][b]);
        }
        free(res[k]);
    }
    free(res);

    if (!par->silent)
        printf("*** BEST K %d (KP %d)\n", par->k, par->kp);
    if (!par->silent)
        printf("*** BEST B %d\n", par->svm_par->python_svm_base);
    if (!par->silent)
        printf("*** BEST A %d\n", par->svm_par->python_svm_binary_vars_num);
    if (!par->silent)
        printf("*** BEST I %.2f\n", par->svm_par->python_svm_penalty_coef);
    if (!par->silent)
        printf("*** BEST E %.2f\n", par->svm_par->python_svm_multicl_reg);
    if (!par->silent)
        printf("*** BEST G %.2f\n", par->svm_par->gamma);
    if (!par->silent)
        printf("*** BEST C %.2f\n", par->svm_par->C);

    if (!par->silent) {
        printf("*** LOCAL MODEL SELECTION ON K, B, A, I, E, G, C ********\n");
        printf("*********************************************************\n");
    }

    return par->svm_par->C;
}


void ParEst::select_best_CGK(double ***res, double *Cs, int n_cs, double *Gs, int n_gs, int *Ks, int n_ks, int *KPs,
                             int n_kps, struct lkm_parameter *par) {
    double max = -1.0;
    int max_ind_c = -1;
    int max_ind_g = -1;
    int max_ind_k = -1;


    for (int k = 0; k < n_ks; k++) {
        if (!par->silent)
            printf("\nK = %d, KP = %d\n", Ks[k], KPs[k]);
        if (!par->silent)
            printf("G/C\t");
        for (int c = 0; c < n_cs; c++)
            if (!par->silent)
                printf("%.2f\t", Cs[c]);
        if (!par->silent)
            printf("\n");

        for (int g = 0; g < n_gs; g++) {
            if (!par->silent)
                printf("%.2f\t", Gs[g]);

            for (int c = 0; c < n_cs; c++) {
                if (!par->silent)
                    printf("%.2f\t", res[k][g][c] * 100.0);
                if (max < res[k][g][c]) {
                    max = res[k][g][c];
                    max_ind_c = c;
                    max_ind_g = g;
                    max_ind_k = k;
                }
            }
            if (!par->silent)
                printf("\n");
        }
        if (!par->silent)
            printf("\n");
    }

    par->svm_par->C = Cs[max_ind_c];
    par->svm_par->gamma = Gs[max_ind_g];
    par->k = Ks[max_ind_k];
    par->kp = KPs[max_ind_k];
}

void ParEst::select_best_CGEIABK(double *******res, double *Cs, int n_cs, double *Gs, int n_gs, int *Es, int n_es, int *Is, int n_is,
                                 int *As, int n_as, int *Bs, int n_bs, int *Ks, int n_ks, int *KPs, int n_kps, struct lkm_parameter *par) {
    double max = -1.0;
    int max_ind_c = -1;
    int max_ind_g = -1;
    int max_ind_e = -1;
    int max_ind_i = -1;
    int max_ind_a = -1;
    int max_ind_b = -1;
    int max_ind_k = -1;

    if (!par->silent)
        printf("\n");
    for (int k = 0; k < n_ks; k++) {
        for (int b = 0; b < n_bs; b++) {
            for (int a = 0; a < n_as; a++) {
                for (int i = 0; i < n_is; i++) {
                    for (int e = 0; e < n_es; e++) {
                        for (int g = 0; g < n_gs; g++) {
                            for (int c = 0; c < n_cs; c++) {
                                if (!par->silent){
                                    printf("[K=%d, B=%d, A=%d, I=%d, E=%d, G=%.2f, C=%.2f] => %.2f\n",
                                           Ks[k], Bs[b], As[a], Is[i], Es[e], Gs[g], Cs[c], res[k][b][a][i][e][g][c] * 100.0);
                                }

                                if (max < res[k][b][a][i][e][g][c]) {
                                    max = res[k][b][a][i][e][g][c];
                                    max_ind_c = c;
                                    max_ind_g = g;
                                    max_ind_e = e;
                                    max_ind_i = i;
                                    max_ind_a = a;
                                    max_ind_b = b;
                                    max_ind_k = k;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!par->silent)
            printf("\n");
    }

    par->svm_par->C = Cs[max_ind_c];
    par->svm_par->gamma = Gs[max_ind_g];
    par->svm_par->python_svm_multicl_reg = Es[max_ind_e];
    par->svm_par->python_svm_penalty_coef = Is[max_ind_i];
    par->svm_par->python_svm_binary_vars_num = As[max_ind_a];
    par->svm_par->python_svm_base = Bs[max_ind_b];
    par->k = Ks[max_ind_k];
    par->kp = KPs[max_ind_k];
}


void ParEst::select_best_C(double *res, double *Cs, int n_cs, struct lkm_parameter *par) {
    double max = -1.0;
    int max_ind = -1;

    for (int i = 0; i < n_cs; i++)
        printf("%f\t", Cs[i]);
    printf("\n");

    for (int i = 0; i < n_cs; i++) {
        printf("%f\t", res[i]);
        if (max < res[i]) {
            max = res[i];
            max_ind = i;
        }
    }
    printf("\n");

    for (int i = 0; i < n_cs; i++)
        if (res[i] == max) {
            par->svm_par->C = Cs[i];
            return;
        }


    par->svm_par->C = Cs[max_ind];
}


void ParEst::select_best_CG(double **res, double *Cs, int n_cs, double *Gs, int n_gs, struct svm_parameter *par) {
    double max = -1.0;
    int max_ind_c = -1;
    int max_ind_g = -1;

    printf("G/C\t");
    for (int c = 0; c < n_cs; c++)
        printf("%f\t", Cs[c]);
    printf("\n");

    for (int g = 0; g < n_gs; g++) {
        printf("%f\t", Gs[g]);

        for (int c = 0; c < n_cs; c++) {
            printf("%.2f\t", res[g][c] * 100.0);
            if (max < res[g][c]) {
                max = res[g][c];
                max_ind_c = c;
                max_ind_g = g;
            }
        }
        printf("\n");
    }

    printf("\n");

    par->C = Cs[max_ind_c];
    par->gamma = Gs[max_ind_g];
}


struct problem *ParEst::get_lp_int(struct problem *lp, int kp) {
    struct problem *lp_int = Malloc(struct problem, 1);
    lp_int->l = kp;
    lp_int->x = Malloc(struct node*, lp_int->l);
    lp_int->y = Malloc(double, lp_int->l);

    for (int r = 0; r < lp_int->l; r++) {
        lp_int->x[r] = lp->x[r];
        lp_int->y[r] = lp->y[r];
    }

    return lp_int;
}


struct problem *ParEst::get_lp_ext(struct problem *lp, int kp, int k) {
    struct problem *lp_ext = Malloc(struct problem, 1);
    lp_ext->l = k - kp;
    lp_ext->x = Malloc(struct node*, lp_ext->l);
    lp_ext->y = Malloc(double, lp_ext->l);

    for (int r = 0; r < lp_ext->l; r++) {
        lp_ext->x[r] = lp->x[kp + r];
        lp_ext->y[r] = lp->y[kp + r];
    }

    return lp_ext;
}



