#ifndef _PAR_EST_H
#define _PAR_EST_H

class ParEst {
public:

    ParEst(int opt = 0, int n_folds = 10);

    ~ParEst();

    double estimateGamma(struct problem *p);

    double estG_quant(struct problem *p, double quant);

    double estC(struct problem **pb, int nms, struct lkm_parameter *par, double *Cs, int n_cs, int kp, int k);

    double estCG(struct problem **pb, int nms, struct lkm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs, int kp,
          int k);

    double estCGK(struct problem **pb, int nms, struct lkm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs, int *Ks,
           int n_ks, int *KPs, int n_kps, FILE *log_fp);

    double estCGEIABK(struct problem **pb, int nms, struct lkm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs,
                      int *Es, int n_es, int *Is, int n_is, int *As, int n_as, int *Bs, int n_bs, int *Ks, int n_ks, int *KPs, int n_kps,
                      FILE *log_fp);

    double estC(struct problem *pb, struct lkm_parameter *par, double *Cs, int n_cs, int kp, int k);

    double estCG(struct problem *pb, struct svm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs, int kp, int k);

    double estCGK(struct problem *pb, struct lkm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs, int *Ks, int n_ks,
           int *KPs, int n_kps);

private:

    int opt;

    int n_folds;

    double *dists;

    double dist_dot(const node *px, const node *py);

    struct problem *get_lp_int(struct problem *lp, int kp);

    struct problem *get_lp_ext(struct problem *lp, int kp, int k);

    double estimateGammaQ(struct problem *p, double quant);

    double estC_int(struct problem *pb, struct svm_parameter *par, double *Cs, int n_cs, double *res, int kp, int k,
                    int val_k, int model_index, FILE *log_fp);

    double estCG_int(struct problem *pb, struct lkm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs, double **res,
                     int kp, int k, int val_k);

    double estCGK_int(struct problem *pb, struct lkm_parameter *par, double *Cs, int n_cs, double *Gs, int n_gs, int *Ks,
                      int n_ks, int *KPs, int n_kps, int val_k, double ***res);

    void select_best_C(double *res, double *Cs, int n_cs, struct lkm_parameter *par);

    void select_best_CG(double **res, double *Cs, int n_cs, double *Gs, int n_gs, struct svm_parameter *par);

    void select_best_CGK(double ***res, double *Cs, int n_cs, double *Gs, int n_gs, int *Ks, int n_ks, int *KPs, int n_kps,
                         struct lkm_parameter *par);
    void select_best_CGEIABK(double *******res, double *Cs, int n_cs, double *Gs, int n_gs, int *Es, int n_es, int *Is, int n_is,
                             int *As, int n_as, int *Bs, int n_bs, int *Ks, int n_ks, int *KPs, int n_kps, struct lkm_parameter *par);
};


#endif
