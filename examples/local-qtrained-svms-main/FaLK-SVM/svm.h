#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 288

#ifdef __cplusplus
extern "C" {
#endif


void read_problem(const char *filename, struct problem *prob);
int get_features_number(const struct problem *prob);
void compute_means_and_stds(const struct problem *prob, int features_number, double *means, double *stds);
void save_problem(const char *filename, const struct problem *prob);

struct svm_model *svm_train(const struct problem *prob, const struct svm_parameter *param);
struct svm_model *python_svm_train(const struct problem *prob, const struct svm_parameter *param, int features_number, const char * filepath);
void svm_cross_validation(const struct problem *prob, const struct svm_parameter *param, int nr_fold, double *target);
void svm_cross_validation_int(const problem *int_prob, const problem *ext_prob, const svm_parameter *param, int nr_fold, double *target);
void svm_cross_validation_print(const struct problem *prob, const struct svm_parameter *par, int nr_fold, char *filename);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
void svm_save_model2(FILE *fp, const svm_model *model);
void python_svm_save_model(const svm_model *model, const char *filepath);
const char* python_svm_get_relative_filepath(const svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);
svm_model *svm_load_model2(FILE *fp, struct problem *pr, const struct svm_parameter *original_param);
svm_model *python_svm_load_model(const char *filepath, const struct svm_parameter *original_param);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
double svm_get_svr_probability(const struct svm_model *model);

void svm_predict_values(const struct svm_model *model, const struct node *x, double *dec_values);
double svm_predict(const struct svm_model *model, const struct node *x);
double python_svm_predict(const struct svm_model *model, const struct node *x, int features_number);
double svm_predict_probability(const struct svm_model *model, const struct node *x, double *prob_estimates);

void svm_destroy_model(struct svm_model *model, bool destroy_param);
void python_svm_destroy_model(struct svm_model *model, bool destroy_param);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void svm_group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm);

struct svm_parameter *copy_svm_parameter(const struct svm_parameter *from);

extern void (*svm_print_string)(const char *);
extern void print_string_stdout(const char *s);
extern void print_null(const char *s);

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
