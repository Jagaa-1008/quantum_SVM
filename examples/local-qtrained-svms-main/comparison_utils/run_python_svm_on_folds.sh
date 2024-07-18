#!/usr/bin/bash

# Function to print the correct usage on terminal
function usage {
    echo "Usage: ./run_python_svm_on_folds.sh [-m main_dir] [-f folds_data_dir] [-l folds_files_prefix] [-v num_folds] [-z python_svm_type] [-t kernel_type] [-g gamma] [-c cost] [-b base] [-a binary_vars_num] [-i penalty_coef] [-e multicl_reg] [-o output_dir]"
    echo "    -m main_dir = directory containing the python_svm_main.py file"
    echo "    -f folds_data_dir = directory containing the folds training/test data files"
    echo "    -l folds_files_prefix = common prefix for the folds training/test data files"
    echo "    -v num_folds = number of folds present in folds_data_dir"
    echo "    -z python_svm_type = type of SVM (0 = quantum for binary classification, 1 = quantum for multiclass classification, 2 = classical for multiclass classification (Crammer-Singer))"
    echo "    -t kernel_type = type of kernel function"
    echo "    -g gamma = gamma parameter for the RBF kernel"
    echo "    -c cost = SVM cost (C) value"
    echo "    -b base = base used to encode the coefficients"
    echo "    -a binary_vars_num = number of binary variables used to encode each coefficient"
    echo "    -i penalty_coef: multiplier for penalty terms (ksi, mu) in quantum SVMs"
    echo "    -e multicl_reg: regularization parameter (beta) for the multiclass quantum SVM"
    echo "    -o output_dir = directory where to store the output files"
    exit 0
}

# Variables default values
main_dir="../Python_models"
folds_data_dir="../results/binary/falk_svm_c/toulouse_t4_2c_500s"
folds_files_prefix="model"
num_folds=10
python_svm_type=0
kernel_type=2
gamma=1
cost=3
base=2
binary_vars_num=2
penalty_coef=1
multicl_reg=1
output_dir="../results/binary/svm_q/toulouse_t4_2c_500s"

# Parse the script arguments
while getopts 'm:f:l:v:z:t:g:c:b:a:i:e:o:h' option; do
    case "$option" in
        m) main_dir="${OPTARG}";;
        f) folds_data_dir="${OPTARG}";;
        l) folds_files_prefix="${OPTARG}";;
        v) num_folds="${OPTARG}";;
        z) python_svm_type="${OPTARG}";;
        t) kernel_type="${OPTARG}";;
        g) gamma="${OPTARG}";;
        c) cost="${OPTARG}";;
        b) base="${OPTARG}";;
        a) binary_vars_num="${OPTARG}";;
        i) penalty_coef="${OPTARG}";;
        e) multicl_reg="${OPTARG}";;
        o) output_dir="${OPTARG}";;
        h) usage;;
        *) usage;;
    esac
done
shift "$((OPTIND -1))"

# Create the output directory
mkdir -p "${output_dir}"

# Set the embeddings directory variable
embeddings_dir="${output_dir}/embeddings"

# Run the script accordingly
for i in $(seq 0 $((num_folds - 1))); do
    echo "============================================================================================================"
    echo "Fold num ${i}"

    fold_i_training_file="${folds_data_dir}/${folds_files_prefix}_fold_${i}_training_set.txt"
    fold_i_test_file="${folds_data_dir}/${folds_files_prefix}_fold_${i}_test_set.txt"

    fold_i_model_file="${output_dir}/model_fold_${i}.pkl"
    fold_i_res_file="${output_dir}/model_fold_${i}_res.csv"

    python "${main_dir}"/python_svm_main.py  --training-data-file "${fold_i_training_file}" \
           --python-svm-type "${python_svm_type}" --kernel-type "${kernel_type}" --gamma "${gamma}" --cost "${cost}" \
           --base "${base}" --binary-vars-num "${binary_vars_num}" --penalty-coef "${penalty_coef}" \
           --multicl-reg "${multicl_reg}" --embeddings-dir "${embeddings_dir}" \
           --pickle-filepath "${fold_i_model_file}" --test-data-file "${fold_i_test_file}" \
           --predictions-out-file "${fold_i_res_file}"

    echo "============================================================================================================"
    echo ""
done
