#!/usr/bin/bash

# Function to print the correct usage on terminal
function usage {
    echo "Usage: ./run_cs_svm_on_folds.sh [-d cs_svm_dir] [-f folds_data_dir] [-b folds_files_prefix] [-n num_folds] [-t kernel_type] [-g gamma] [-c cost] [-o output_dir]"
    echo "    -d cs_svm_dir = directory containing the CS-SVM executable files for training/test"
    echo "    -f folds_data_dir = directory containing the folds training/test data files"
    echo "    -b folds_files_prefix = common prefix for the folds training/test data files"
    echo "    -n num_folds = number of folds present in folds_data_dir"
    echo "    -t kernel_type = type of kernel function"
    echo "    -g gamma = gamma parameter for the RBF kernel"
    echo "    -c cost = CS-SVM C parameter (it is equivalent to 1.0/beta)"
    echo "    -o output_dir = directory where to store the output files"
    exit 0
}

# Variables default values
cs_svm_dir="../Python_models/classical/crammer_singer_svm/CSSVM_lib"
folds_data_dir="../results/multiclass/falk_svm_cs/toulouse_t4_3c_150s"
folds_files_prefix="model"
num_folds=10
kernel_type=2
gamma=1
cost=1
output_dir="../results/multiclass/svm_cs/toulouse_t4_3c_150s"

# Parse the script arguments
while getopts 'd:f:b:n:t:g:c:o:h' option; do
    case "$option" in
        d) cs_svm_dir="${OPTARG}";;
        f) folds_data_dir="${OPTARG}";;
        b) folds_files_prefix="${OPTARG}";;
        n) num_folds="${OPTARG}";;
        t) kernel_type="${OPTARG}";;
        g) gamma="${OPTARG}";;
        c) cost="${OPTARG}";;
        o) output_dir="${OPTARG}";;
        h) usage;;
        *) usage;;
    esac
done
shift "$((OPTIND -1))"

# Create the output directory
mkdir -p "${output_dir}"

# Run training and test scripts accordingly
for i in $(seq 0 $((num_folds - 1))); do
    echo "============================================================================================================"
    echo "Fold num ${i}"

    fold_i_training_file="${folds_data_dir}/${folds_files_prefix}_fold_${i}_training_set.txt"
    fold_i_test_file="${folds_data_dir}/${folds_files_prefix}_fold_${i}_test_set.txt"

    fold_i_model_file="${output_dir}/model_fold_${i}.txt"
    fold_i_res_file="${output_dir}/model_fold_${i}_res.csv"

    "${cs_svm_dir}"/svm_multiclass_learn -c "${cost}" -t "${kernel_type}" -g "${gamma}" -v 0 "${fold_i_training_file}" "${fold_i_model_file}"
    echo ""
    "${cs_svm_dir}"/svm_multiclass_classify "${fold_i_test_file}" "${fold_i_model_file}" "${fold_i_res_file}"

    echo "============================================================================================================"
    echo ""
done
