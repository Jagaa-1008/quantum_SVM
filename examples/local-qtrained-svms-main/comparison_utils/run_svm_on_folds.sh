#!/usr/bin/bash

# Function to print the correct usage on terminal
function usage {
    echo "Usage: ./run_svm_on_folds.sh [-l lib_svm_dir] [-f folds_data_dir] [-b folds_files_prefix] [-n num_folds] [-t kernel_type] [-g gamma] [-c cost] [-o output_dir]"
    echo "    -l lib_svm_dir = LibSVM directory containing the executable files for training/test"
    echo "    -f folds_data_dir = directory containing the folds training/test data files"
    echo "    -b folds_files_prefix = common prefix for the folds training/test data files"
    echo "    -n num_folds = number of folds present in folds_data_dir"
    echo "    -t kernel_type = type of kernel function"
    echo "    -g gamma = gamma parameter for the RBF kernel"
    echo "    -c cost = SVM cost (C) value"
    echo "    -o output_dir = directory where to store the output files"
    exit 0
}

# Variables default values
lib_svm_dir="LibSVM_2.88"
folds_data_dir="../results/binary/falk_svm_c/toulouse_t4_2c_500s"
folds_files_prefix="model"
num_folds=10
kernel_type=2
gamma=1
cost=3
output_dir="../results/binary/svm_c/toulouse_t4_2c_500s"

# Parse the script arguments
while getopts 'l:f:b:n:t:g:c:o:h' option; do
    case "$option" in
        l) lib_svm_dir="${OPTARG}";;
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

    ./"${lib_svm_dir}"/svm-train -t "${kernel_type}" -g "${gamma}" -c "${cost}" "${fold_i_training_file}" "${fold_i_model_file}"
    echo ""
    ./"${lib_svm_dir}"/svm-predict "${fold_i_test_file}" "${fold_i_model_file}" "${fold_i_res_file}"

    echo "============================================================================================================"
    echo ""
done
