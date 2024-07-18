# Local Quantum-Trained SVMs
This repository provides the code for the article "E. Zardini, A. Delilbasic, E. Blanzieri, G. Cavallaro, D. Pastorello. Local Binary and Multiclass SVMs Trained on a Quantum Annealer".

In particular, the code provided here is based on:
- the [FaLK-SVM source code](http://disi.unitn.it/~segata/FaLKM-lib), associated with the article "N. Segata, E. Blanzieri (2010). Fast and Scalable Local Kernel Machines. JMLR";
- the [QBSVM source code](https://gitlab.jsc.fz-juelich.de/sdlrs/quantum-svm-algorithms-for-rs-data-classification/-/tree/master/experiments/QA_SVM?ref_type=heads), associated with the article "D. Willsch, M. Willsch, H. De Raedt, K. Michielsen (2020). Support Vector Machines on the D-Wave Quantum Annealer. Computer Physics Communications";
- the [QMSVM source code](https://gitlab.jsc.fz-juelich.de/sdlrs/qmsvm), associated with the article "A. Delilbasic, B. Le Saux, M. Riedel, K. Michielsen, G. Cavallaro (2023). A Single-Step Multiclass SVM Based on Quantum Annealing for Remote Sensing Data Classification. IEEE JSTARS".

## 1. Prerequisites
In order to compile and run the code, you need to have:
- gcc and g++ compilers;
- Python 3. 

In particular, Python 3.8.10 has been used for the experiments presented in the article. If you do not have Python 3 on your machine, we suggest to install Anaconda Python.

You may also want to create a virtual environment before performing the setup step. If you are using Anaconda Python, the shell command is the following:
```shell
conda create -n "venv" python=3.8.10
```

Lastly, to use the quantum annealers provided by D-Wave, you must have an account on [D-Wave Leap](https://cloud.dwavesys.com/leap/login/?next=/leap/).

## 2. Setup
Once you have met the prerequisites, download the repository. From the command line, the command is the following:
```shell
git clone https://github.com/ZarHenry96/local-qtrained-svms.git
```

Then, you need to download:
- the [LibSVM v2.88 source code](https://github.com/cjlin1/libsvm/releases/tag/v288), which must be placed (without the `libsvm-288` folder) inside the `comparison_utils/LibSVM_2.88` folder;
- the [CS SVM source code](https://www.cs.cornell.edu/people/tj/svm_light/svm_multiclass.html), which must be placed (without the `svm_multiclass` folder) inside the `Python_models/classical/crammer_singer_svm/CSSVM_lib` folder.

At this point, from the project root folder, activate the virtual environment, if you are using it (for Anaconda, the command is `conda activate venv`), and install the required modules by runnning the following command:
```shell
pip install -r requirements.txt
```

After that, you have to compile:
- the LibSVM code, by running the command `make` within the `comparison_utils/LibSVM_2.88` folder. The resulting executables are used by the script `run_svm_on_folds.sh`;
- all the other C++ code, by running the command `make` from the project root folder. Read the two notes below **before** executing the command.

**Note #1:** If you are using a virtual environment created with `venv` (and not with Anaconda Python), you have to replace `$(CONDA_PREFIX)` with `$(VIRTUAL_ENV)` inside `Makefile` and `FaLK-SVM/Makefile` to successfully compile the code.

**Note #2:** Depending on your gcc compiler, the CS SVM code might not compile if you do not:
- move the `$(LDFLAGS)` variables to the end of lines 45 and 48 inside `Python_models/classical/crammer_singer_svm/CSSVM_lib/Makefile`; 
- add `extern` to the declaration of the variable `verbosity` inside `Python_models/classical/crammer_singer_svm/CSSVM_lib/svm_light/svm_hideo.c` (line 34).

Lastly, you need to configure the access to the D-Wave solvers. To do this, follow the instructions provided [here](https://docs.ocean.dwavesys.com/en/stable/overview/install.html#set-up-your-environment).

## 3. Execution
After compiling the code, you can run it by calling the `FaLK-SVM-train.out` and `FaLK-SVM-predict.out` executables with the desired parameters. 

If you want to reproduce the experiments presented in the paper, you can use the commands provided in the `*_commands.txt` files available in the `resources` folder. In particular:
- the datasets used in the experiments, based on [SemCity Toulouse](http://rs.ipb.uni-bonn.de/data/) and [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx), are contained in the `datasets/svmlight_datasets.tar.gz` archive. These datasets have been generated using the scripts available in the `datasets` folder. In the case of Toulouse, the labels with the highest annotator index available have been used as the ground truth. To use these datasets, you have to uncompress the archive first;
- to save the console output into a log file (as done in the experiments presented in the article), the output directory must already exist. You can create the directory structure used in the experiments by running the command `./resources/make_exp_res_dirs.sh` from the project root folder.

Lastly, the scripts located inside the `postprocess` and `visualization` folders allow computing additional performance metrics and visualize the label predictions, respectively. Specifically, the latter has been employed in the large-scale experiment (the commands used are provided in the `resources/large_scale_exp_commands.txt` file).

