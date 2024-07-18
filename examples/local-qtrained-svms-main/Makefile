CXX? = g++
CFLAGS = --std=c++17 -I$(CONDA_PREFIX)/include/python3.8 -Wall -Wconversion -O3 -fPIC -Wno-unused-variable -Wno-unused-but-set-variable
LDFLAGS = -L$(CONDA_PREFIX)/lib -lpython3.8 -lm
PY_MODELS_SUBDIR = Python_models
FALK_SUBDIR = FaLK-SVM

all: py-models-subdir FaLK-SVM-train FaLK-SVM-predict

FaLK-SVM-train: FaLK-SVM-train.cpp falk-subdir-make
	$(CXX) $(CFLAGS) FaLK-SVM-train.cpp $(addprefix $(FALK_SUBDIR)/, cover_tree.o fast_lsvm.o knn.o par_est.o per_eval.o svm.o) -o FaLK-SVM-train.out $(LDFLAGS)
FaLK-SVM-predict: FaLK-SVM-train.cpp falk-subdir-make
	$(CXX) $(CFLAGS) FaLK-SVM-predict.cpp $(addprefix $(FALK_SUBDIR)/, cover_tree.o fast_lsvm.o knn.o par_est.o per_eval.o svm.o) -o FaLK-SVM-predict.out $(LDFLAGS)

py-models-subdir:
	$(MAKE) -C $(PY_MODELS_SUBDIR) all
falk-subdir-make:
	$(MAKE) -C $(FALK_SUBDIR) all

clean: py-models-clean falk-subdir-clean
	rm -f *~ FaLK-SVM-train.out FaLK-SVM-predict.out
py-models-clean:
	$(MAKE) -C $(PY_MODELS_SUBDIR) clean
falk-subdir-clean:
	$(MAKE) -C $(FALK_SUBDIR) clean
