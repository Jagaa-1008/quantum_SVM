CXX? = g++
CFLAGS = --std=c++17 -I$(CONDA_PREFIX)/include/python3.8 -Wall -Wconversion -O3 -fPIC -Wno-unused-variable -Wno-unused-but-set-variable

all: cover_tree.o fast_lsvm.o knn.o par_est.o per_eval.o svm.o

cover_tree.o: cover_tree.cpp cover_tree.h defs.h
	$(CXX) $(CFLAGS) -c cover_tree.cpp
fast_lsvm.o: fast_lsvm.cpp fast_lsvm.h defs.h
	$(CXX) $(CFLAGS) -c fast_lsvm.cpp
knn.o: knn.cpp knn.h defs.h
	$(CXX) $(CFLAGS) -c knn.cpp
par_est.o: par_est.cpp par_est.h defs.h
	$(CXX) $(CFLAGS) -c par_est.cpp
per_eval.o: per_eval.cpp per_eval.h defs.h
	$(CXX) $(CFLAGS) -c per_eval.cpp
svm.o: svm.cpp svm.h defs.h
	$(CXX) $(CFLAGS) -c svm.cpp

clean:
	rm -f *~ *.o
