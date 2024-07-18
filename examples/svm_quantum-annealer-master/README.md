# Remote Sensing Image Classification with Ensembles of Support Vector Machines on the D-Wave Quantum Annealer

🗃 This repository contains Python functions and processing pipelines documented in Jupyter notebook for pixel-wise binary classification of remote sensing multispectral images with the D-Wave 2000Q Quantum Annealer (QA).

More information can be found in the conference paper connected to this repository

📜 G. Cavallaro, D. Willsch, M. Willsch, K. Michielsen, and M. Riedel, “Approaching Remote Sensing Image Classification with Ensembles of Support Vector Machines on the D-Wave Quantum Annealer,” in Proceedings of the IEEE International Geoscience and Remote Sensing Symposium (IGARSS), pp. 1973-1976, 2020, https://doi.org/10.1109/IGARSS39084.2020.9323544  

Support Vector Machine (SVM) is a popular supervised Machine Learning (ML) method that is widely used for classification and regression problems.  Recently, a method to train SVMs on a D-Wave 2000Q Quantum Annealer (QA) was proposed for binary classification of some biological data. First, ensembles  of  weak  quantum  SVMs  are  generated  by  training each classifier on a disjoint training subset that can be fit into the QA.  Then, the computed weak solutions are fused for making predictions on unseen data. In this work, the classification of Remote Sensing (RS) multispectral images with SVMs trained on a QA is discussed.  Furthermore, an open code repository is released to facilitate an early entry into the practical application of QA, a new disruptive compute technology.

The work is a follow up of this publication:

📃 D. Willsch, M. Willsch, H. De Raedt, and K. Michielsen, “Support Vector Machines on the D-Wave Quantum Annealer” in Computer Physics Communications, vol. 248, 2020, https://doi.org/10.1016/j.cpc.2019.107006 

----------

👌Everyone can make a free account to run on the D-Wave2000Q computer: 

- Make a free account to run on the D-Wave through 👉 (https://www.dwavesys.com/take-leap

- Install Ocean Software with 'pip install dwave-ocean-sdk' 👉 https://docs.ocean.dwavesys.com/en/latest/overview/install.html

- Configuring the D-Wave System as a Solver with 'dwave config create' 👉 https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html


📐 Now you can proceed in two was:

(1) Follow the instructions of the Jupyter Notebook 👉 run_SVM.ipynb

(2) Make your processing pipeline by using the Python functions: calibrate.py, train.py and test.py. 
    (See in the instructions in files)
    
Have fun!

📬 For any problem, feel free to contact me at g.cavallaro@fz-juelich.de 

