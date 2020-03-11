# BQNS

This is the implementation of the proposed method in https://arxiv.org/abs/. The implementation is based on Tensorflow 1.12. The "imgs" folder contains all the figures presented in the paper. The datsets folder constains the datasets created and used for generating the results in the paper as well as the trained models. The "notebooks" folder contains a Jupyter notebook as an example on how to use the source code. To make use of parallel processing, all code should be run as python files from the terminal rather than running a notebook.
 
The "src" folder contains the following source files:

- Generating datasets:
	- datasets_cat1.py  		: This module implements functions for generating the category 1 datasets used for training the testing of the proposed algorithm
	- datasets_cat2.py			: This module implements functions for generating the category 2 datasets used for training the testing of the proposed algorithm
	- datasets_cat3.py 			: This module implements functions for generating the category 3 datasets used for training the testing of the proposed algorithm
	- simulator.py				: This module implements a noisy qubit simulator

- Training models:
	- train_model_cat1.py		: This module is for training the ML model using category 1 datasets
	- train_model_cat2.py		: This module is for training the ML model using category 2 datasets
	- train_model_cat3.py		: This module is for training the ML model using category 3 datasets
	- qubitmlmodel.py           : This module impelements the machine learning-based model for the qubit
	- QNS_AS.py 				: This module implements the Alvarez-Suter algorithm for quantum noise spectroscopy

- Analysis and results:
	- monte_carlo_analysis.py   : This module implements the anaylsis of the Monte Carlo method to specify the suitable number of noise realizations needed 
	- applications.py			: This module is for implementing different applications using the trained model
	- analysis.py				: This module does all the plots for the performance analysis of the trained models

- Makefile: This is the GNU MAKEFILE that allows running the code easily from any Unix-like system 

In order to run the provided code, run the Makefile in the src folder (run the following command from the terminal: make all). If you want to use our generated datasets, download the zip folder at http:// , and unzip all the contents in the dataset folder. This will include the trained model (those files with extension .mlmodel). If you delete the trained model files and ran the Makefile it will only train the models and generate the results but it will not regenerate the datasets. 