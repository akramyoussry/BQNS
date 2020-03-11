datasets = ./../datasets/
imgs    = ./../imgs/

all: $(datasets)applications.out $(datasets)analysis.out


$(datasets)applications.out:$(datasets)trained_model_CPMG_G_X_28_3000.mlmodel QNS_AS.py qubitmlmodel.py applications.py
	mkdir -p $(imgs)
	python applications.py > $(datasets)applications.out

$(datasets)analysis.out: $(datasets)trained_model_CPMG_G_X_28_3000.mlmodel \
						 $(datasets)trained_model_CPMG_S_X_28_3000.mlmodel \
						 $(datasets)trained_model_CPMG_G_XY_7_3000.mlmodel \
						 $(datasets)trained_model_CPMG_G_XY_pi_7_3000.mlmodel \
						 $(datasets)trained_model_CPMG_G_XY_7_nl_3000.mlmodel \
						 $(datasets)trained_model_CPMG_G_XY_pi_7_nl_3000.mlmodel \
						 analysis.py \
						 $(datasets)CPMG_G_X_28.ds $(datasets)CPMG_S_X_28.ds \
						 $(datasets)CPMG_G_XY_7.ds $(datasets)CPMG_G_XY_pi_7.ds \
						 $(datasets)CPMG_G_XY_7_nl.ds $(datasets)CPMG_G_XY_pi_7_nl.ds \
						 $(datasets)montecarlo_0.ds $(datasets)montecarlo_1.ds $(datasets)montecarlo_2.ds\
						 $(datasets)montecarlo_pulses.ds
	mkdir -p $(imgs)					 
	python analysis.py > $(datasets)analysis.out

datasets_cat1: $(datasets)CPMG_G_X_28.ds $(datasets)CPMG_S_X_28.ds $(datasets)trained_model_CPMG_G_X_28_3000.mlmodel $(datasets)trained_model_CPMG_S_X_28_3000.mlmodel

datasets_cat2: $(datasets)CPMG_G_XY_7.ds $(datasets)CPMG_G_XY_pi_7.ds $(datasets)trained_model_CPMG_G_XY_7_3000.mlmodel $(datasets)trained_model_CPMG_G_XY_pi_7_3000.mlmodel

datasets_cat3: $(datasets)CPMG_G_XY_7_nl.ds $(datasets)CPMG_G_XY_pi_7_nl.ds $(datasets)trained_model_CPMG_G_XY_7_nl_3000.mlmodel $(datasets)trained_model_CPMG_G_XY_pi_7_nl_3000.mlmodel

$(datasets)CPMG_G_X_28.ds $(datasets)CPMG_S_X_28.ds &: simulator.py datasets_cat1.py 
	mkdir -p $(datasets)
	python datasets_cat1.py

$(datasets)CPMG_G_XY_7.ds $(datasets)CPMG_G_XY_pi_7.ds &: simulator.py datasets_cat2.py 
	mkdir -p $(datasets)
	python datasets_cat2.py

$(datasets)CPMG_G_XY_7_nl.ds $(datasets)CPMG_G_XY_pi_7_nl.ds &: simulator.py datasets_cat3.py
	mkdir -p $(datasets)
	python datasets_cat3.py

$(datasets)trained_model_CPMG_G_X_28_3000.mlmodel $(datasets)trained_model_CPMG_S_X_28_3000.mlmodel &: $(datasets)CPMG_G_X_28.ds $(datasets)CPMG_S_X_28.ds QNS_AS.py qubitmlmodel.py train_model_cat1.py
	python train_model_cat1.py

$(datasets)trained_model_CPMG_G_XY_7_3000.mlmodel $(datasets)trained_model_CPMG_G_XY_pi_7_3000.mlmodel &: $(datasets)CPMG_G_XY_7.ds $(datasets)CPMG_G_XY_pi_7.ds QNS_AS.py qubitmlmodel.py train_model_cat2.py
	python train_model_cat2.py

$(datasets)trained_model_CPMG_G_XY_7_nl_3000.mlmodel $(datasets)trained_model_CPMG_G_XY_pi_7_nl_3000.mlmodel &: $(datasets)CPMG_G_XY_7_nl.ds $(datasets)CPMG_G_XY_pi_7_nl.ds  QNS_AS.py  qubitmlmodel.py train_model_cat3.py
	python train_model_cat3.py



$(datasets)montecarlo_0.ds $(datasets)montecarlo_1.ds $(datasets)montecarlo_2.ds $(datasets)montecarlo_pulses.ds: simulator.py  monte_carlo_analysis.py
	python monte_carlo_analysis.py
	






