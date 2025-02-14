# conda activate mixed_methods

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11/

export TF_ENABLE_ONEDNN_OPTS=0
#this is for consitency of results and preventing small variations in calculation via floating point rounding

export TF_FORCE_GPU_ALLOW_GROWTH=true
#to limit the memory consumption of tensorflow

IMAGES_SUBFOLDER=test_images

DATASET_FILE_1x1_CORR=dataset_files/TZA_Rect_DHS4_corr_pos_Meta_dataframe_1x1.csv

DATASET_FILE_3x3_CORR=dataset_files/TZA_Rect_DHS4_corr_pos_Meta_dataframe_3x3.csv

MODEL_PATH=models/TZA_Rect_complete_plus_TZA_surr_FT_CNN_regression_logvalues_Fine_tuning_marseille_mobilenetv2_2023-10-03--12-27-21_TZArs_Epochs_50_LR_001_BS_100_L2reg_01_p25_TZA_rs_GPU_.tf

WEALTH_RATINGS=dataset_files/Cluster_wealth_ratings.csv

EXPERT_RATINGS=dataset_files/Complete_dataset_4Sept22.csv

# python evaluate_classification.py $MODEL_PATH $IMAGES_SUBFOLDER $WEALTH_RATINGS $EXPERT_RATINGS $DATASET_FILE_1x1_CORR 1x1

python evaluate_classification.py $MODEL_PATH $IMAGES_SUBFOLDER $WEALTH_RATINGS $EXPERT_RATINGS $DATASET_FILE_3x3_CORR 3x3
