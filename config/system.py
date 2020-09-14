import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# SET THESE PATHS MANUALLY #########################################
# ==================================================================

# ==================================================================
# name of the host - used to check if running on cluster or not
# ==================================================================
local_hostnames = ['bmicdl05']

# ==================================================================
# project dirs
# ==================================================================
project_root = '/usr/bmicnas01/data-biwi-01/nkarani/projects/dg_seg/'
project_data_root = os.path.join(project_root, 'data/')
project_code_root = os.path.join(project_root, 'methods/baselines/uda/adv_feature_classifier/')

# ==================================================================
# data dirs
# ==================================================================
orig_data_root_hcp = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/HCP/3T_Structurals_Preprocessed/'
orig_data_root_abide = '/usr/bmicnas01/data-biwi-01/nkarani/projects/dg_seg/data/original/abide/'
orig_data_root_pfizer = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/PfizerData/'
orig_data_root_ixi = '/usr/bmicnas01/data-biwi-01/nkarani/projects/dg_seg/data/original/ixi/'
orig_data_root_nci = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/NCI_Prostate'
orig_data_root_pirad_erc = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/Prostate/'
orig_data_root_promise = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/Prostate_PROMISE12/TrainingData/'
orig_data_root_acdc = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/ACDC_challenge_new/'
orig_data_root_rvsc = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/Cardiac_RVSC/AllData/'
orig_data_root_mmwhs_mr = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/MMWHS/MR/'
orig_data_root_mmwhs_ct = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/MMWHS/CT/'

# ==================================================================
# dirs where the pre-processed data is stored
# ==================================================================
preproc_folder_hcp = os.path.join(project_data_root,'preprocessed/hcp/')
preproc_folder_abide = os.path.join(project_data_root,'preprocessed/abide/')
preproc_folder_ixi = os.path.join(project_data_root,'preprocessed/ixi/')
preproc_folder_nci = os.path.join(project_data_root,'preprocessed/nci/')
preproc_folder_pirad_erc = os.path.join(project_data_root,'preprocessed/pirad_erc/')
preproc_folder_promise = os.path.join(project_data_root,'preprocessed/promise/')
preproc_folder_acdc = os.path.join(project_data_root,'preprocessed/acdc/')
preproc_folder_rvsc = os.path.join(project_data_root,'preprocessed/rvsc/')
preproc_folder_mmwhs_mr = os.path.join(project_data_root,'preprocessed/mmwhs/mr/')
preproc_folder_mmwhs_ct = os.path.join(project_data_root,'preprocessed/mmwhs/ct/')

# ==================================================================
# log root
# ==================================================================
log_root = os.path.join(project_code_root, 'logdir/')