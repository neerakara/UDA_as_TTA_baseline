# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import tensorflow as tf
import numpy as np
import utils
import utils_vis
import model as model
import config.system as sys_config

import data.data_hcp as data_hcp
import data.data_abide as data_abide
import data.data_nci as data_nci
import data.data_pirad_erc as data_pirad_erc

from skimage.transform import rescale
import sklearn.metrics as met

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import uda_cycle_gan as exp_config
    
# ==================================================================
# main function for training
# ==================================================================
def predict_segmentation(subject_name,
                         image,
                         normalize = True,
                         evaluate_target_domain = True):
    
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ================================================================
        # create placeholders
        # ================================================================
        images_pl = tf.placeholder(tf.float32,
                                   shape = [None] + list(exp_config.image_size) + [1],
                                   name = 'images')

        # ================================================================
        # ================================================================
        if evaluate_target_domain is False:
            images_sd_to_td = model.transform_images(images_pl,
                                                     exp_config,
                                                     training_pl = tf.constant(False, dtype=tf.bool),
                                                     scope_name = 'generator_sd_to_td',
                                                     scope_reuse = False)
            
            # ================================================================
            # insert a normalization module in front of the segmentation network
            # the normalization module is trained for each test image
            # ================================================================
            images_normalized, added_residual = model.normalize(images_sd_to_td,
                                                                exp_config,
                                                                training_pl = tf.constant(False, dtype=tf.bool))
            
            # ================================================================
            # build the graph that computes predictions from the inference model
            # ================================================================
            logits, softmax, preds = model.predict_i2l(images_normalized,
                                                       exp_config,
                                                       training_pl = tf.constant(False, dtype=tf.bool))
            
            # ================================================================
            # divide the vars into segmentation network, normalization network and the discriminator network
            # ================================================================
            i2l_vars = []
            normalization_vars = []
            generator_vars = []
            
            for v in tf.global_variables():
                var_name = v.name        
                if 'image_normalizer' in var_name:
                    normalization_vars.append(v)
                    i2l_vars.append(v) # the normalization vars also need to be restored from the pre-trained i2l mapper
                elif 'i2l_mapper' in var_name:
                    i2l_vars.append(v)
                elif 'generator' in var_name:
                    generator_vars.append(v)
                    
            # ================================================================
            # create savers
            # ================================================================
            saver_i2l = tf.train.Saver(var_list = i2l_vars)
            saver_normalizer = tf.train.Saver(var_list = normalization_vars) 
            saver_generators = tf.train.Saver(var_list = generator_vars)
        
        else:
            # ================================================================
            # insert a normalization module in front of the segmentation network
            # the normalization module is trained for each test image
            # ================================================================
            images_normalized, added_residual = model.normalize(images_pl,
                                                                exp_config,
                                                                training_pl = tf.constant(False, dtype=tf.bool))
        
            # ================================================================
            # build the graph that computes predictions from the inference model
            # ================================================================
            logits, softmax, preds = model.predict_i2l(images_normalized,
                                                       exp_config,
                                                       training_pl = tf.constant(False, dtype=tf.bool))
                        
            # ================================================================
            # divide the vars into segmentation network and normalization network
            # ================================================================
            i2l_vars = []
            normalization_vars = []
        
            for v in tf.global_variables():
                var_name = v.name        
                i2l_vars.append(v)
                if 'image_normalizer' in var_name:
                    normalization_vars.append(v)
                    
            # ================================================================
            # create saver
            # ================================================================
            saver_i2l = tf.train.Saver(var_list = i2l_vars)
            saver_normalizer = tf.train.Saver(var_list = normalization_vars) 
                                            
        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
                
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()
                
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        sess.run(init_ops)
        
        # ================================================================
        # Restore the segmentation network parameters
        # ================================================================
        if exp_config.uda is False:
            logging.info('============================================================')        
            path_to_model = sys_config.log_root + exp_config.expname_i2l + '/models/'
            checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
            saver_i2l.restore(sess, checkpoint_path)
            
        if exp_config.uda is True:
            logging.info('============================================================')        
            path_to_model = sys_config.log_root + exp_config.expname_uda + '/models/'
            checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'lowest_loss.ckpt') #lowest_loss
            saver_i2l.restore(sess, checkpoint_path)

            if evaluate_target_domain is False:            
                # also save generators at the best loss points
                path_to_model = sys_config.log_root + exp_config.expname_uda + '/models/'
                checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'generators_at_lowest_loss.ckpt')
                saver_generators.restore(sess, checkpoint_path)
        
        # ================================================================
        # Restore the normalization network parameters
        # ================================================================
        if normalize is True:
            logging.info('============================================================')
            path_to_model = os.path.join(sys_config.log_root, exp_config.expname_normalizer) + '/subject_' + subject_name + '/models/'
            checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_score.ckpt')
            saver_normalizer.restore(sess, checkpoint_path)
            logging.info('============================================================')
        
        # ================================================================
        # Make predictions for the image at the resolution of the image after pre-processing
        # ================================================================
        mask_predicted = []
        img_normalized = []
        
        for b_i in range(0, image.shape[0], 1):
        
            X = np.expand_dims(image[b_i:b_i+1, ...], axis=-1)
            
            mask_predicted.append(sess.run(preds, feed_dict={images_pl: X}))
            img_normalized.append(sess.run(images_normalized, feed_dict={images_pl: X}))
        
        mask_predicted = np.squeeze(np.array(mask_predicted)).astype(float)  
        img_normalized = np.squeeze(np.array(img_normalized)).astype(float)  
        
        sess.close()
        
        return mask_predicted, img_normalized
    
# ================================================================
# ================================================================
def rescale_and_crop(arr,
                     px,
                     py,
                     nx,
                     ny,
                     order_interpolation,
                     num_rotations):
    
    # 'target_resolution_brain' contains the resolution that the images were rescaled to, during the pre-processing.
    # we need to undo this rescaling before evaluation
    scale_vector = [exp_config.target_resolution_brain[0] / px,
                    exp_config.target_resolution_brain[1] / py]

    arr_list = []
    
    for zz in range(arr.shape[0]):
     
        # ============
        # rotate the labels back to the original orientation
        # ============            
        arr2d_rotated = np.rot90(np.squeeze(arr[zz, :, :]), k=num_rotations)
        
        arr2d_rescaled = rescale(arr2d_rotated,
                                 scale_vector,
                                 order = order_interpolation,
                                 preserve_range = True,
                                 multichannel = False,
                                 mode = 'constant')

        arr2d_rescaled_cropped = utils.crop_or_pad_slice_to_size(arr2d_rescaled, nx, ny)

        arr_list.append(arr2d_rescaled_cropped)
    
    arr_orig_res_and_size = np.array(arr_list)
    arr_orig_res_and_size = arr_orig_res_and_size.swapaxes(0, 1).swapaxes(1, 2)
    
    return arr_orig_res_and_size
        
# ==================================================================
# ==================================================================
def main():
    
    # ===================================
    # read the test images
    # ===================================
    if exp_config.evaluate_td is True:
        test_dataset_name = exp_config.test_dataset
    else:
        test_dataset_name = exp_config.train_dataset
    
    if test_dataset_name is 'HCPT1':
        logging.info('Reading HCPT1 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        
        image_depth = exp_config.image_depth_hcp
        idx_start = 50
        idx_end = 70       
        
        data_test = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                         preprocessing_folder = sys_config.preproc_folder_hcp,
                                                         idx_start = idx_start,
                                                         idx_end = idx_end,                
                                                         protocol = 'T1',
                                                         size = exp_config.image_size,
                                                         depth = image_depth,
                                                         target_resolution = exp_config.target_resolution_brain)
        
        imts = data_test['images']
        name_test_subjects = data_test['patnames']
        num_test_subjects = imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)       
        
        orig_data_res_x = data_test['px'][:]
        orig_data_res_y = data_test['py'][:]
        orig_data_res_z = data_test['pz'][:]
        orig_data_siz_x = data_test['nx'][:]
        orig_data_siz_y = data_test['ny'][:]
        orig_data_siz_z = data_test['nz'][:]
        
    elif test_dataset_name is 'HCPT2':
        logging.info('Reading HCPT2 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        
        image_depth = exp_config.image_depth_hcp
        idx_start = 50
        idx_end = 70
        
        data_test = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                         preprocessing_folder = sys_config.preproc_folder_hcp,
                                                         idx_start = idx_start,
                                                         idx_end = idx_end,           
                                                         protocol = 'T2',
                                                         size = exp_config.image_size,
                                                         depth = image_depth,
                                                         target_resolution = exp_config.target_resolution_brain)
        
        imts = data_test['images']
        name_test_subjects = data_test['patnames']
        num_test_subjects = imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)       
        
        orig_data_res_x = data_test['px'][:]
        orig_data_res_y = data_test['py'][:]
        orig_data_res_z = data_test['pz'][:]
        orig_data_siz_x = data_test['nx'][:]
        orig_data_siz_y = data_test['ny'][:]
        orig_data_siz_z = data_test['nz'][:]
        
    elif test_dataset_name is 'CALTECH':
        logging.info('Reading CALTECH images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'CALTECH/')
        
        image_depth = exp_config.image_depth_caltech
        idx_start = 16
        idx_end = 36         
        
        data_test = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                           preprocessing_folder = sys_config.preproc_folder_abide,
                                                           site_name = 'CALTECH',
                                                           idx_start = idx_start,
                                                           idx_end = idx_end,             
                                                           protocol = 'T1',
                                                           size = exp_config.image_size,
                                                           depth = image_depth,
                                                           target_resolution = exp_config.target_resolution_brain)        
    
        imts = data_test['images']
        name_test_subjects = data_test['patnames']
        num_test_subjects = imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)       
        
        orig_data_res_x = data_test['px'][:]
        orig_data_res_y = data_test['py'][:]
        orig_data_res_z = data_test['pz'][:]
        orig_data_siz_x = data_test['nx'][:]
        orig_data_siz_y = data_test['ny'][:]
        orig_data_siz_z = data_test['nz'][:]
            
    elif test_dataset_name is 'NCI':
        data_test = data_nci.load_and_maybe_process_data(input_folder=sys_config.orig_data_root_nci,
                                                         preprocessing_folder=sys_config.preproc_folder_nci,
                                                         size=exp_config.image_size,
                                                         target_resolution=exp_config.target_resolution_prostate,
                                                         force_overwrite=False,
                                                         cv_fold_num = 1)

        imts = data_test['images_test']
        name_test_subjects = data_test['patnames_test']

        orig_data_res_x = data_test['px_test'][:]
        orig_data_res_y = data_test['py_test'][:]
        orig_data_res_z = data_test['pz_test'][:]
        orig_data_siz_x = data_test['nx_test'][:]
        orig_data_siz_y = data_test['ny_test'][:]
        orig_data_siz_z = data_test['nz_test'][:]

        num_test_subjects = orig_data_siz_z.shape[0]
        ids = np.arange(num_test_subjects)

    elif test_dataset_name is 'PIRAD_ERC':

        idx_start = 0
        idx_end = 20
        ids = np.arange(idx_start, idx_end)

        data_test = data_pirad_erc.load_data(input_folder=sys_config.orig_data_root_pirad_erc,
                                             preproc_folder=sys_config.preproc_folder_pirad_erc,
                                             idx_start=idx_start,
                                             idx_end=idx_end,
                                             size=exp_config.image_size,
                                             target_resolution=exp_config.target_resolution_prostate,
                                             labeller='ek')
        imts = data_test['images']
        name_test_subjects = data_test['patnames']

        orig_data_res_x = data_test['px'][:]
        orig_data_res_y = data_test['py'][:]
        orig_data_res_z = data_test['pz'][:]
        orig_data_siz_x = data_test['nx'][:]
        orig_data_siz_y = data_test['ny'][:]
        orig_data_siz_z = data_test['nz'][:]

        num_test_subjects = orig_data_siz_z.shape[0]
        
    # ================================   
    # set the log directory
    # ================================   
    if exp_config.normalize is True:
        log_dir = os.path.join(sys_config.log_root, exp_config.expname_normalizer)
    else:
        if exp_config.uda is False:        
            log_dir = sys_config.log_root + exp_config.expname_i2l
        else:
            log_dir = sys_config.log_root + exp_config.expname_uda

    # ================================   
    # open a text file for writing the mean dice scores for each subject that is evaluated
    # ================================       
    results_file = open(log_dir + '/' + test_dataset_name + '_' + 'test' + '.txt', "w")
    results_file.write("================================== \n") 
    results_file.write("Test results \n") 
    
    # ================================================================
    # For each test image, load the best model and compute the dice with this model
    # ================================================================
    dice_per_label_per_subject = []
    hsd_per_label_per_subject = []

    for sub_num in range(5):#(num_test_subjects): 

        subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
        subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
        image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
        
        # ==================================================================
        # setup logging
        # ==================================================================
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        subject_name = str(name_test_subjects[sub_num])[2:-1]
        logging.info('============================================================')
        logging.info('Subject id: %s' %sub_num)
    
        # ==================================================================
        # predict segmentation at the pre-processed resolution
        # ==================================================================
        predicted_labels, normalized_image = predict_segmentation(subject_name,
                                                                  image,
                                                                  exp_config.normalize,
                                                                  exp_config.evaluate_td)

        # ==================================================================
        # read the original segmentation mask
        # ==================================================================
        if test_dataset_name is 'HCPT1':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_hcp.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_hcp,
                                                                              idx = ids[sub_num],
                                                                              protocol = 'T1',
                                                                              preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                              depth = image_depth)
            num_rotations = 0  
            
        elif test_dataset_name is 'HCPT2':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_hcp.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_hcp,
                                                                              idx = ids[sub_num],
                                                                              protocol = 'T2',
                                                                              preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                              depth = image_depth)
            num_rotations = 0  

        elif test_dataset_name is 'CALTECH':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_abide.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_abide,
                                                                               site_name = 'CALTECH',
                                                                               idx = ids[sub_num],
                                                                               depth = image_depth)
            num_rotations = 0

        elif test_dataset_name is 'STANFORD':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_abide.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_abide,
                                                                               site_name = 'STANFORD',
                                                                               idx = ids[sub_num],
                                                                               depth = image_depth)
            num_rotations = 0
            
        elif test_dataset_name is 'NCI':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_nci.load_without_size_preprocessing(sys_config.orig_data_root_nci,
                                                                               cv_fold_num=1,
                                                                               train_test='test',
                                                                               idx=ids[sub_num])
            num_rotations = 0

        elif test_dataset_name is 'PIRAD_ERC':
            # image will be normalized to [0,1]
            image_orig, labels_orig = data_pirad_erc.load_without_size_preprocessing(sys_config.orig_data_root_pirad_erc,
                                                                                     ids[sub_num],
                                                                                     labeller='ek')
            num_rotations = -3
            
        # ==================================================================
        # convert the predicitons back to original resolution
        # ==================================================================
        predicted_labels_orig_res_and_size = rescale_and_crop(predicted_labels,
                                                              orig_data_res_x[sub_num],
                                                              orig_data_res_y[sub_num],
                                                              orig_data_siz_x[sub_num],
                                                              orig_data_siz_y[sub_num],
                                                              order_interpolation = 0,
                                                              num_rotations = num_rotations)
        
        normalized_image_orig_res_and_size = rescale_and_crop(normalized_image,
                                                              orig_data_res_x[sub_num],
                                                              orig_data_res_y[sub_num],
                                                              orig_data_siz_x[sub_num],
                                                              orig_data_siz_y[sub_num],
                                                              order_interpolation = 1,
                                                              num_rotations = num_rotations)
        
        # ==================================================================
        # If only whole-gland comparisions are desired, merge the labels in both ground truth segmentations as well as the predictions
        # ==================================================================
        if exp_config.whole_gland_results is True:
            predicted_labels_orig_res_and_size[predicted_labels_orig_res_and_size!=0] = 1
            labels_orig[labels_orig!=0] = 1
            nl = 2
            savepath = log_dir + '/' + test_dataset_name + '_test_' + subject_name + '_whole_gland.png'
        else:
            nl = exp_config.nlabels
            savepath = log_dir + '/' + test_dataset_name + '_test_' + subject_name + '.png'
            
        # ==================================================================
        # compute dice at the original resolution
        # ==================================================================    
        dice_per_label_this_subject = met.f1_score(labels_orig.flatten(),
                                                   predicted_labels_orig_res_and_size.flatten(),
                                                   average=None)
        
        # ==================================================================    
        # compute Hausforff distance at the original resolution
        # ==================================================================    
        compute_hsd = False
        if compute_hsd is True:
            hsd_per_label_this_subject = utils.compute_surface_distance(y1 = labels_orig,
                                                                        y2 = predicted_labels_orig_res_and_size,
                                                                        nlabels = exp_config.nlabels)
        else:
            hsd_per_label_this_subject = np.zeros((exp_config.nlabels))
        
        # ================================================================
        # save sample results
        # ================================================================
        d_vis = 32 # 256
        ids_vis = np.arange(0, 32, 4) # ids = np.arange(48, 256-48, (256-96)//8)
        utils_vis.save_sample_prediction_results(x = utils.crop_or_pad_volume_to_size_along_z(image_orig, d_vis),
                                                 x_norm = utils.crop_or_pad_volume_to_size_along_z(normalized_image_orig_res_and_size, d_vis),
                                                 y_pred = utils.crop_or_pad_volume_to_size_along_z(predicted_labels_orig_res_and_size, d_vis),
                                                 gt = utils.crop_or_pad_volume_to_size_along_z(labels_orig, d_vis),
                                                 num_rotations = - num_rotations, # rotate for consistent visualization across datasets
                                                 savepath = savepath,
                                                 nlabels = nl,
                                                 ids=ids_vis)
                                   
        # ================================
        # write the mean fg dice of this subject to the text file
        # ================================
        results_file.write(subject_name + ":: dice (mean, std over all FG labels): ")
        results_file.write(str(np.round(np.mean(dice_per_label_this_subject[1:]), 3)) + ", " + str(np.round(np.std(dice_per_label_this_subject[1:]), 3)))
        results_file.write(", hausdorff distance (mean, std over all FG labels): ")
        results_file.write(str(np.round(np.mean(hsd_per_label_this_subject), 3)) + ", " + str(np.round(np.std(dice_per_label_this_subject[1:]), 3)) + "\n")
        
        dice_per_label_per_subject.append(dice_per_label_this_subject)
        hsd_per_label_per_subject.append(hsd_per_label_this_subject)
    
    # ================================================================
    # write per label statistics over all subjects    
    # ================================================================
    dice_per_label_per_subject = np.array(dice_per_label_per_subject)
    hsd_per_label_per_subject =  np.array(hsd_per_label_per_subject)
    
    # ================================
    # In the array images_dice, in the rows, there are subjects
    # and in the columns, there are the dice scores for each label for a particular subject
    # ================================
    results_file.write("================================== \n") 
    results_file.write("Label: dice mean, std. deviation over all subjects\n")
    for i in range(dice_per_label_per_subject.shape[1]):
        results_file.write(str(i) + ": " + str(np.round(np.mean(dice_per_label_per_subject[:,i]), 3)) + ", " + str(np.round(np.std(dice_per_label_per_subject[:,i]), 3)) + "\n")
    results_file.write("================================== \n") 
    results_file.write("Label: hausdorff distance mean, std. deviation over all subjects\n")
    for i in range(hsd_per_label_per_subject.shape[1]):
        results_file.write(str(i+1) + ": " + str(np.round(np.mean(hsd_per_label_per_subject[:,i]), 3)) + ", " + str(np.round(np.std(hsd_per_label_per_subject[:,i]), 3)) + "\n")
    
    # ==================
    # write the mean dice over all subjects and all labels
    # ==================
    results_file.write("================================== \n") 
    results_file.write("DICE Mean, std. deviation over foreground labels over all subjects: " + str(np.round(np.mean(dice_per_label_per_subject[:,1:]), 3)) + ", " + str(np.round(np.std(dice_per_label_per_subject[:,1:]), 3)) + "\n")
    results_file.write("HSD Mean, std. deviation over labels over all subjects: " + str(np.round(np.mean(hsd_per_label_per_subject), 3)) + ", " + str(np.round(np.std(hsd_per_label_per_subject), 3)) + "\n")
    results_file.write("================================== \n") 
    results_file.close()
        
# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()