import os
import numpy as np
import logging
import gc
import re
import skimage.io as io
import SimpleITK as sitk
import h5py
from skimage import transform
import utils
import config.system as sys_config
import subprocess
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5


# ===============================================================
# convert to nii and do bias correction for all images
    # For conversion to nii:
        # the affine matrix is simply set as the identity matrix.
        # Also, the other header info such as the image resolution is not copied into the nii images
        # This must be read from the original mhd images
# ===============================================================
def convert_to_nii_and_correct_bias_field():
    
    input_folder = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/Prostate_PROMISE12/TrainingData/'
    output_folder = '/usr/bmicnas01/data-biwi-01/nkarani/projects/generative_segmentation/data/preproc_data/promise/'

    for dirName, subdirList, fileList in os.walk(input_folder):               
        
        for filename in fileList:
            
            if re.match(r'Case\d\d.mhd', filename):
                
                patient_id = filename[4:6]
                
                img_input_path = input_folder + filename
                seg_input_path = input_folder + 'Case' + patient_id + '_segmentation.mhd'
                img_output_path = output_folder + 'Case' + patient_id + '.nii.gz'
                seg_output_path = output_folder + 'Case' + patient_id + '_segmentation.nii.gz'
                img_bias_corrected_output_path = output_folder + 'Case' + patient_id + '_n4.nii.gz'
                
                # returns arrays with dimensions sorted as z-x-y
                img = io.imread(img_input_path, plugin='simpleitk')
                seg = io.imread(seg_input_path, plugin='simpleitk')
                
                utils.save_nii(img_path = img_output_path, data = img, affine = np.eye(4))
                utils.save_nii(img_path = seg_output_path, data = seg, affine = np.eye(4))
                
                # ================================
                # do bias field correction
                # ================================
                input_img = img_output_path
                output_img = img_bias_corrected_output_path
                subprocess.call(["/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4_th", input_img, output_img])

    
# ===============================================================
# ===============================================================
def test_train_val_split(patient_id,
                         cv_fold_number):
    
    if cv_fold_number == 1:
        if patient_id < 20: return 'train'
        elif patient_id < 30: return 'validation'
        else: return 'test'
        
    elif cv_fold_number == 2:
        if (patient_id % 5 is 0) or (patient_id % 5 is 2): return 'train'
        if (patient_id % 5 is 3): return 'validation'
        else: return 'test'

# ===============================================================
# ===============================================================
def count_slices_and_patient_ids_list(input_folder,
                                      cv_fold_number):

    num_slices = {'train': 0, 'test': 0, 'validation': 0}       
    patient_ids_list = {'train': [], 'test': [], 'validation': []}
    
    # we know that there are 50 subjects in this dataset: Case00 through till Case49        
    for dirName, subdirList, fileList in os.walk(input_folder):               
        
        for filename in fileList:
            
            if re.match(r'Case\d\d.nii.gz', filename):
                
                patient_id = filename[4:6]
                train_test = test_train_val_split(int(patient_id), cv_fold_number)
                filepath = input_folder + '/' + filename
                patient_ids_list[train_test].append(patient_id)
                img = utils.load_nii(filepath)[0]
                num_slices[train_test] += img.shape[0]               

    return num_slices, patient_ids_list

# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 preproc_folder, # bias corrected images will be saved here already
                 output_file,
                 size,
                 target_resolution,
                 cv_fold_num):

    # =======================
    # create the hdf5 file where everything will be written
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # =======================
    # read all the images and count the number of slices along the append axis (the one with the lowest resolution)
    # =======================
    logging.info('Counting files and parsing meta data...')    
    # using the bias corrected images in the preproc folder for this step
    num_slices, patient_ids_list = count_slices_and_patient_ids_list(preproc_folder,
                                                                     cv_fold_number = cv_fold_num)
        
    # =======================
    # set the number of slices according to what has been found from the previous function
    # =======================
    nx, ny = size
    n_test = num_slices['test']
    n_train = num_slices['train']
    n_val = num_slices['validation']

    # =======================
    # Create datasets for images and masks
    # =======================
    data = {}
    for tt, num_points in zip(['test', 'train', 'validation'], [n_test, n_train, n_val]):

        if num_points > 0:
            data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size), dtype=np.float32)
            data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, [num_points] + list(size), dtype=np.uint8)

    mask_list = {'test': [], 'train': [], 'validation': []}
    img_list = {'test': [], 'train': [], 'validation': []}
    nx_list = {'test': [], 'train': [], 'validation': []}
    ny_list = {'test': [], 'train': [], 'validation': []}
    nz_list = {'test': [], 'train': [], 'validation': []}
    px_list = {'test': [], 'train': [], 'validation': []}
    py_list = {'test': [], 'train': [], 'validation': []}
    pz_list = {'test': [], 'train': [], 'validation': []}
    pat_names_list = {'test': [], 'train': [], 'validation': []}              
                
    # =======================
    # read data of each subject, preprocess it and write to the hdf5 file
    # =======================
    logging.info('Parsing image files')
    for train_test in ['test', 'train', 'validation']:

        write_buffer = 0
        counter_from = 0
        patient_counter = 0
        
        for patient_id in patient_ids_list[train_test]:
            
            filepath_orig_mhd_format = input_folder + 'Case' + patient_id + '.mhd'
            filepath_orig_nii_format = preproc_folder + 'Case' + patient_id + '.nii.gz'
            filepath_bias_corrected_nii_format = preproc_folder + 'Case' + patient_id + '_n4.nii.gz'
            filepath_seg_nii_format = preproc_folder + 'Case' + patient_id + '_segmentation.nii.gz'

            patient_counter += 1
            pat_names_list[train_test].append('case' + patient_id)

            logging.info('================================')
            logging.info('Doing: %s' % filepath_orig_mhd_format)
            
            # ================================    
            # read the original mhd image, in order to extract pixel resolution information
            # ================================    
            img_mhd = sitk.ReadImage(filepath_orig_mhd_format)
            pixel_size = img_mhd.GetSpacing()
            px_list[train_test].append(float(pixel_size[0]))
            py_list[train_test].append(float(pixel_size[1]))
            pz_list[train_test].append(float(pixel_size[2]))

            # ================================    
            # read bias corrected image
            # ================================    
            img = utils.load_nii(filepath_bias_corrected_nii_format)[0]

            # ================================    
            # normalize the image
            # ================================    
            img = utils.normalise_image(img, norm_type='div_by_max')

            # ================================    
            # read the labels
            # ================================    
            mask = utils.load_nii(filepath_seg_nii_format)[0]            
            
            # ================================    
            # skimage io with simple ITKplugin was used to read the images in the convert_to_nii_and_correct_bias_field function.
            # this lead to the arrays being read as z-x-y
            # move the axes appropriately, so that the resolution read above is correct for the corresponding axes.
            # ================================    
            img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
            mask = np.swapaxes(np.swapaxes(mask, 0, 1), 1, 2)
            
            # ================================    
            # write to the dimensions now
            # ================================    
            nx_list[train_test].append(mask.shape[0])
            ny_list[train_test].append(mask.shape[1])
            nz_list[train_test].append(mask.shape[2])

            print('mask.shape')
            print(mask.shape)
            print('img.shape')
            print(img.shape)
            
            ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
            scale_vector = [pixel_size[0] / target_resolution[0],
                            pixel_size[1] / target_resolution[1]]

            for zz in range(img.shape[2]):

                slice_img = np.squeeze(img[:, :, zz])
                slice_rescaled = transform.rescale(slice_img,
                                                   scale_vector,
                                                   order=1,
                                                   preserve_range=True,
                                                   multichannel=False,
                                                   mode = 'constant')

                slice_mask = np.squeeze(mask[:, :, zz])
                mask_rescaled = transform.rescale(slice_mask,
                                                  scale_vector,
                                                  order=0,
                                                  preserve_range=True,
                                                  multichannel=False,
                                                  mode='constant')

                slice_cropped = utils.crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                mask_cropped = utils.crop_or_pad_slice_to_size(mask_rescaled, nx, ny)

                img_list[train_test].append(slice_cropped)
                mask_list[train_test].append(mask_cropped)

                write_buffer += 1

                # Writing needs to happen inside the loop over the slices
                if write_buffer >= MAX_WRITE_BUFFER:

                    counter_to = counter_from + write_buffer
                    _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
                    _release_tmp_memory(img_list, mask_list, train_test)

                    # reset stuff for next iteration
                    counter_from = counter_to
                    write_buffer = 0


        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
        _release_tmp_memory(img_list, mask_list, train_test)

    # Write the small datasets
    for tt in ['test', 'train', 'validation']:
        hdf5_file.create_dataset('nx_%s' % tt, data=np.asarray(nx_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('ny_%s' % tt, data=np.asarray(ny_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('nz_%s' % tt, data=np.asarray(nz_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('px_%s' % tt, data=np.asarray(px_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('py_%s' % tt, data=np.asarray(py_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('pz_%s' % tt, data=np.asarray(pz_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('patnames_%s' % tt, data=np.asarray(pat_names_list[tt], dtype="S10"))
    
    # After test train loop:
    hdf5_file.close()

# ===============================================================
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         train_test,
                         img_list,
                         mask_list,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['masks_%s' % train_test][counter_from:counter_to, ...] = mask_arr

# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(img_list, mask_list, train_test):
    
    img_list[train_test].clear()
    mask_list[train_test].clear()
    gc.collect()

# ===============================================================
# ===============================================================
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                size,
                                target_resolution,
                                force_overwrite=False,
                                cv_fold_num = 1):

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_2d_size_%s_res_%s_cv_fold_%d.hdf5' % (size_str, res_str, cv_fold_num)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder,
                     preprocessing_folder,
                     data_file_path,
                     size,
                     target_resolution,
                     cv_fold_num)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')

# ===============================================================
# function to read a single subjects image and labels without any pre-processing
# ===============================================================
def load_without_size_preprocessing(preproc_folder,
                                    patient_id):
                    
    # ==================
    # read bias corrected image and ground truth segmentation
    # ==================
    filepath_bias_corrected_nii_format = preproc_folder + 'Case' + patient_id + '_n4.nii.gz'
    filepath_seg_nii_format = preproc_folder + 'Case' + patient_id + '_segmentation.nii.gz'
    
    # ================================    
    # read bias corrected image
    # ================================    
    image = utils.load_nii(filepath_bias_corrected_nii_format)[0]

    # ================================    
    # normalize the image
    # ================================    
    image = utils.normalise_image(image, norm_type='div_by_max')

    # ================================    
    # read the labels
    # ================================    
    label = utils.load_nii(filepath_seg_nii_format)[0]            
    
    # ================================    
    # skimage io with simple ITKplugin was used to read the images in the convert_to_nii_and_correct_bias_field function.
    # this lead to the arrays being read as z-x-y
    # move the axes appropriately, so that the resolution read above is correct for the corresponding axes.
    # ================================    
    image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
    label = np.swapaxes(np.swapaxes(label, 0, 1), 1, 2)
    
    return image, label

# ===============================================================
# ===============================================================
if __name__ == '__main__':
    input_folder = sys_config.orig_data_root_promise
    preprocessing_folder = sys_config.preproc_folder_promise

    data_promise = load_and_maybe_process_data(input_folder,
                                               preprocessing_folder,
                                               (256, 256),
                                               (0.625, 0.625),
                                               force_overwrite=False,
                                               cv_fold_num = 1)
