import os
import numpy as np
import logging
import gc
import h5py
import glob
import utils
from skimage.transform import rescale
import config.system as sys_config
from shutil import copyfile
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# =============================================================================================
# FUNCTION FOR COPYING RELEVANT FILES FROM DIFFERENT SITES TO THE LOCAL DISK
# =============================================================================================
def copy_site_files_ixi():
     
    src_folder = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hcp_segmentation/data/IXI/HH/T2w/'
    dst_folder = '/usr/bmicnas01/data-biwi-01/nkarani/projects/generative_segmentation/data/preproc_data/ixi/'

    # read the filenames which have segmentations available
    src_folders_list = sorted(glob.glob(src_folder + '*_seg.nii'))
    
    # iterate through all images which have segmentations available
    for patient_num in range(len(src_folders_list)):
        
        # get the name of the ground truth annotation for this subject
        path_seg = src_folders_list[patient_num]
        path_img = path_seg[:-8] + '.nii.gz'
        patient_name = path_seg[path_seg[:-1].rfind('/') + 1 : -8]
        
        if patient_name == 'IXI014-HH-1236-T2': # this subject has very poor resolution - 256x256x28
            continue
        
        dst_folder_this_patient = dst_folder + patient_name
        if not os.path.exists(dst_folder_this_patient):
            os.makedirs(dst_folder_this_patient)
            
        # copy image to destination
        copyfile(path_img, dst_folder_this_patient + '/img.nii.gz') 
        
        # copy segmentation to destination
        copyfile(path_seg, dst_folder_this_patient + '/seg.nii') 
                            
# =============================================================================================
# DO BIAS FIELD CORRECTION FOR ALL IMAGES
# =============================================================================================
def correct_bias_field():
    
    base_folder = '/usr/bmicnas01/data-biwi-01/nkarani/projects/generative_segmentation/data/preproc_data/ixi/'
    folders_list = sorted(glob.glob(base_folder + '*/'))
    
    for patient_num in range(len(folders_list)):
        
        print('============================================')
        print('subject ' + str(patient_num+1) + ' out of ' + str(len(folders_list)))
        print('============================================')
        
        input_img = folders_list[patient_num] + "img.nii.gz"
        output_img = folders_list[patient_num] + "img_n4.nii.gz"
        subprocess.call(["/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4_th", input_img, output_img])
            
# ===============================================================
# get the path for this image and its label
# ===============================================================
def get_image_and_label_paths(foldername):
        
    _patname = foldername[foldername[:-1].rfind('/') + 1 : -1]
    _imgpath = foldername + 'img_n4.nii.gz'
    _segpath = foldername + 'seg.nii'
                            
    return _patname, _imgpath, _segpath
                            
# ===============================================================
# count the number of 2d slices in the all the images to be pre-processed.
# this will be used to define the size of the image array in the hdf5 file
# ===============================================================
def count_slices(folders_list,
                 idx_start,
                 idx_end,
                 depth):

    num_slices = 0
    
    for idx in range(idx_start, idx_end):    
        
        _, image_path, _ = get_image_and_label_paths(folders_list[idx])        
        image, _, _ = utils.load_nii(image_path)
        num_slices = num_slices + depth # the number of slices along the append axis will be fixed to this number
        
    return num_slices

# ===============================================================
# ===============================================================
def center_image_and_label(image, label):
    
    orig_image_size_x = image.shape[0]
    orig_image_size_y = image.shape[1]
    
    fg_coords = np.where(label > 0)
    x_fg_min = np.min(np.array(fg_coords)[0,:])
    x_fg_max = np.max(np.array(fg_coords)[0,:])
    y_fg_min = np.min(np.array(fg_coords)[1,:])
    y_fg_max = np.max(np.array(fg_coords)[1,:])
    
    border = 20
    x_min = np.maximum(x_fg_min - border, 0)
    x_max = np.minimum(x_fg_max + border, orig_image_size_x)
    y_min = np.maximum(y_fg_min - border, 0)
    y_max = np.minimum(y_fg_max + border, orig_image_size_y)
    
    image_cropped = image[x_min:x_max, y_min:y_max, :]
    label_cropped = label[x_min:x_max, y_min:y_max, :]
    
    return image_cropped, label_cropped

# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 output_file,
                 idx_start,
                 idx_end,
                 protocol,
                 size,
                 depth,
                 target_resolution,
                 preprocessing_folder):

    # ========================    
    # read the filenames
    # ========================
    folders_list = sorted(glob.glob(input_folder + '/*/'))
    logging.info('Number of images in the dataset: %s' % str(len(folders_list)))

    # =======================
    # create a hdf5 file
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # ===============================
    # Create datasets for images and labels
    # ===============================
    data = {}
    num_slices = count_slices(folders_list,
                              idx_start,
                              idx_end,
                              depth)
    
    data['images'] = hdf5_file.create_dataset("images", [num_slices] + list(size), dtype=np.float32)
    data['labels'] = hdf5_file.create_dataset("labels", [num_slices] + list(size), dtype=np.uint8)
    
    # ===============================
    # initialize lists
    # ===============================        
    label_list = []
    image_list = []
    nx_list = []
    ny_list = []
    nz_list = []
    px_list = []
    py_list = []
    pz_list = []
    pat_names_list = []
    
    # ===============================        
    # ===============================        
    write_buffer = 0
    counter_from = 0
    
    # ===============================
    # iterate through the requested indices
    # ===============================
    for idx in range(idx_start, idx_end):
        
        # ==================
        # get file paths
        # ==================
        patient_name, image_path, label_path = get_image_and_label_paths(folders_list[idx])
        
        # ============
        # read the image and normalize it to be between 0 and 1
        # ============
        image, _, image_hdr = utils.load_nii(image_path)
        image = np.swapaxes(image, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
        
        # ==================
        # read the label file
        # ==================        
        label, _, _ = utils.load_nii(label_path)        
        label = np.swapaxes(label, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
        # labels have already been grouped as required
        
        # ============
        # create a segmentation mask and use it to get rid of the skull in the image
        # ============
        label_mask = np.copy(label)
        label_mask[label > 0] = 1
        image = image * label_mask

        # ==================
        # crop out some portion of the image, which are all zeros (rough registration via visual inspection)
        # ==================
        image, label = center_image_and_label(image, label)
            
        # plt.figure(); plt.imshow(image[:,:,50], cmap='gray'); plt.title(patient_name); plt.show(); plt.close()        
                
        # ==================
        # crop volume along z axis (as there are several zeros towards the ends)
        # ==================
        image = utils.crop_or_pad_volume_to_size_along_z(image, depth)
        label = utils.crop_or_pad_volume_to_size_along_z(label, depth)     

        # ==================
        # collect some header info.
        # ==================
        px_list.append(float(image_hdr.get_zooms()[0]))
        py_list.append(float(image_hdr.get_zooms()[2])) # since axes 1 and 2 have been swapped. this is important when dealing with pixel dimensions
        pz_list.append(float(image_hdr.get_zooms()[1]))
        nx_list.append(image.shape[0]) 
        ny_list.append(image.shape[1]) # since axes 1 and 2 have been swapped. however, only the final axis locations are relevant when dealing with shapes
        nz_list.append(image.shape[2])
        pat_names_list.append(patient_name)
        
        # ==================
        # normalize the image
        # ==================
        image_normalized = utils.normalise_image(image, norm_type='div_by_max')
                        
        # ======================================================  
        ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
        # ======================================================
        scale_vector = [image_hdr.get_zooms()[0] / target_resolution[0],
                        image_hdr.get_zooms()[2] / target_resolution[1]] # since axes 1 and 2 have been swapped. this is important when dealing with pixel dimensions

        for zz in range(image.shape[2]):

            # ============
            # rescale the images and labels so that their orientation matches that of the nci dataset
            # ============            
            image2d_rescaled = rescale(np.squeeze(image_normalized[:, :, zz]),
                                                  scale_vector,
                                                  order=1,
                                                  preserve_range=True,
                                                  multichannel=False,
                                                  mode = 'constant')
 
            label2d_rescaled = rescale(np.squeeze(label[:, :, zz]),
                                                  scale_vector,
                                                  order=0,
                                                  preserve_range=True,
                                                  multichannel=False,
                                                  mode='constant')

            # ============            
            # rotate to align with other datasets
            # ============   
            image2d_rescaled_rotated = np.rot90(image2d_rescaled, k=0)
            label2d_rescaled_rotated = np.rot90(label2d_rescaled, k=0)
            
            # ============            
            # crop or pad to make of the same size
            # ============            
            image2d_rescaled_rotated_cropped = utils.crop_or_pad_slice_to_size(image2d_rescaled_rotated, size[0], size[1])
            label2d_rescaled_rotated_cropped = utils.crop_or_pad_slice_to_size(label2d_rescaled_rotated, size[0], size[1])            

            # ============   
            # append to list
            # ============   
            image_list.append(image2d_rescaled_rotated_cropped)
            label_list.append(label2d_rescaled_rotated_cropped)

            write_buffer += 1

            # Writing needs to happen inside the loop over the slices
            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer

                _write_range_to_hdf5(data,
                                     image_list,
                                     label_list,
                                     counter_from,
                                     counter_to)
                
                _release_tmp_memory(image_list,
                                    label_list)

                # update counters 
                counter_from = counter_to
                write_buffer = 0
        
    logging.info('Writing remaining data')
    counter_to = counter_from + write_buffer
    _write_range_to_hdf5(data,
                         image_list,
                         label_list,
                         counter_from,
                         counter_to)
    _release_tmp_memory(image_list,
                        label_list)

    # Write the small datasets
    hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
    hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
    hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
    hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
    hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
    hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
    hdf5_file.create_dataset('patnames', data=np.asarray(pat_names_list, dtype="S10"))
    
    # After test train loop:
    hdf5_file.close()

# ===============================================================
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         img_list,
                         mask_list,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list, dtype=np.float32)
    lab_arr = np.asarray(mask_list, dtype=np.uint8)

    hdf5_data['images'][counter_from : counter_to, ...] = img_arr
    hdf5_data['labels'][counter_from : counter_to, ...] = lab_arr

# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(img_list,
                        mask_list):

    img_list.clear()
    mask_list.clear()
    gc.collect()
    
# ===============================================================
# function to read a single subjects image and labels without any pre-processing
# ===============================================================
def load_without_size_preprocessing(input_folder,
                                    site_name,
                                    idx,
                                    depth):
    
    # ========================    
    # read the filenames
    # ========================
    filenames = sorted(glob.glob(input_folder + site_name + '/*/'))

    # ==================
    # get file paths
    # ==================
    patient_name, image_path, label_path = get_image_and_label_paths(filenames[idx])
    
    # ============
    # read the image and normalize it to be between 0 and 1
    # ============
    image, _, image_hdr = utils.load_nii(image_path)
    image = np.swapaxes(image, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
    
    # ==================
    # read the label file
    # ==================        
    label, _, _ = utils.load_nii(label_path)        
    label = np.swapaxes(label, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
    label = utils.group_segmentation_classes(label) # group the segmentation classes as required
    
    # ============
    # create a segmentation mask and use it to get rid of the skull in the image
    # ============
    label_mask = np.copy(label)
    label_mask[label > 0] = 1
    image = image * label_mask

    # ==================
    # crop out some portion of the image, which are all zeros (rough registration via visual inspection)
    # ==================
    if site_name is 'CALTECH':
        image = image[:, 80:, :]
        label = label[:, 80:, :]
    elif site_name is 'STANFORD':
        image, label = center_image_and_label(image, label)
    
    # ==================
    # crop volume along z axis (as there are several zeros towards the ends)
    # ==================
    image = utils.crop_or_pad_volume_to_size_along_z(image, depth)
    label = utils.crop_or_pad_volume_to_size_along_z(label, depth)     
    
    # ==================
    # normalize the image    
    # ==================
    image = utils.normalise_image(image, norm_type='div_by_max')
    
    return image, label

# ===============================================================
# ===============================================================
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                idx_start,
                                idx_end,
                                protocol,
                                size,
                                depth,
                                target_resolution,
                                force_overwrite=False):

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])
    
    preprocessing_folder = preprocessing_folder + '/'

    data_file_name = 'data_%s_2d_size_%s_depth_%d_res_%s_from_%d_to_%d.hdf5' % (protocol, size_str, depth, res_str, idx_start, idx_end)
    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder,
                     data_file_path,
                     idx_start,
                     idx_end,
                     protocol,
                     size,
                     depth,
                     target_resolution,
                     preprocessing_folder)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')

# ===============================================================
# ===============================================================
if __name__ == '__main__':
    
    #copy_site_files_abide_caltech()

    input_folder = sys_config.orig_data_root_ixi
    preprocessing_folder = sys_config.preproc_folder_ixi

    data_ixi = load_and_maybe_process_data(input_folder,
                                           preprocessing_folder,
                                           idx_start = 0,
                                           idx_end = 2,
                                           protocol = 'T2',
                                           size = (256, 256),
                                           depth = 256, 
                                           target_resolution = (0.7, 0.7),
                                           force_overwrite=True)