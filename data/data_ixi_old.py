import os
import glob
import numpy as np
import logging
import utils
import image_utils
import config.system as sys_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ===============================================================     
# This function unzips and pre-processes the data if this has not already been done.
# If this already been done, it reads the processed data and returns it.                    
# ===============================================================                         
def load_data(input_folder,
              preproc_folder,
              idx_start,
              idx_end,
              force_overwrite = False):
    
    # create the pre-processing folder, if it does not exist
    utils.makefolder(preproc_folder)    
    
    logging.info('============================================================')
    logging.info('Loading data...')
        
    # make appropriate filenames according to the requested indices of training, validation and test images
    config_details = 'from%dto%d_' % (idx_start, idx_end)
    filepath_images = preproc_folder + config_details + 'images.npy'
    filepath_masks = preproc_folder + config_details + 'annotations15.npy'
    filepath_affine = preproc_folder + config_details + 'affines.npy'
    filepath_patnames = preproc_folder + config_details + 'patnames.npy'
    
    # if the images have not already been extracted, do so
    if not os.path.exists(filepath_images) or force_overwrite:
        logging.info('This configuration of protocol and data indices has not yet been preprocessed')
        logging.info('Preprocessing now...')
        images, masks, affines, patnames = prepare_data(input_folder,
                                                        preproc_folder,
                                                        idx_start,
                                                        idx_end)
    else:
        logging.info('Already preprocessed this configuration. Loading now...')
        # read from already created npy files
        images = np.load(filepath_images)
#         reading hist eq images
#        images = np.load('/usr/bmicnas01/data-biwi-01/nkarani/projects/hcp_segmentation/data/preproc_data/ixi/from18to38_images_histeq5.npy')
#        images = np.reshape(images,[-1,images.shape[2], images.shape[3]])
        masks = np.load(filepath_masks)
        affines = np.load(filepath_affine)
        patnames = np.load(filepath_patnames)
        
    return images, masks, affines, patnames


# ===============================================================
# Main function that prepares a dataset from the raw challenge data to an hdf5 dataset.
# Extract the required files from their zipped directories
# ===============================================================
def prepare_data(input_folder,
                 preproc_folder,
                 idx_start,
                 idx_end):
    
    images = []
    affines = []
    patnames = []
    masks = []
        
    # read the filenames which have segmentations available
    filenames = sorted(glob.glob(input_folder + '*_seg.nii'))
    logging.info('Number of images in the dataset that have ground truth annotations: %s' % str(len(filenames)))
        
    # iterate through all indices
    for idx in range(len(filenames)):
        
        # only consider images within the indices requested
        if (idx < idx_start) or (idx >= idx_end):
            #logging.info('skipping subject: %d' %idx)
            continue
        
        logging.info('==============================================')
        
        # get the name of the ground truth annotation for this subject
        filename_seg = filenames[idx]
        filename_img = filename_seg[:-8]+'.nii.gz'
        _patname = filename_seg[filename_seg[:-1].rfind('/') + 1 : -8]
        
        if _patname == 'IXI014-HH-1236-T2': # this subject has very poor resolution - 256x256x28
            continue
        
        # read the image
        logging.info('reading image: %s' % _patname)
        _img_data, _img_affine, _img_header = utils.load_nii(filename_img)            
        # make all the images of the same size by appending zero slices to facilitate appending
        # most images are of the size 256*256*130
        if (_img_data.shape[2] is not 130):
            num_zero_slices = 130-_img_data.shape[2]
            zero_slices = np.zeros((_img_data.shape[0], _img_data.shape[1], num_zero_slices))
            _img_data = np.concatenate((_img_data, zero_slices), axis=-1)
        # normalise the image
        _img_data = image_utils.normalise_image(_img_data, norm_type='div_by_max')
        # save the pre-processed image
        utils.makefolder(preproc_folder + _patname)
        savepath = preproc_folder + _patname + '/preprocessed_image.nii'
        utils.save_nii(savepath, _img_data, _img_affine)
        # append to the list of all images, affines and patient names
        images.append(_img_data)
        affines.append(_img_affine)
        patnames.append(_patname)

        # read the segmentation mask (already grouped)
        _seg_data, _seg_affine, _seg_header = utils.load_nii(filename_seg)
        # make all the images of the same size by appending zero slices to facilitate appending
        # most images are of the size 256*256*130
        if (_seg_data.shape[2] is not 130):
            num_zero_slices = 130-_seg_data.shape[2]
            zero_slices = np.zeros((_seg_data.shape[0], _seg_data.shape[1], num_zero_slices))
            _seg_data = np.concatenate((_seg_data, zero_slices), axis=-1)
        # save the pre-processed segmentation ground truth
        utils.makefolder(preproc_folder + _patname)
        savepath = preproc_folder + _patname + '/preprocessed_gt15.nii'
        utils.save_nii(savepath, _seg_data, _seg_affine)
        # append to the list of all masks
        masks.append(_seg_data)

    # convert the lists to arrays
    images = np.array(images)
    affines = np.array(affines)
    patnames = np.array(patnames)
    masks = np.array(masks, dtype = 'uint8')
        
    # merge along the y-zis to get a stack of x-z slices, for the images as well as the masks
    images = images.swapaxes(1,2)
    images = images.reshape(-1,images.shape[2], images.shape[3])
    masks = masks.swapaxes(1,2)
    masks = masks.reshape(-1,masks.shape[2], masks.shape[3])
    
    # save the processed images and masks so that they can be directly read the next time
    # make appropriate filenames according to the requested indices of training, validation and test images
    logging.info('Saving pre-processed files...')
    config_details = 'from%dto%d_' % (idx_start, idx_end)
    
    filepath_images = preproc_folder + config_details + 'images.npy'
    filepath_masks = preproc_folder + config_details + 'annotations15.npy'
    filepath_affine = preproc_folder + config_details + 'affines.npy'
    filepath_patnames = preproc_folder + config_details + 'patnames.npy'
    
    np.save(filepath_images, images)
    np.save(filepath_masks, masks)
    np.save(filepath_affine, affines)
    np.save(filepath_patnames, patnames)
                      
    return images, masks, affines, patnames


## ===============================================================
## Main function that runs if this file is run directly
## ===============================================================
if __name__ == '__main__':

    #copy_site_files()

#    i, g, _, _ = load_data(sys_config.orig_data_root_ixi, sys_config.preproc_folder_ixi, 10, 12, force_overwrite = True) # train    
#    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
    i, g, _, _ = load_data(sys_config.orig_data_root_ixi, sys_config.preproc_folder_ixi, 16, 18, force_overwrite = True) # validation
    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
#    i, g, _, _ = load_data(sys_config.orig_data_root_ixi, sys_config.preproc_folder_ixi, 18, 38, force_overwrite = True) # test
#    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
#    i, g, _, _ = load_data(sys_config.orig_data_root_ixi, sys_config.preproc_folder_ixi, 0, 16, force_overwrite = True) # train from scratch
#    logging.info('%s, %s' %(str(i.shape), str(g.shape)))
        
    
# ===============================================================
# End of file
# ===============================================================