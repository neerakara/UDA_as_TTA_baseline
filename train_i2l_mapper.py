# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import utils
import model as model
import config.system as sys_config

import data.data_hcp as data_hcp
import data.data_ixi as data_ixi
import data.data_abide as data_abide

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2l as exp_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name_i2l)
logging.info('Logging directory: %s' %log_dir)

# ==================================================================
# main function for training
# ==================================================================
def run_training(continue_run):

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name_i2l)

    # ============================
    # Initialize step number - this is number of mini-batch runs
    # ============================
    init_step = 0

    # ============================
    # if continue_run is set to True, load the model parameters saved earlier
    # else start training from scratch
    # ============================
    if continue_run:
        logging.info('============================================================')
        logging.info('Continuing previous run')
        try:
            init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'models/model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 as otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('Did not find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0
        logging.info('============================================================')

    # ============================
    # Load data
    # ============================   
    logging.info('============================================================')
    logging.info('Loading data...')
    if exp_config.train_dataset is 'HCPT1':
        logging.info('Reading HCPT1 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        data_brain_train = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                                preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                idx_start = 0,
                                                                idx_end = 20,             
                                                                protocol = 'T1',
                                                                size = exp_config.image_size,
                                                                depth = exp_config.image_depth_hcp,
                                                                target_resolution = exp_config.target_resolution_brain)
        imtr, gttr = [ data_brain_train['images'], data_brain_train['labels'] ]
        
        data_brain_val = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                              preprocessing_folder = sys_config.preproc_folder_hcp,
                                                              idx_start = 20,
                                                              idx_end = 25,             
                                                              protocol = 'T1',
                                                              size = exp_config.image_size,
                                                              depth = exp_config.image_depth_hcp,
                                                              target_resolution = exp_config.target_resolution_brain)
        imvl, gtvl = [ data_brain_val['images'], data_brain_val['labels'] ]
        
    elif exp_config.train_dataset is 'HCPT2':
        logging.info('Reading HCPT2 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        data_brain_train = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                                preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                idx_start = 0,
                                                                idx_end = 20,             
                                                                protocol = 'T2',
                                                                size = exp_config.image_size,
                                                                depth = exp_config.image_depth_hcp,
                                                                target_resolution = exp_config.target_resolution_brain)
        imtr, gttr = [ data_brain_train['images'], data_brain_train['labels'] ]
        
        data_brain_val = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                              preprocessing_folder = sys_config.preproc_folder_hcp,
                                                              idx_start = 20,
                                                              idx_end = 25,             
                                                              protocol = 'T2',
                                                              size = exp_config.image_size,
                                                              depth = exp_config.image_depth_hcp,
                                                              target_resolution = exp_config.target_resolution_brain)
        imvl, gtvl = [ data_brain_val['images'], data_brain_val['labels'] ]
        
    elif exp_config.train_dataset is 'CALTECH':
        logging.info('Reading CALTECH images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'CALTECH/')      
        data_brain_train = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                  preprocessing_folder = sys_config.preproc_folder_abide,
                                                                  site_name = 'CALTECH',
                                                                  idx_start = 0,
                                                                  idx_end = 10,             
                                                                  protocol = 'T1',
                                                                  size = exp_config.image_size,
                                                                  depth = exp_config.image_depth_caltech,
                                                                  target_resolution = exp_config.target_resolution_brain)
        imtr, gttr = [ data_brain_train['images'], data_brain_train['labels'] ]
        
        data_brain_val = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                preprocessing_folder = sys_config.preproc_folder_abide,
                                                                site_name = 'CALTECH',
                                                                idx_start = 10,
                                                                idx_end = 15,             
                                                                protocol = 'T1',
                                                                size = exp_config.image_size,
                                                                depth = exp_config.image_depth_caltech,
                                                                target_resolution = exp_config.target_resolution_brain)
        imvl, gtvl = [ data_brain_val['images'], data_brain_val['labels'] ]
        
    elif exp_config.train_dataset is 'STANFORD':
        logging.info('Reading STANFORD images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'STANFORD/')      
        data_brain_train = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                  preprocessing_folder = sys_config.preproc_folder_abide,
                                                                  site_name = 'STANFORD',
                                                                  idx_start = 0,
                                                                  idx_end = 10,             
                                                                  protocol = 'T1',
                                                                  size = exp_config.image_size,
                                                                  depth = exp_config.image_depth_stanford,
                                                                  target_resolution = exp_config.target_resolution_brain)
        imtr, gttr = [ data_brain_train['images'], data_brain_train['labels'] ]
        
        data_brain_val = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                preprocessing_folder = sys_config.preproc_folder_abide,
                                                                site_name = 'STANFORD',
                                                                idx_start = 10,
                                                                idx_end = 15,             
                                                                protocol = 'T1',
                                                                size = exp_config.image_size,
                                                                depth = exp_config.image_depth_stanford,
                                                                target_resolution = exp_config.target_resolution_brain)
        imvl, gtvl = [ data_brain_val['images'], data_brain_val['labels'] ]
        
    elif exp_config.train_dataset is 'IXI':
        logging.info('Reading IXI images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_ixi)      
        data_brain_train = data_ixi.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_ixi,
                                                                preprocessing_folder = sys_config.preproc_folder_ixi,
                                                                idx_start = 0,
                                                                idx_end = 12,             
                                                                protocol = 'T2',
                                                                size = exp_config.image_size,
                                                                depth = exp_config.image_depth_ixi,
                                                                target_resolution = exp_config.target_resolution_brain)
        imtr, gttr = [ data_brain_train['images'], data_brain_train['labels'] ]
        
        data_brain_val = data_ixi.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_ixi,
                                                              preprocessing_folder = sys_config.preproc_folder_ixi,
                                                              idx_start = 12,
                                                              idx_end = 17,             
                                                              protocol = 'T2',
                                                              size = exp_config.image_size,
                                                              depth = exp_config.image_depth_ixi,
                                                              target_resolution = exp_config.target_resolution_brain)
        imvl, gtvl = [ data_brain_val['images'], data_brain_val['labels'] ]
        
    logging.info('Training Images: %s' %str(imtr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Training Labels: %s' %str(gttr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Validation Images: %s' %str(imvl.shape))
    logging.info('Validation Labels: %s' %str(gtvl.shape))
    logging.info('============================================================')
                
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ============================
        # set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(exp_config.run_number)
        np.random.seed(exp_config.run_number)

        # ================================================================
        # create placeholders
        # ================================================================
        logging.info('Creating placeholders...')
        image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [1]
        mask_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size)
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')
        labels_pl = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name = 'labels')
        learning_rate_pl = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')
        
        # ================================================================
        # insert a normalization module in front of the segmentation network
        # the normalization module will be adapted for each test image
        # ================================================================
        images_normalized, _ = model.normalize(images_pl,
                                               exp_config,
                                               training_pl)

        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        logits, _, _ = model.predict_i2l(images_normalized,
                                         exp_config,
                                         training_pl)
        
        print('shape of inputs: ', images_pl.shape) # (batch_size, 256, 256, 1)
        print('shape of logits: ', logits.shape) # (batch_size, 256, 256, nlabels)
        
        # ================================================================
        # create a list of all vars that must be optimized wrt
        # ================================================================
        i2l_vars = []
        for v in tf.trainable_variables():
            i2l_vars.append(v)
        
        # ================================================================
        # add ops for calculation of the supervised training loss
        # ================================================================
        loss_op = model.loss(logits,
                             labels_pl,
                             nlabels=exp_config.nlabels,
                             loss_type=exp_config.loss_type_i2l)        
        tf.summary.scalar('loss', loss_op)
        
        # ================================================================
        # add optimization ops.
        # Create different ops according to the variables that must be trained
        # ================================================================
        print('creating training op...')
        train_op = model.training_step(loss_op,
                                       i2l_vars,
                                       exp_config.optimizer_handle,
                                       learning_rate_pl,
                                       update_bn_nontrainable_vars=True)

        # ================================================================
        # add ops for model evaluation
        # ================================================================
        print('creating eval op...')
        eval_loss = model.evaluation_i2l(logits,
                                         labels_pl,
                                         images_pl,
                                         nlabels = exp_config.nlabels,
                                         loss_type = exp_config.loss_type_i2l)

        # ================================================================
        # build the summary Tensor based on the TF collection of Summaries.
        # ================================================================
        print('creating summary op...')
        summary = tf.summary.merge_all()

        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
        
        # ================================================================
        # find if any vars are uninitialized
        # ================================================================
        logging.info('Adding the op to get a list of initialized variables...')
        uninit_vars = tf.report_uninitialized_variables()

        # ================================================================
        # create saver
        # ================================================================
        saver = tf.train.Saver(max_to_keep=10)
        saver_best_dice = tf.train.Saver(max_to_keep=3)
        
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # ================================================================
        # summaries of the validation errors
        # ================================================================
        vl_error = tf.placeholder(tf.float32, shape=[], name='vl_error')
        vl_error_summary = tf.summary.scalar('validation/loss', vl_error)
        vl_dice = tf.placeholder(tf.float32, shape=[], name='vl_dice')
        vl_dice_summary = tf.summary.scalar('validation/dice', vl_dice)
        vl_summary = tf.summary.merge([vl_error_summary, vl_dice_summary])

        # ================================================================
        # summaries of the training errors
        # ================================================================        
        tr_error = tf.placeholder(tf.float32, shape=[], name='tr_error')
        tr_error_summary = tf.summary.scalar('training/loss', tr_error)
        tr_dice = tf.placeholder(tf.float32, shape=[], name='tr_dice')
        tr_dice_summary = tf.summary.scalar('training/dice', tr_dice)
        tr_summary = tf.summary.merge([tr_error_summary, tr_dice_summary])
        
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        logging.info('Freezing the graph now!')
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        logging.info('============================================================')
        logging.info('initializing all variables...')
        sess.run(init_ops)

        # ================================================================
        # print names of all variables
        # ================================================================
        logging.info('============================================================')
        logging.info('This is the list of all variables:' )
        for v in tf.trainable_variables(): print(v.name)
        
        # ================================================================
        # print names of uninitialized variables
        # ================================================================
        logging.info('============================================================')
        logging.info('This is the list of uninitialized variables:' )
        uninit_variables = sess.run(uninit_vars)
        for v in uninit_variables: print(v)

        # ================================================================
        # continue run from a saved checkpoint
        # ================================================================
        if continue_run:
            # Restore session
            logging.info('============================================================')
            logging.info('Restroring session from: %s' %init_checkpoint_path)
            saver.restore(sess, init_checkpoint_path)

        # ================================================================
        # ================================================================        
        step = init_step
        curr_lr = exp_config.learning_rate
        best_dice = 0

        # ================================================================
        # run training epochs
        # ================================================================
        while (step < exp_config.max_steps):

            if step % 1000 is 0:
                logging.info('============================================================')
                logging.info('step %d' % step)
        
            # ================================================               
            # batches
            # ================================================            
            for batch in iterate_minibatches(imtr,
                                             gttr,
                                             batch_size = exp_config.batch_size,
                                             train_or_eval = 'train'):
                
                curr_lr = exp_config.learning_rate
                start_time = time.time()
                x, y = batch

                # ===========================
                # avoid incomplete batches
                # ===========================
                if y.shape[0] < exp_config.batch_size:
                    step += 1
                    continue
                
                # ===========================
                # create the feed dict for this training iteration
                # ===========================
                feed_dict = {images_pl: x,
                             labels_pl: y,
                             learning_rate_pl: curr_lr,
                             training_pl: True}
                
                # ===========================
                # opt step
                # ===========================
                _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)

                # ===========================
                # compute the time for this mini-batch computation
                # ===========================
                duration = time.time() - start_time

                # ===========================
                # write the summaries and print an overview fairly often
                # ===========================
                if (step+1) % exp_config.summary_writing_frequency == 0:                    
                    logging.info('Step %d: loss = %.3f (%.3f sec for the last step)' % (step+1, loss, duration))
                    
                    # ===========================
                    # Update the events file
                    # ===========================
                    summary_str = sess.run(summary, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # ===========================
                # Compute the loss on the entire training set
                # ===========================
                if step % exp_config.train_eval_frequency == 0:
                    logging.info('Training Data Eval:')
                    train_loss, train_dice = do_eval(sess,
                                                     eval_loss,
                                                     images_pl,
                                                     labels_pl,
                                                     training_pl,
                                                     imtr,
                                                     gttr,
                                                     exp_config.batch_size)                    
                    
                    tr_summary_msg = sess.run(tr_summary,
                                              feed_dict={tr_error: train_loss,
                                                         tr_dice: train_dice})
                    
                    summary_writer.add_summary(tr_summary_msg, step)
                    
                # ===========================
                # Save a checkpoint periodically
                # ===========================
                if step % exp_config.save_frequency == 0:
                    checkpoint_file = os.path.join(log_dir, 'models/model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

                # ===========================
                # Evaluate the model periodically on a validation set 
                # ===========================
                if step % exp_config.val_eval_frequency == 0:
                    logging.info('Validation Data Eval:')
                    val_loss, val_dice = do_eval(sess,
                                                 eval_loss,
                                                 images_pl,
                                                 labels_pl,
                                                 training_pl,
                                                 imvl,
                                                 gtvl,
                                                 exp_config.batch_size)                    
                    
                    vl_summary_msg = sess.run(vl_summary,
                                              feed_dict={vl_error: val_loss,
                                                         vl_dice: val_dice})
                    
                    summary_writer.add_summary(vl_summary_msg, step)

                    # ===========================
                    # save model if the val dice is the best yet
                    # ===========================
                    if val_dice > best_dice:
                        best_dice = val_dice
                        best_file = os.path.join(log_dir, 'models/best_dice.ckpt')
                        saver_best_dice.save(sess, best_file, global_step=step)
                        logging.info('Found new average best dice on validation sets! - %f -  Saving model.' % val_dice)

                # ================================================  
                # increment step
                # ================================================  
                step += 1
                
        sess.close()

# ==================================================================
# ==================================================================
def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in iterate_minibatches(images,
                                     labels,
                                     batch_size,
                                     train_or_eval = 'eval'):

        x, y = batch

        if y.shape[0] < batch_size:
            continue
        
        feed_dict = {images_placeholder: x,
                     labels_placeholder: y,
                     training_time_placeholder: False}
        
        loss, fg_dice = sess.run(eval_loss, feed_dict=feed_dict)
        
        loss_ii += loss
        dice_ii += fg_dice
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average segmentation loss: %.4f, average dice: %.4f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice

# ==================================================================
# ==================================================================
def iterate_minibatches(images,
                        labels,
                        batch_size,
                        train_or_eval = 'train'):

    # ===========================
    # generate indices to randomly select subjects in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.random.permutation(n_images)

    # ===========================
    for b_i in range(n_images // batch_size):

        if b_i + batch_size > n_images:
            continue
        
        batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])
        
        x = images[batch_indices, ...]
        y = labels[batch_indices, ...]

        # ===========================    
        # data augmentation (contrast changes + random elastic deformations)
        # ===========================      
        if exp_config.da_ratio > 0:

            # ===========================    
            # doing data aug both during training as well as during evaluation on the validation set (used for model selection)
            # ===========================                  
            if train_or_eval is 'train' or train_or_eval is 'eval':
            # if train_or_eval is 'train':
                x, y = utils.do_data_augmentation(images = x,
                                                  labels = y,
                                                  data_aug_ratio = exp_config.da_ratio)

        x = np.expand_dims(x, axis=-1)
        
        yield x, y
        
# ==================================================================
# ==================================================================
def main():
    
    # ===========================
    # Create dir if it does not exist
    # ===========================
    continue_run = exp_config.continue_run
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        tf.gfile.MakeDirs(log_dir + '/models')
        continue_run = False

    # ===========================
    # Copy experiment config file
    # ===========================
    shutil.copy(exp_config.__file__, log_dir)

    run_training(continue_run)

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
