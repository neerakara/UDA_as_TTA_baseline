# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import shutil
import tensorflow as tf
import numpy as np
import model as model
import config.system as sys_config
import utils
import data.data_hcp as data_hcp
import data.data_abide as data_abide

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2i_uda as exp_config

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
# ==================================================================
# main function for training
# ==================================================================
def run_uda_training(log_dir,
                     images_sd_tr,
                     labels_sd_tr,
                     images_sd_vl,
                     labels_sd_vl,
                     images_td_tr,
                     images_td_vl):
            
    # ================================================================
    # reset the graph built so far and build a new TF graph
    # ================================================================
    tf.reset_default_graph()
    with tf.Graph().as_default():
        
        # ============================
        # set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(exp_config.run_num_uda)
        np.random.seed(exp_config.run_num_uda)

        # ================================================================
        # create placeholders - segmentation net
        # ================================================================
        images_sd_pl = tf.placeholder(tf.float32, shape = [exp_config.batch_size] + list(exp_config.image_size) + [1], name = 'images_sd')        
        images_td_pl = tf.placeholder(tf.float32, shape = [exp_config.batch_size] + list(exp_config.image_size) + [1], name = 'images_td')   
        labels_sd_pl = tf.placeholder(tf.uint8, shape = [exp_config.batch_size] + list(exp_config.image_size), name = 'labels_sd')
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')

        # ================================================================
        # insert a normalization module in front of the segmentation network
        # ================================================================
        images_sd_normalized, _ = model.normalize(images_sd_pl, exp_config, training_pl, scope_reuse = False)
        images_td_normalized, _ = model.normalize(images_td_pl, exp_config, training_pl, scope_reuse = True)
        
        # ================================================================
        # segmentation network
        # ================================================================
        predicted_seg_sd_logits, _, _ = model.predict_i2l(images_sd_normalized, exp_config, training_pl)
        
        # ================================================================
        # discriminator on the normalized images
        # ================================================================
        d_logits_sd = model.discriminator(images_sd_normalized, exp_config, training_pl, scope_reuse = False) 
        d_logits_td = model.discriminator(images_td_normalized, exp_config, training_pl, scope_reuse = True)
                                        
        # ================================================================
        # add ops for calculation of the discriminator loss 
        # ================================================================
        d_loss_sd = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(d_logits_sd), logits=d_logits_sd)
        d_loss_td = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(d_logits_td), logits=d_logits_td) 
        loss_d_op = tf.reduce_mean(d_loss_sd + d_loss_td)     
        tf.summary.scalar('tr_losses/loss_discriminator', loss_d_op) 

        # ================================================================
        # add ops for calculation of the adversarial loss that tries to get domain invariant features in the normalized image space
        # ================================================================
        loss_g_op = model.loss_invariance(d_logits_td)
        tf.summary.scalar('tr_losses/loss_invariant_features', loss_g_op) 
        
        # ================================================================
        # add ops for calculation of the supervised segmentation loss
        # ================================================================
        loss_seg_op = model.loss(predicted_seg_sd_logits,
                                 labels_sd_pl,
                                 nlabels = exp_config.nlabels,
                                 loss_type = exp_config.loss_type_i2l)        
        tf.summary.scalar('tr_losses/loss_segmentation', loss_seg_op)
        
        # ================================================================
        # total training loss for uda
        # ================================================================
        loss_total_op = loss_seg_op + exp_config.lambda_uda * loss_g_op
        tf.summary.scalar('tr_losses/loss_total_uda', loss_total_op)
        
        # ================================================================
        # merge all summaries
        # ================================================================
        summary = tf.summary.merge_all()
        
        # ================================================================
        # divide the vars into segmentation network, normalization network and the discriminator network
        # ================================================================
        i2l_vars = []
        normalization_vars = []
        discriminator_vars = []
        
        for v in tf.global_variables():
            var_name = v.name        
            if 'image_normalizer' in var_name:
                normalization_vars.append(v)
                i2l_vars.append(v) # the normalization vars also need to be restored from the pre-trained i2l mapper
            elif 'i2l_mapper' in var_name:
                i2l_vars.append(v)
            elif 'discriminator' in var_name:
                discriminator_vars.append(v)
                                
        # ================================================================
        # add optimization ops
        # ================================================================
        train_i2l_op = model.training_step(loss_total_op,
                                           i2l_vars,
                                           exp_config.optimizer_handle,
                                           learning_rate = exp_config.learning_rate)
        
        train_discriminator_op = model.training_step(loss_d_op,
                                                     discriminator_vars,
                                                     exp_config.optimizer_handle,
                                                     learning_rate = exp_config.learning_rate)

        # ================================================================
        # add ops for model evaluation
        # ================================================================
        eval_loss = model.evaluation_i2l(predicted_seg_sd_logits,
                                         labels_sd_pl,
                                         images_sd_pl,
                                         d_logits_td,
                                         nlabels = exp_config.nlabels,
                                         loss_type = exp_config.loss_type_i2l)
                
        # ================================================================
        # build the summary Tensor based on the TF collection of Summaries.
        # ================================================================
        if exp_config.debug: print('creating summary op...')

        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
        
        # ================================================================
        # find if any vars are uninitialized
        # ================================================================
        if exp_config.debug: logging.info('Adding the op to get a list of initialized variables...')
        uninit_vars = tf.report_uninitialized_variables()
        
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # ================================================================
        # create savers
        # ================================================================
        saver = tf.train.Saver(var_list = i2l_vars)
        saver_lowest_loss = tf.train.Saver(var_list = i2l_vars, max_to_keep=3)    
        
        # ================================================================
        # summaries of the validation errors
        # ================================================================
        vl_error_seg = tf.placeholder(tf.float32, shape=[], name='vl_error_seg')
        vl_error_seg_summary = tf.summary.scalar('validation/loss_seg', vl_error_seg)
        vl_dice = tf.placeholder(tf.float32, shape=[], name='vl_dice')
        vl_dice_summary = tf.summary.scalar('validation/dice', vl_dice)
        vl_error_invariance = tf.placeholder(tf.float32, shape=[], name='vl_error_invariance')
        vl_error_invariance_summary = tf.summary.scalar('validation/loss_invariance', vl_error_invariance)
        vl_summary = tf.summary.merge([vl_error_seg_summary, vl_dice_summary, vl_error_invariance_summary])

        # ================================================================
        # summaries of the training errors
        # ================================================================        
        tr_error_seg = tf.placeholder(tf.float32, shape=[], name='tr_error_seg')
        tr_error_seg_summary = tf.summary.scalar('training/loss_seg', tr_error_seg)
        tr_dice = tf.placeholder(tf.float32, shape=[], name='tr_dice')
        tr_dice_summary = tf.summary.scalar('training/dice', tr_dice)
        tr_error_invariance = tf.placeholder(tf.float32, shape=[], name='tr_error_invariance')
        tr_error_invariance_summary = tf.summary.scalar('training/loss_invariance', tr_error_invariance)
        tr_summary = tf.summary.merge([tr_error_seg_summary, tr_dice_summary, tr_error_invariance_summary])
                        
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        if exp_config.debug:
            logging.info('============================================================')
            logging.info('Freezing the graph now!')
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        if exp_config.debug:
            logging.info('============================================================')
            logging.info('initializing all variables...')
        sess.run(init_ops)
        
        # ================================================================
        # print names of uninitialized variables
        # ================================================================
        uninit_variables = sess.run(uninit_vars)
        if exp_config.debug:
            logging.info('============================================================')
            logging.info('This is the list of uninitialized variables:' )
            for v in uninit_variables: print(v)

        # ================================================================
        # Restore the segmentation network parameters and the pre-trained i2i mapper parameters
        # After the adaptation for the 1st TD subject is done, start the adaptation for the subsequent subjects with those parameters
        # ================================================================
        logging.info('============================================================')        
        path_to_model = sys_config.log_root + exp_config.expname_i2l + '/models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_lowest_loss.restore(sess, checkpoint_path)
                               
        # ================================================================
        # run training steps
        # ================================================================
        step = 0
        lowest_loss = 10000.0

        while (step < exp_config.max_steps):
                
            # ================================================               
            # batches
            # ================================================    
            for batch in iterate_minibatches(images_sd = images_sd_tr,
                                             labels_sd = labels_sd_tr,
                                             images_td = images_td_tr,
                                             batch_size = exp_config.batch_size):

                x_sd, y_sd, x_td = batch   
                
                # ===========================
                # define feed dict for this iteration
                # ===========================   
                feed_dict = {images_sd_pl: x_sd,
                             labels_sd_pl: y_sd,
                             images_td_pl: x_td,
                             training_pl: True}
                
                # ================================================     
                # update i2l and D successively
                # ================================================               
                sess.run(train_i2l_op, feed_dict=feed_dict)
                sess.run(train_discriminator_op, feed_dict=feed_dict)
                
                # ================================================  
                # increment step
                # ================================================  
                step += 1
            
                # ===========================
                # write the summaries and print an overview fairly often
                # ===========================
                if (step+1) % exp_config.summary_writing_frequency == 0:                                        
                    summary_str = sess.run(summary, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    
                # ===========================
                # Compute the loss on the entire training set
                # ===========================
                if step % exp_config.train_eval_frequency == 0:
                    logging.info('Training Data Eval:')
                    train_loss_seg, train_dice, train_loss_invariance = do_eval(sess,
                                                                                eval_loss, 
                                                                                images_sd_pl,
                                                                                labels_sd_pl,
                                                                                images_td_pl,
                                                                                training_pl,
                                                                                images_sd_tr,
                                                                                labels_sd_tr,
                                                                                images_td_tr,
                                                                                exp_config.batch_size)
                    tr_summary_msg = sess.run(tr_summary, feed_dict={tr_error_seg: train_loss_seg,
                                                                     tr_dice: train_dice,
                                                                     tr_error_invariance: train_loss_invariance})
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
                    val_loss_seg, val_dice, val_loss_invariance = do_eval(sess,
                                                                          eval_loss,
                                                                          images_sd_pl,
                                                                          labels_sd_pl,
                                                                          images_td_pl,
                                                                          training_pl,
                                                                          images_sd_vl,
                                                                          labels_sd_vl,
                                                                          images_td_vl,
                                                                          exp_config.batch_size)
                    vl_summary_msg = sess.run(vl_summary, feed_dict={vl_error_seg: val_loss_seg,
                                                                     vl_dice: val_dice,
                                                                     vl_error_invariance: val_loss_invariance})                    
                    summary_writer.add_summary(vl_summary_msg, step)

                    # ===========================
                    # save model if the val dice is the best yet
                    # ===========================
                    val_total_loss = val_loss_seg + exp_config.lambda_uda*val_loss_invariance
                    if val_total_loss < lowest_loss:
                        lowest_loss = val_total_loss
                        lowest_loss_file = os.path.join(log_dir, 'models/lowest_loss.ckpt')
                        saver_lowest_loss.save(sess, lowest_loss_file, global_step=step)
                        logging.info('Found new average best loss on validation set at step %d' % step)
                
        # ================================================================    
        # close tf session
        # ================================================================    
        sess.close()

    return 0
        
# ==================================================================
# ==================================================================
def do_eval(sess,
            eval_loss,
            images_sd_placeholder,
            labels_sd_placeholder,
            images_td_placeholder,
            training_time_placeholder,
            images_sd,
            labels_sd,
            images_td,
            batch_size):

    loss_seg_ii = 0
    loss_invar_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in iterate_minibatches(images_sd,
                                     labels_sd,
                                     images_td,
                                     batch_size):

        x_sd, y_sd, x_td = batch
        
        feed_dict = {images_sd_placeholder: x_sd,
                     labels_sd_placeholder: y_sd,
                     images_td_placeholder: x_td,
                     training_time_placeholder: False}
        
        loss_seg, fg_dice, loss_invariance = sess.run(eval_loss, feed_dict=feed_dict)
        
        loss_seg_ii += loss_seg
        loss_invar_ii += loss_invariance
        dice_ii += fg_dice
        num_batches += 1

    avg_loss_seg = loss_seg_ii / num_batches
    avg_loss_invar = loss_invar_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average segmentation loss: %.4f, average dice: %.4f, average invariance loss: %.4f' % (avg_loss_seg, avg_dice, avg_loss_invar))

    return avg_loss_seg, avg_dice, avg_loss_invar

# ==================================================================
# ==================================================================
def iterate_minibatches(images_sd,
                        labels_sd,
                        images_td,
                        batch_size):
        
    images_sd_ = np.copy(images_sd)
    labels_sd_ = np.copy(labels_sd)
    images_td_ = np.copy(images_td)
    
    # generate indices to randomly select subjects in each minibatch
    n_images_sd = images_sd_.shape[0]
    n_images_td = images_td_.shape[0]
    random_indices_sd = np.random.permutation(n_images_sd)

    # generate batches in a for loop
    for b_i in range(n_images_sd // batch_size):
        
        if b_i + batch_size > n_images_sd:
            continue
        
        # extract random sd batch        
        batch_indices_sd = np.sort(random_indices_sd[b_i*batch_size:(b_i+1)*batch_size])        
        x_sd = images_sd_[batch_indices_sd, ...]
        y_sd = labels_sd_[batch_indices_sd, ...]        
        
        # extract random td batch
        random_indices_td = np.random.permutation(n_images_td)
        batch_indices_td = np.sort(random_indices_td[:batch_size])
        x_td = images_td_[batch_indices_td, ...]

        # augment sd batch
        if exp_config.da_ratio > 0:
            x_sd, y_sd = utils.do_data_augmentation(images = x_sd,
                                                    labels = y_sd,
                                                    data_aug_ratio = exp_config.da_ratio)

            x_td, _ = utils.do_data_augmentation(images = x_td,
                                                 labels = y_sd, # these will not be used, passing the sd labels so that the same function can be used..
                                                 data_aug_ratio = exp_config.da_ratio)

        x_sd = np.expand_dims(x_sd, axis=-1)        
        x_td = np.expand_dims(x_td, axis=-1)

        yield x_sd, y_sd, x_td

# ==================================================================
# ==================================================================
def main():
        
    # ============================
    # Load SD data
    # ============================   
    logging.info('============================================================')
    logging.info('Loading SD data...')
    if exp_config.train_dataset is 'HCPT1':
        logging.info('Reading HCPT1 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        data_brain_train_sd = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                                preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                idx_start = 0,
                                                                idx_end = 20,             
                                                                protocol = 'T1',
                                                                size = exp_config.image_size,
                                                                depth = exp_config.image_depth_hcp,
                                                                target_resolution = exp_config.target_resolution_brain)
        imtr_sd, gttr_sd = [ data_brain_train_sd['images'], data_brain_train_sd['labels'] ]
        
        data_brain_val_sd = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                              preprocessing_folder = sys_config.preproc_folder_hcp,
                                                              idx_start = 20,
                                                              idx_end = 25,             
                                                              protocol = 'T1',
                                                              size = exp_config.image_size,
                                                              depth = exp_config.image_depth_hcp,
                                                              target_resolution = exp_config.target_resolution_brain)
        imvl_sd, gtvl_sd = [ data_brain_val_sd['images'], data_brain_val_sd['labels'] ]
        
    # ============================
    # Load TD unlabelled images
    # ============================   
    logging.info('============================================================')
    logging.info('Loading TD unlabelled images...')    
    if exp_config.test_dataset is 'HCPT2':
        logging.info('Reading HCPT2 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        image_depth = exp_config.image_depth_hcp
        data_brain_train_td = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                               preprocessing_folder = sys_config.preproc_folder_hcp,
                                                               idx_start = 0,
                                                               idx_end = 20,
                                                               protocol = 'T2',
                                                               size = exp_config.image_size,
                                                               depth = image_depth,
                                                               target_resolution = exp_config.target_resolution_brain)
        imtr_td = data_brain_train_td['images']
        
        data_brain_val_td = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                                 preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                 idx_start = 20,
                                                                 idx_end = 25,
                                                                 protocol = 'T2',
                                                                 size = exp_config.image_size,
                                                                 depth = image_depth,
                                                                 target_resolution = exp_config.target_resolution_brain)
        imvl_td = data_brain_val_td['images']
        
    elif exp_config.test_dataset is 'CALTECH':
        logging.info('Reading CALTECH images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'CALTECH/')
        image_depth = exp_config.image_depth_caltech
        data_brain_train_td = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                 preprocessing_folder = sys_config.preproc_folder_abide,
                                                                 site_name = 'CALTECH',
                                                                 idx_start = 0,
                                                                 idx_end = 10,
                                                                 protocol = 'T1',
                                                                 size = exp_config.image_size,
                                                                 depth = image_depth,
                                                                 target_resolution = exp_config.target_resolution_brain)        
        imtr_td = data_brain_train_td['images']
        
        data_brain_val_td = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                   preprocessing_folder = sys_config.preproc_folder_abide,
                                                                   site_name = 'CALTECH',
                                                                   idx_start = 10,
                                                                   idx_end = 15,             
                                                                   protocol = 'T1',
                                                                   size = exp_config.image_size,
                                                                   depth = exp_config.image_depth_caltech,
                                                                   target_resolution = exp_config.target_resolution_brain)
        imvl_td = data_brain_val_td['images']
    
    # ================================================================
    # create a text file for writing results
    # results of individual subjects will be appended to this file
    # ================================================================
    log_dir_uda = os.path.join(sys_config.log_root, exp_config.expname_uda)
    if not tf.gfile.Exists(log_dir_uda):
        tf.gfile.MakeDirs(log_dir_uda)
        tf.gfile.MakeDirs(log_dir_uda + '/models')

    # ===========================
    # Copy experiment config file
    # ===========================
    shutil.copy(exp_config.__file__, log_dir_uda)

    # ================================================================
    # run uda training 
    # ================================================================                        
    run_uda_training(log_dir_uda,
                     imtr_sd,
                     gttr_sd,
                     imvl_sd,
                     gtvl_sd,
                     imtr_td,
                     imvl_td)    

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()