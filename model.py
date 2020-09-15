import tensorflow as tf
from tfwrapper import losses, layers
import matplotlib
import matplotlib.cm

# ================================================================
# get the logits from the network
# also return softmax and final segmentations
# ================================================================
def predict_i2l(images,
                exp_config,
                training_pl,
                scope_reuse = False):

    logits = exp_config.model_handle_i2l(images,
                                         nlabels = exp_config.nlabels,
                                         training_pl = training_pl,
                                         scope_reuse = scope_reuse)[-1]
    
    softmax = tf.nn.softmax(logits)
    mask = tf.argmax(softmax, axis=-1)

    return logits, softmax, mask

# ================================================================
# get features from all levels
# ================================================================
def get_all_features(images,
                     exp_config,
                     scope_reuse = False):

    features = exp_config.model_handle_i2l(images,
                                           nlabels = exp_config.nlabels,
                                           training_pl = False,
                                           scope_reuse = scope_reuse)[:-1]
    
    return features
    
# ================================================================
# resize features
# ================================================================
def resize_features(features, size, name):
        
    for f in range(len(features)):
        
        this_feature = features[f]
        this_feature_resized = layers.bilinear_upsample2D(this_feature,
                                                          size,
                                                          name + str(f))
        if f is 0:
            features_resized = this_feature_resized
        else:
            features_resized = tf.concat((features_resized,
                                          this_feature_resized), axis=-1)
            
    return features_resized

# ================================================================
# ================================================================
def normalize(images,
              exp_config,
              training_pl,
              scope_reuse = False):
    
    images_normalized, added_residual = exp_config.model_handle_normalizer(images,
                                                                           exp_config,
                                                                           training_pl,
                                                                           scope_reuse)
    
    return images_normalized, added_residual

# ================================================================
# image to image transformation wrapper
# ================================================================
def transform_images(images,
                     exp_config,
                     training_pl,
                     scope_name = 'generator',
                     scope_reuse = False):

    transformed_images = exp_config.model_handle_generator(images,
                                                           training_pl = training_pl,
                                                           scope_name = scope_name,
                                                           scope_reuse = scope_reuse)
    
    return transformed_images

# ================================================================
# ================================================================
def discriminator(images,
                  exp_config,
                  training_pl,
                  scope_name = 'discriminator',
                  scope_reuse = False):
    
    logits = exp_config.model_handle_discriminator(images,
                                                   training_pl,
                                                   scope_name = scope_name,
                                                   scope_reuse = scope_reuse)
    
    return logits
    
# ================================================================
# ================================================================
def loss(logits,
         labels,
         nlabels,
         loss_type,
         mask_for_loss_within_mask = None,
         are_labels_1hot = False):
    '''
    Loss to be minimised by the neural network
    :param logits: The output of the neural network before the softmax
    :param labels: The ground truth labels in standard (i.e. not one-hot) format
    :param nlabels: The number of GT labels
    :param loss_type: Can be 'crossentropy'/'dice'/
    :return: The segmentation
    '''

    if are_labels_1hot is False:
        labels = tf.one_hot(labels, depth=nlabels)

    if loss_type == 'crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels)
        
    elif loss_type == 'crossentropy_reverse':
        predicted_probabilities = tf.nn.softmax(logits)
        segmentation_loss = losses.pixel_wise_cross_entropy_loss_using_probs(predicted_probabilities, labels)
        
    elif loss_type == 'dice':
        segmentation_loss = losses.dice_loss(logits, labels)
        
    elif loss_type == 'dice_within_mask':
        if mask_for_loss_within_mask is not None:
            segmentation_loss = losses.dice_loss_within_mask(logits, labels, mask_for_loss_within_mask)

    else:
        raise ValueError('Unknown loss: %s' % loss_type)

    return segmentation_loss

# ================================================================
# loss the encourages feature invariance
# ================================================================
def loss_invariance(d_logits_td):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(d_logits_td), logits = d_logits_td))

# ================================================================
# ================================================================
def training_step(loss,
                  var_list,
                  optimizer_handle,
                  learning_rate):
    
    optimizer = optimizer_handle(learning_rate = learning_rate) 
    train_op = optimizer.minimize(loss, var_list = var_list)  
    opt_memory_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([train_op, opt_memory_update_ops])

    return train_op

# ================================================================
# ================================================================
def evaluate_losses(logits,
                    labels,
                    nlabels,
                    loss_type,
                    are_labels_1hot = False):
    '''
    A function to compute various loss measures to compare the predicted and ground truth annotations
    '''
    
    # =================
    # supervised loss that is being optimized
    # =================
    supervised_loss = loss(logits = logits,
                           labels = labels,
                           nlabels = nlabels,
                           loss_type = loss_type,
                           are_labels_1hot = are_labels_1hot)

    # =================
    # per-structure dice for each label
    # =================
    if are_labels_1hot is False:
        labels = tf.one_hot(labels, depth=nlabels)
        
    dice_all_imgs_all_labels, mean_dice, mean_dice_fg = losses.compute_dice(logits, labels)
    
    return supervised_loss, dice_all_imgs_all_labels, mean_dice, mean_dice_fg

# ================================================================
# ================================================================
def evaluation_i2l_uda_invariant_features(logits_sd,
                                          labels_sd,
                                          images_sd,
                                          d_logits_td,
                                          nlabels,
                                          loss_type):

    # =================
    # compute segmentation loss and foreground dice
    # =================
    supervised_loss, dice_all_imgs_all_labels, mean_dice, mean_dice_fg = evaluate_losses(logits_sd,
                                                                                         labels_sd,
                                                                                         nlabels,
                                                                                         loss_type)
    # =================
    # compute feature invariance loss
    # =================
    invariance_loss = loss_invariance(d_logits_td)
    
    return supervised_loss, mean_dice, invariance_loss

# ================================================================
# ================================================================
def write_image_summary_uda_invariant_features(logits_sd,
                                               labels_sd,
                                               images_sd,
                                               nlabels):  
    # =================
    # write some segmentations to tensorboard
    # =================
    mask = tf.argmax(tf.nn.softmax(logits_sd, axis=-1), axis=-1)
    mask_gt = labels_sd
    
    gt = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=0, nlabels=nlabels)    
    pr = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=0, nlabels=nlabels)
    im = prepare_tensor_for_summary(images_sd, mode='image', n_idx_batch=0, nlabels=nlabels)
    
    return tf.summary.merge([tf.summary.image('label_true', gt),
                             tf.summary.image('label_pred', pr),
                             tf.summary.image('image_sd', im)])

# ================================================================
# ================================================================
def evaluation_i2l_uda_cycle_gan(logits_sd,
                                 labels_sd,
                                 images_sd,
                                 images_td,
                                 images_sd_to_td_to_sd,
                                 images_td_to_sd_to_td,
                                 d_logits_sd_fake,
                                 d_logits_td_fake,
                                 nlabels,
                                 loss_type):

    # =================
    # compute segmentation loss and foreground dice
    # =================
    supervised_loss, dice_all_imgs_all_labels, mean_dice, mean_dice_fg = evaluate_losses(logits_sd,
                                                                                         labels_sd,
                                                                                         nlabels,
                                                                                         loss_type)
    # =================
    # compute invariance losses
    # =================
    invariance_loss_sd = loss_invariance(d_logits_sd_fake)
    invariance_loss_td = loss_invariance(d_logits_td_fake)
    
    # =================
    # compute cycle consistency losses
    # =================
    cycle_loss_sd = tf.reduce_mean(tf.abs(images_sd_to_td_to_sd - images_sd))
    cycle_loss_td = tf.reduce_mean(tf.abs(images_td_to_sd_to_td - images_td))
    
    return supervised_loss, mean_dice, invariance_loss_sd, invariance_loss_td, cycle_loss_sd, cycle_loss_td

# ================================================================
# ================================================================
def write_image_summary_uda_cgan(logits_sd,
                                 labels_sd,
                                 images_sd,
                                 images_td,
                                 images_sd_to_td,
                                 images_td_to_sd,
                                 images_sd_to_td_to_sd,
                                 images_td_to_sd_to_td,
                                 nlabels):
    
    # =================
    # write some segmentations to tensorboard
    # =================
    mask = tf.argmax(tf.nn.softmax(logits_sd, axis=-1), axis=-1)
    mask_gt = labels_sd
    
    gt = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=0, nlabels=nlabels)    
    pr = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=0, nlabels=nlabels)
    im_sd = prepare_tensor_for_summary(images_sd, mode='image', n_idx_batch=0, nlabels=nlabels)
    im_td = prepare_tensor_for_summary(images_td, mode='image', n_idx_batch=0, nlabels=nlabels)
    im_sd_td = prepare_tensor_for_summary(images_sd_to_td, mode='image', n_idx_batch=0, nlabels=nlabels)
    im_td_sd = prepare_tensor_for_summary(images_td_to_sd, mode='image', n_idx_batch=0, nlabels=nlabels)
    im_sd_td_sd = prepare_tensor_for_summary(images_sd_to_td_to_sd, mode='image', n_idx_batch=0, nlabels=nlabels)
    im_td_sd_td = prepare_tensor_for_summary(images_td_to_sd_to_td, mode='image', n_idx_batch=0, nlabels=nlabels)
        
    return tf.summary.merge([tf.summary.image('label_true', gt),
                             tf.summary.image('label_pred', pr),
                             tf.summary.image('image_sd', im_sd),
                             tf.summary.image('image_td', im_td),                            
                             tf.summary.image('image_sd2td', im_sd_td),
                             tf.summary.image('image_td2sd', im_td_sd),
                             tf.summary.image('image_sd2td2sd', im_sd_td_sd),
                             tf.summary.image('image_td2sd2td', im_td_sd_td)])

# ================================================================
# ================================================================
def prepare_tensor_for_summary(img,
                               mode,
                               n_idx_batch=0,
                               n_idx_z=60,
                               nlabels=None):
    '''
    Format a tensor containing imgaes or segmentation masks such that it can be used with
    tf.summary.image(...) and displayed in tensorboard. 
    :param img: Input image or segmentation mask
    :param mode: Can be either 'image' or 'mask. The two require slightly different slicing
    :param idx: Which index of a minibatch to display. By default it's always the first
    :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label.. 
    :return: Tensor ready to be used with tf.summary.image(...)
    '''

    if mode == 'mask':
        if img.get_shape().ndims == 3:
            V = tf.slice(img, (n_idx_batch, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (n_idx_batch, n_idx_z, 0, 0), (1, 1, -1, -1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (n_idx_batch, 0, 0, n_idx_z, 0), (1, -1, -1, 1, 1))
        else: raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    elif mode == 'image':
        if img.get_shape().ndims == 3:
            V = tf.slice(img, (n_idx_batch, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (n_idx_batch, 0, 0, 0), (1, -1, -1, 1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (n_idx_batch, 0, 0, n_idx_z, 0), (1, -1, -1, 1, 1))
        else: raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    else: raise ValueError('Unknown mode: %s. Must be image or mask' % mode)

    if mode=='image' or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= (nlabels - 1)  # The largest value in a label map is nlabels - 1.

    V *= 255
    V = tf.cast(V, dtype=tf.uint8) # (1,224,224)
    V = tf.squeeze(V)
    V = tf.expand_dims(V, axis=0)
    
    # gather
    if mode == 'mask':
        cmap = 'viridis'
        cm = matplotlib.cm.get_cmap(cmap)
        colors = tf.constant(cm.colors, dtype=tf.float32)
        V = tf.gather(colors, tf.cast(V, dtype=tf.int32)) # (1,224,224,3)
        
    elif mode == 'image':
        V = tf.reshape(V, tf.stack((-1, tf.shape(img)[1], tf.shape(img)[2], 1))) # (1,224,224,1)
    
    return V