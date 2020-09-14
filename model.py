import tensorflow as tf
from tfwrapper import losses
import matplotlib
import matplotlib.cm

# ================================================================
# ================================================================
def predict_i2l(images,
                exp_config,
                training_pl):
    '''
    Returns the prediction for an image given a network from the model zoo
    :param images: An input image tensor
    :param inference_handle: A model function from the model zoo
    :return: A prediction mask, and the corresponding softmax output
    '''

    logits = exp_config.model_handle_i2l(images,
                                         nlabels = exp_config.nlabels,
                                         training_pl = training_pl)
    
    softmax = tf.nn.softmax(logits)
    mask = tf.argmax(softmax, axis=-1)

    return logits, softmax, mask

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
# ================================================================
def discriminator(images,
                  exp_config,
                  training_pl,
                  scope_reuse = False):
    
    logits = exp_config.model_handle_discriminator(images,
                                                   training_pl,
                                                   scope_reuse)
    
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
    '''
    Creates the optimisation operation which is executed in each training iteration of the network
    :param loss: The loss to be minimised
    :var_list: list of params that this loss should be optimized wrt.
    :param optimizer_handle: A handle to one of the tf optimisers 
    :param learning_rate: Learning rate
    :return: The training operation
    '''

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
def evaluation_i2l(logits_sd,
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
    
    # =================
    # write some segmentations to tensorboard
    # =================
    mask = tf.argmax(tf.nn.softmax(logits_sd, axis=-1), axis=-1)
    mask_gt = labels_sd
    
    gt1 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=0, nlabels=nlabels)
    gt2 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=1, nlabels=nlabels)
    gt3 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=2, nlabels=nlabels)
    
    pred1 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=0, nlabels=nlabels)
    pred2 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=1, nlabels=nlabels)
    pred3 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=2, nlabels=nlabels)
    
    img1 = prepare_tensor_for_summary(images_sd, mode='image', n_idx_batch=0, nlabels=nlabels)
    img2 = prepare_tensor_for_summary(images_sd, mode='image', n_idx_batch=1, nlabels=nlabels)
    img3 = prepare_tensor_for_summary(images_sd, mode='image', n_idx_batch=2, nlabels=nlabels)
    
    tf.summary.image('example_labels_true', tf.concat([gt1, gt2, gt3], axis=0))
    tf.summary.image('example_labels_pred', tf.concat([pred1, pred2, pred3], axis=0))
    tf.summary.image('example_images', tf.concat([img1, img2, img3], axis=0))

    return supervised_loss, mean_dice, invariance_loss

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
