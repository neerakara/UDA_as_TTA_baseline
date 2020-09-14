# ======================================================================
# import stuff
# ======================================================================
import tensorflow as tf
from tfwrapper import layers

# ======================================================================
# 2D Unet for mapping from images to segmentation labels
# ======================================================================
def unet2D_i2l(images,
               nlabels,
               training_pl,
               scope_reuse = False): 

    n0 = 16
    n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0
    
    with tf.variable_scope('i2l_mapper') as scope:
        
        if scope_reuse:
            scope.reuse_variables()
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv2D_layer_bn(x=images, name='conv1_1', num_filters=n1, training = training_pl)
        conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training = training_pl)
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training = training_pl)
        conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training = training_pl)
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training = training_pl)
        conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training = training_pl)
        pool3 = layers.max_pool_layer2d(conv3_1)
    
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv2D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training = training_pl)
        conv4_2 = layers.conv2D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv3 = layers.bilinear_upsample2D(conv4_2, size = (tf.shape(conv3_2)[1],tf.shape(conv3_2)[2]), name='upconv3')
        concat3 = tf.concat([deconv3, conv3_2], axis=-1)        
        conv5_1 = layers.conv2D_layer_bn(x=concat3, name='conv5_1', num_filters=n3, training = training_pl)
        conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n3, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv2 = layers.bilinear_upsample2D(conv5_2, size = (tf.shape(conv2_2)[1],tf.shape(conv2_2)[2]), name='upconv2')
        concat2 = tf.concat([deconv2, conv2_2], axis=-1)        
        conv6_1 = layers.conv2D_layer_bn(x=concat2, name='conv6_1', num_filters=n2, training = training_pl)
        conv6_2 = layers.conv2D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n2, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv1 = layers.bilinear_upsample2D(conv6_2, size = (tf.shape(conv1_2)[1],tf.shape(conv1_2)[2]), name='upconv1')
        concat1 = tf.concat([deconv1, conv1_2], axis=-1)        
        conv7_1 = layers.conv2D_layer_bn(x=concat1, name='conv7_1', num_filters=n1, training = training_pl)
        conv7_2 = layers.conv2D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n1, training = training_pl)
    
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred = layers.conv2D_layer(x=conv7_2, name='pred', num_filters=nlabels, kernel_size=1)

    return pool1, pool2, pool3, conv4_2, conv5_2, conv6_2, conv7_2, pred

# ======================================================================
# normalization network
# ======================================================================
def net2D_i2i(images,
              exp_config,
              training_pl,
              scope_reuse = False):
        
    with tf.variable_scope('image_normalizer') as scope:       
        
        if scope_reuse:
            scope.reuse_variables()
                
        num_layers = exp_config.norm_num_hidden_layers
        n1 = exp_config.norm_num_filters_per_layer
        k = exp_config.norm_kernel_size
        
        out = images
        
        for l in range(num_layers):
            out = tf.layers.conv2d(inputs=out,
                                   filters=n1,
                                   kernel_size=k,
                                   padding='SAME',
                                   name='norm_conv1_'+str(l+1),
                                   use_bias=True,
                                   activation=None)
            
            if exp_config.norm_batch_norm is True:
                out = tf.layers.batch_normalization(inputs=out, name = 'norm_conv1_' + str(l+1) + '_bn', training = training_pl)
            
            if exp_config.norm_activation is 'elu':
                out = tf.nn.elu(out)
                
            elif exp_config.norm_activation is 'relu':
                out = tf.nn.relu(out)
                
            elif exp_config.norm_activation is 'rbf':            
                # ==================
                # fixed scale
                # ==================
                # scale = 0.2
                # ==================
                # learnable scale - one scale per layer
                # ==================
                # scale = tf.Variable(initial_value = 0.2, name = 'scale_'+str(l+1))
                # ==================
                # learnable scale - one scale activation unit
                # ==================
                # init_value = tf.random_normal([1,1,1,n1], mean=0.2, stddev=0.05)
                # scale = tf.Variable(initial_value = init_value, name = 'scale_'+str(l+1))
                
                scale = tf.get_variable(name = 'scale_'+str(l+1), shape = [1,1,1,n1],
                                        initializer = tf.initializers.random_normal())
                out = tf.exp(-(out**2) / (scale**2))
        
        delta = tf.layers.conv2d(inputs=out,
                                  filters=1,
                                  kernel_size=k,
                                  padding='SAME',
                                  name='norm_conv1_'+str(num_layers+1),
                                  use_bias=True,
                                  activation=tf.identity)
        
        # =========================
        # Only model an additive residual effect with the normalizer
        # =========================
        output = images + delta
        
    return output, delta

# ======================================================================
# discriminator for adverserial feature learning
# ======================================================================
def discriminator(features,
                  training_pl,
                  scope_reuse = False):
    
    with tf.variable_scope('discriminator', reuse = scope_reuse):
                
        out = features
        num_layers = 5
        n0 = 16
        
        for l in range(num_layers):
            out = tf.layers.conv2d(inputs=out,
                                   filters=(l+1)*n0,
                                   kernel_size=3,
                                   padding='SAME',
                                   name='D_conv_'+str(l+1) + '_1',
                                   use_bias=True,
                                   activation=None)
            out = tf.layers.batch_normalization(inputs=out, name = 'D_conv_' + str(l+1)  + '_1' + '_bn', training = training_pl)
            out = tf.nn.relu(out)
            
            out = tf.layers.conv2d(inputs=out,
                                   filters=(l+1)*n0,
                                   kernel_size=3,
                                   padding='SAME',
                                   name='D_conv_'+str(l+1) + '_2',
                                   use_bias=True,
                                   activation=None)
            out = tf.layers.batch_normalization(inputs=out, name = 'D_conv_' + str(l+1)  + '_2' + '_bn', training = training_pl)
            out = tf.nn.relu(out)
            
            out = layers.max_pool_layer2d(out)
            
        batch_size = out.get_shape()[0].value
        out = tf.reshape(out, [batch_size, -1])
        logits = tf.layers.dense(out, 1, name = 'D_logits')
        # outputs = tf.nn.sigmoid(logits)

        return logits