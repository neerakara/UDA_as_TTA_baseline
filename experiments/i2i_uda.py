import model_zoo
import tensorflow as tf

# ======================================================================
# test settings
# ======================================================================
train_dataset = 'HCPT1' # STANFORD / CALTECH / HCPT2 / 'HCPT1'
test_dataset = 'CALTECH' # STANFORD / CALTECH / HCPT2
run_num = 1
uda = True
normalize = False

# ====================================================
# normalizer architecture
# ====================================================
model_handle_normalizer = model_zoo.net2D_i2i
norm_kernel_size = 3
norm_num_hidden_layers = 2
norm_num_filters_per_layer = 16
norm_activation = 'rbf'
norm_batch_norm = False

# ====================================================
# settings of the image to label mapper 
# ====================================================
model_handle_i2l = model_zoo.unet2D_i2l
tr_str = 'tr' + train_dataset
da_ratio = 0.25
run_str = '_r' + str(run_num)
expname_i2l = 'i2l/' + tr_str + run_str

# ====================================================
# uda dir name
# ====================================================
ts_str = 'ts' + test_dataset
lambda_uda = 0.01
lambda_str = '_lambda_uda' + str(lambda_uda)
run_num_uda = 1
run_str_uda = '_r' + str(run_num_uda)

model_handle_discriminator = model_zoo.discriminator
model_handle_generator = model_zoo.generator
lambda_cgan1 = 1.0
lambda_cgan2 = 1.0
lambda_cgan3 = 10.0
lambda_cgan4 = 10.0

expname_uda = expname_i2l + '/uda_cgan/' + ts_str + lambda_str + run_str_uda

# ======================================================================
# data settings
# ======================================================================
data_mode = '2D'
image_size = (256, 256)
image_depth_hcp = 256
image_depth_caltech = 256
image_depth_stanford = 132
target_resolution_brain = (0.7, 0.7)
nlabels = 15
batch_size = 8

# ======================================================================
# training settings
# ======================================================================
optimizer_handle = tf.train.AdamOptimizer
learning_rate = 1e-4
loss_type_i2l = 'dice'

debug = False
max_steps = 30001
summary_writing_frequency = 100
train_eval_frequency = 5000
val_eval_frequency = 250
save_frequency = 1000