import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import h5py as h5
import utils
import tf_utils
import parser_ops
import UnrollNet
parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #set it to available GPU


if args.transfer_learning:
    print('Getting weights from trained model:')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    loadChkPoint_tl = tf.train.latest_checkpoint(args.TL_path)
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(args.TL_path + '/modelTst.meta')
        new_saver.restore(sess, loadChkPoint_tl)
        trainable_variable_collections = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        pretrained_model_weights = [sess.run(v) for v in trainable_variable_collections]


save_dir ='saved_models'
directory = os.path.join(save_dir, 'ZS_SSL_' + args.data_opt + '_Rate'+ str(args.acc_rate)+'_'+ str(args.num_reps)+'reps')
if not os.path.exists(directory):
    os.makedirs(directory)

print('create a test model for the testing')
test_graph_generator = tf_utils.test_graph(directory)

#................................................................................
start_time = time.time()
print('.................ZS-SSL Training.....................')
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data..........................................
print('Loading data  for training............... ')
data = sio.loadmat(args.data_dir) 
kspace_train,sens_maps, original_mask= data['kspace'], data['sens_maps'], data['mask']
args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB  = kspace_train.shape

print('Normalize the kspace to 0-1 region')
kspace_train= kspace_train / np.max(np.abs(kspace_train[:]))

#..................Generate validation mask.....................................
cv_trn_mask, cv_val_mask = utils.uniform_selection(kspace_train,original_mask, rho=args.rho_val)
remainder_mask, cv_val_mask=np.copy(cv_trn_mask),np.copy(np.complex64(cv_val_mask))

print('size of kspace: ', kspace_train[np.newaxis,...].shape, ', maps: ', sens_maps.shape, ', mask: ', original_mask.shape)

trn_mask, loss_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                                np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
# train data
nw_input = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
ref_kspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
#...............................................................................
# validation data
ref_kspace_val = np.empty((args.num_reps,args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
nw_input_val = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

print('create training&loss masks and generate network inputs... ')
#train data
for jj in range(args.num_reps):
    trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(kspace_train,remainder_mask, rho=args.rho_train)

    sub_kspace = kspace_train * np.tile(trn_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    ref_kspace[jj, ...] = kspace_train * np.tile(loss_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    nw_input[jj, ...] = utils.sense1(sub_kspace,sens_maps)

#..............................validation data.....................................
nw_input_val = utils.sense1(kspace_train * np.tile(cv_trn_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB)),sens_maps)[np.newaxis]
ref_kspace_val=kspace_train*np.tile(cv_val_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB))[np.newaxis]


# %%  zeropadded outer edges of k-space with no signal- check readme file for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency
if args.data_opt=='Coronal_PD' :
    trn_mask[:, :, 0:17] = np.ones((args.num_reps, args.nrow_GLOB, 17))
    trn_mask[:, :, 352:args.ncol_GLOB] = np.ones((args.num_reps, args.nrow_GLOB, 16))

# %% Prepare the data for the training
sens_maps = np.tile(sens_maps[np.newaxis],(args.num_reps,1,1,1))
sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))
ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
nw_input = utils.complex2real(nw_input)

# %% validation data 
ref_kspace_val = utils.complex2real(np.transpose(ref_kspace_val, (0, 3, 1, 2)))
nw_input_val = utils.complex2real(nw_input_val)

print('size of ref kspace: ', ref_kspace.shape, ', nw_input: ', nw_input.shape, ', maps: ', sens_maps.shape, ', mask: ', trn_mask.shape)

# %% set the batch size
total_batch = int(np.floor(np.float32(nw_input.shape[0]) / (args.batchSize)))
kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='refkspace')
sens_mapsP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='sens_maps')
trn_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='trn_mask')
loss_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='loss_mask')
nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, 2), name='nw_input')

# %% creating the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((kspaceP,nw_inputP,sens_mapsP,trn_maskP,loss_maskP)).shuffle(buffer_size= 10*args.batchSize).batch(args.batchSize)
cv_dataset = tf.data.Dataset.from_tensor_slices((kspaceP,nw_inputP,sens_mapsP,trn_maskP,loss_maskP)).shuffle(buffer_size=10*args.batchSize).batch(args.batchSize)
iterator=tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_iterator=iterator.make_initializer(train_dataset)
cv_iterator = iterator.make_initializer(cv_dataset)

ref_kspace_tensor,nw_input_tensor,sens_maps_tensor,trn_mask_tensor,loss_mask_tensor = iterator.get_next('getNext')
#%% make training model
nw_output_img, nw_output_kspace, *_ = UnrollNet.UnrolledNet(nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor).model

scalar = tf.constant(0.5, dtype=tf.float32)
loss = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + \
       tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1))

all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep=1) #only keep the model corresponding to lowest validation error
sess_trn_filename = os.path.join(directory, 'model')
totalLoss,totalTime=[],[]
total_val_loss = []
avg_cost = 0
print('training......................................................')
lowest_val_loss = np.inf
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print('Number of trainable parameters: ', sess.run(all_trainable_vars))
    feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, sens_mapsP: sens_maps}

    print('Training...')
    # if for args.stop_training consecutive epochs validation loss doesnt go below the lowest val loss,\
    #  stop the training
    if args.transfer_learning:
        print('transferring weights from pretrained model to the new model:')
        trainable_collection_test = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        initialize_model_weights = [v for v in trainable_collection_test]
        for ii in range(len(initialize_model_weights)):
            sess.run(initialize_model_weights[ii].assign(pretrained_model_weights[ii]))
    ep, val_loss_tracker = 0, 0 
    while ep<args.epochs and val_loss_tracker<args.stop_training:
        sess.run(train_iterator, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        try:
            for jj in range(total_batch):
                tmp, _, _ = sess.run([loss, update_ops, optimizer])
                avg_cost += tmp / total_batch    
            toc = time.time() - tic
            totalLoss.append(avg_cost)
        except tf.errors.OutOfRangeError:
            pass
        #%%..................................................................
        # perform validation
        sess.run(cv_iterator, feed_dict={kspaceP: ref_kspace_val, nw_inputP: nw_input_val, trn_maskP: cv_trn_mask[np.newaxis], loss_maskP: cv_val_mask[np.newaxis], sens_mapsP: sens_maps[0][np.newaxis]})
        val_loss = sess.run([loss])[0]
        total_val_loss.append(val_loss)
        # ..........................................................................................................
        print("Epoch:", ep, "elapsed_time =""{:f}".format(toc), "trn loss =", "{:.5f}".format(avg_cost)," val loss =", "{:.5f}".format(val_loss))        
        if val_loss<=lowest_val_loss:
            lowest_val_loss = val_loss    
            saver.save(sess, sess_trn_filename, global_step=ep)
            val_loss_tracker = 0 #reset the val loss tracker each time a new lowest val error is achieved
        else:
            val_loss_tracker += 1
        sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'trn_loss': totalLoss, 'val_loss': total_val_loss})
        ep += 1
    
end_time = time.time()
print('Training completed in  ', str(ep), ' epochs, ',((end_time - start_time) / 60), ' minutes')
