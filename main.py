#import os.path
#import os
#import shutil
import tensorflow as tf
#from keras import backend as K
import helper
import warnings
from distutils.version import LooseVersion
#from seg_mobilenet import SegMobileNet
#import project_tests as tests
from tqdm import tqdm
import numpy as np
#from IPython import embed
from augmentation import rotate_both, flip_both, blur_both, illumination_change_both  # noqa
from dhSegment.network.model import inference_resnet_v1_50
#from dhSegment.utils import ModelParams, TrainingParams

# https://keras.io/backend/
#KERAS_TRAIN = 1
#KERAS_TEST = 0

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
        tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def build_estimator(logits, correct_labels, params):
    """
    Build the TensorFLow loss and optimizer operations.
    :param logits: TF Tensor of the last layer in the inference_resnet
        shape [batch_size, image_height, image_width, num_classes]
    :param correct_labels: TF Placeholder for the correct label image
        shape [batch_size, image_height, image_width, num_classes]
    :param params: dictionary of parameters for the whole program
    
    :return: Tuple of (logits, train_op, loss)
    """    
    # EXTRACT PARAMS 
    model_params = params['model_params']
    training_params = params['training_params']   
    # BUILD PREDICTION OP 
    #   (FOR PREDICTION PHASE AND INFERENCE PHASE - REAL PRODUCTION)
    prediction_probs = tf.nn.softmax(logits, name='softmax')
    prediction_labels = tf.argmax(logits, axis=-1, name='label_preds')
    predict_op = {'probs': prediction_probs, 'labels': prediction_labels}
    
    # BUILD LOSS OP
    regularized_loss = tf.losses.get_regularization_loss()
    
#    onehot_labels = tf.one_hot(indices=labels, depth=model_params.n_classes)
    with tf.name_scope("loss"):
        per_pixel_loss = tf.nn.softmax_cross_entropy_with_logits(
                                logits=logits,
                                labels=correct_labels, name='per_pixel_loss'
                                )
        if training_params.focal_loss_gamma > 0.0:
            # Probability per pixel of getting the correct label
            probs_correct_label = tf.reduce_max(tf.multiply(prediction_probs, correct_labels))
            modulation = tf.pow((1. - probs_correct_label), training_params.focal_loss_gamma)
            per_pixel_loss = tf.multiply(per_pixel_loss, modulation)

        loss = tf.reduce_mean(per_pixel_loss)
        loss += regularized_loss
    
    # BUILD TRAIN OP
    if training_params.exponential_learning:
#        global_step = tf.train.get_or_create_global_step()
        global_step = tf.Variable(0, trainable=False) # experiment
        learning_rate = tf.train.exponential_decay(
                                        training_params.learning_rate, 
                                        global_step, 
                                        decay_steps=200,
                                        decay_rate=0.95, 
                                        staircase=False
                                        )
    else:
        learning_rate = training_params.learning_rate
        
    tf.summary.scalar('learning_rate', learning_rate)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # Ensures that we execute the update_ops before performing the train_step
        
        # Passing global_step to minimize() will increment it at each step.
        train_op = optimizer.minimize(loss, global_step = global_step)
            
    # BUILD MEAN IOU (METRIC) EVALUATION OP
    squeezed_correct_labels = tf.argmax(correct_labels, axis=-1,
                                        name = "squeezed_correct_labels")
    with tf.variable_scope("iou"): # as scope:
        iou, iou_op = tf.metrics.mean_iou(squeezed_correct_labels, 
                                          prediction_labels, 
                                          model_params.n_classes)
    iou_metric_vars = [v for v in tf.local_variables()
                   if v.name.split('/')[0] == 'iou']
    iou_metric_reset_ops = tf.variables_initializer(iou_metric_vars)   
    metrics = {"iou": iou, 
               "iou_op": iou_op, 
               "iou_metric_reset": iou_metric_reset_ops}
    
    return predict_op, train_op, loss, metrics, global_step


# tests.test_optimize(optimize)


def train_and_evaluate(sess, input_images, correct_labels, training,
                       train_batches_fn, val_batches_fn, train_op, 
                       loss, metrics, params, summary,
                       loss_summary, iou_summary, 
                       writer_train, writer_val, global_step):
    """
    Train and evaluate model
    :param sess: TF Session
    :param input_images: 
    :param correct_labels:
    :param training:
    :param train_batches_fn:
    :param val_batches_fn:
    :param train_op:
    :param loss:
    :param metrics:
    :param params:
    :param summary:
    """
    # Initialization
    iou = metrics["iou"]
    iou_op = metrics["iou_op"]
    iou_metric_reset_ops = metrics["iou_metric_reset"]
    n_epochs = params["training_params"]["n_epochs"]
    batch_size = params["training_params"]["batch_size"]
    
#    writer = tf.summary.FileWriter('./logs', graph=sess.graph)

    epoch_pbar = tqdm(range(n_epochs))
    for epoch in epoch_pbar:
        # training ...
        sess.run(iou_metric_reset_ops)
        train_loss = 0.0
        iteration_counter = 0
        for images, labels in train_batches_fn(batch_size):
            loss_val, global_step_val, _ = \
                    sess.run([loss, global_step, train_op, iou_op], 
                             feed_dict= {input_images: images,
                                         correct_labels: labels,
                                         training: True}
                             )
            train_loss += loss_val
            iteration_counter += 1

        train_iou = sess.run(iou)
        train_loss /= iteration_counter
        
        # computing summary
        summary_val = sess.run(
                        summary, feed_dict={loss_summary: train_loss,
                                            iou_summary: train_iou}
                        )
        writer_train.add_summary(summary_val, global_step_val)
        writer_train.flush()
 
        # evaluating ...
        sess.run(iou_metric_reset_ops)
        val_loss = 0.0
        iteration_counter = 0
        for image, label in val_batches_fn(batch_size):
            loss_val, _ = sess.run([loss, iou_op], 
                                   feed_dict= {
                                           input_images: images,
                                           correct_labels: labels,
                                           training: True
                                           }
                                   )
            val_loss += loss_val
            iteration_counter += 1

        val_iou = sess.run(iou)
        val_loss /= iteration_counter
        
        # computing summary
        summary_val = sess.run(
                        summary, feed_dict={loss_summary: val_loss,
                                            iou_summary: val_iou}
                        )
        writer_val.add_summary(summary_val, global_step_val)
        writer_val.flush()
        
        epoch_pbar.write(
            "Epoch %03d: loss: %.4f mIoU: %.4f val_loss: %.4f val_mIoU: %.4f"
            % (epoch, train_loss, train_iou, val_loss, val_iou)
            )
       
        # saving checkpoints
        if epoch % 2 == 0:
            weight_path = 'checkpoint/ep-%03d-val_loss-%.4f.hdf5' \
                          % (epoch, val_loss)
            model.save_weights(weight_path)


def augmentation_fn(image, label):
    """Wrapper for augmentation methods
    """
    image = np.uint8(image)
    label = np.uint8(label)
    image, label = flip_both(image, label, p=0.5)
    image, label = rotate_both(image, label, p=0.5, ignore_label=1)
    image, label = blur_both(image, label, p=0.5)
    image, label = illumination_change_both(image, label, p=0.5)
    return image, label == 1


# tests.test_train_nn(train_nn)
def run():
    params = {
            "classes_file": "data/pages/classes.txt",
            "eval_data": "data/pages/val",
            "gpu": "0",
            "model_output_dir": "data/page_model",
            "model_params": {
                    "batch_norm": True,
                    "batch_renorm": True,
                    "correct_resnet_version": False,
                    "intermediate_conv": None,
                    "max_depth": 512,
                    "n_classes": 4,
                    "pretrained_model_file": "pretrained_models/resnet_v1_50.ckpt",
                    "pretrained_model_name": "resnet50",
                    "selected_levels_upscaling": [True, True, True, True, True],
                    "upscale_params": [[32, 0], [64, 0], [128, 0], [256, 0], [512, 0]],
                    "weight_decay": 1e-06},
            "prediction_type": "CLASSIFICATION",
            "pretrained_model_name": "resnet50",
            "restore_model": False,
            "seed": 625110693,
            "train_data": "data/pages/train/",
            "training_params": {
                    "batch_size": 1,
                    "data_augmentation": True,
                    "data_augmentation_color": True,
                    "data_augmentation_flip_lr": True,
                    "data_augmentation_flip_ud": True,
                    "data_augmentation_max_rotation": 0.2,
                    "data_augmentation_max_scaling": 0.2,
                    "evaluate_every_epoch": 1,
                    "exponential_learning": True,
                    "focal_loss_gamma": 0.0,
                    "input_resized_size": 720000,
                    "learning_rate": 5e-05,
                    "local_entropy_ratio": 0.0,
                    "local_entropy_sigma": 3,
                    "make_patches": False,
                    "n_epochs": 30,
                    "patch_shape": [300, 300],
                    "training_margin": 0,
                    "weights_labels": None
                    },
            "image_shape" : (896, 576),
            "learning_rate_val" : 0.001,
            
            }
    
    ##################################
    ### DEFINE COMPUTATION GRAPH #####
    ##################################
    # 1. Set up params' values (configuration) from dictionary params
    image_shape = params["image_shape"]
    num_classes = params["model_params"]["n_classes"]
    model_params = params['model_params']
#    training_params = params['training_params']
    
    # 2. Create input tensor for computation graphs
    input_images = tf.placeholder(
        tf.float32,
        shape=[None, image_shape[0], image_shape[1], num_classes], 
        name='input_images')
    correct_labels = tf.placeholder(
        tf.float32,
        shape=[None, image_shape[0], image_shape[1], num_classes],
        name='correct_labels')
    
    training = tf.placeholder(tf.bool(), name='training')    # for batch norm layers 
    
    # 3. Build computation graph
    logits = inference_resnet_v1_50(input_images,
                                            model_params,
                                            model_params.n_classes,
                                            use_batch_norm=model_params.batch_norm,
                                            weight_decay=model_params.weight_decay,
                                            is_training=training
                                            )
    predict_op, train_op, loss, metrics, global_step = \
            build_estimator(logits, correct_labels, params)
    
    # 4. Build summary
    loss_summary = tf.placeholder(tf.float32)
    iou_summary = tf.placeholder(tf.float32)

    tf.summary.scalar("loss", loss_summary)
    tf.summary.scalar("iou", iou_summary)
    summary = tf.summary.merge_all()
    ##################################
    ### CREATE AND RUN SESSION #######
    ##################################

    with tf.Session() as sess:         
        # 1. SET UP TRAIN DATA, VAL DATA, AND TEST DATA
        train_batches_fn = helper.gen_batches_function(
                params["train_data"], image_shape,
                train_augmentation_fn = augmentation_fn)
        val_batches_fn = helper.gen_batches_function(
                params["eval_data"], image_shape)

        # INITIALIZE ALL VARIABLES
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        # LOAD PRETRAINED WEIGHTS INTO GRAPH OPS BELONGING TO RESNET BASE MODEL
        key_restore_model = 'resnet_v1_50'
        pretrained_restorer = \
                tf.train.Saver(var_list= [v for v in tf.global_variables()
                                            if key_restore_model in v.name])
        pretrained_restorer.restore(sess, model_params.pretrained_model_file)
        
        
        # 2. TRAIN AND EVALUATE LEARNING MODEL
        writer_train = tf.summary.FileWriter('./logs/train', graph=sess.graph)
        writer_val = tf.summary.FileWriter('./logs/val')
        train_and_evaluate(sess, input_images, correct_labels, training,
                           train_batches_fn, val_batches_fn,
                           train_op, loss, metrics, params, summary,
                           loss_summary, iou_summary, 
                           writer_train, writer_val, global_step)
        
        # 3. USING THE TRAINED MODEL TO PREDICT TEST SET
        helper.save_inference_samples(
            runs_dir, data_dir, sess, image_shape,
            logits, training, input_images)
        # OPTIONAL: Apply the trained model to a video

        # 4. 

if __name__ == '__main__':
    run()
