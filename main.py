#import os.path
import os
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
from dh_segment.network.model import inference_resnet_v1_50
from dh_segment.utils import ModelParams, TrainingParams
from datetime import datetime
import time
import input_helper

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
    model_params = ModelParams(**params['model_params'])
    training_params = TrainingParams.from_dict(params['training_params'])
#    model_params = params['model_params']
#    training_params = params['training_params']   
    
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
        
#    lr_summary = tf.summary.scalar('learning_rate', learning_rate)
#    TODO: revise to write learning_rate each global step on 
#   tensorboard, also on console to watch and debug
#    summary = {"lr_summary" : lr_summary} # g 
#    summary = {"learning_rate" : lr_summary}
    
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
    
    return predict_op, train_op, loss, metrics, global_step, learning_rate


# tests.test_optimize(optimize)


def train_and_evaluate(sess, input_images, correct_labels, training,
                       predict_op, train_op, loss, metrics, learning_rate,
                       global_step, summary, params):
    """
    Train, evaluate model, output to console, output to disk for tensorboard
    and save checkpoints for future training and production.
    :param 
    """
    # Set up parameters (configuration)
    iou = metrics["iou"]
    iou_op = metrics["iou_op"]
    iou_metric_reset_ops = metrics["iou_metric_reset"]
    n_epochs = params["training_params"]["n_epochs"]
    batch_size = params["training_params"]["batch_size"]
#    image_shape = params["image_shape"]
#    n_classes = params["model_params"]["n_classes"]
    training_params = TrainingParams.from_dict(params['training_params'])
    
    # SET UP TRAIN DATA, VAL DATA
#    train_batches_fn = helper.gen_batches_function(
#            params["train_data"], image_shape, n_classes, 
#            augmentation_fn = augmentation_fn)
#    val_batches_fn = helper.gen_batches_function(
#            params["eval_data"], image_shape, n_classes)
    train_batches_fn = input_helper.input_fn(
                input_data = os.path.join(params["train_data"], "images"), 
                params = params, 
                input_label_dir = os.path.join(params["train_data"], "labels"), 
                num_epochs=training_params.evaluate_every_epoch, 
                batch_size=training_params.batch_size,
                data_augmentation=training_params.data_augmentation,
                make_patches=training_params.make_patches,
                image_summaries=True,
                num_threads=32
                )
    val_batches_fn = input_helper.input_fn(
                input_data = os.path.join(params["eval_data"], "images"),
                params = params, 
                input_label_dir = os.path.join(params["eval_data"], "labels"),
                batch_size=1,
                data_augmentation=False,
                make_patches=False,
                image_summaries=False,
                num_threads=32)
    
    # Create writers for training step and evaluation step
    # Write time stamp into log directory
    now = datetime.now()
    datetime_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = "./logs/" + datetime_str
    train_dir = output_dir + "/train"
    val_dir = output_dir + "/val"
    if os.path.isdir(output_dir):
        raise (output_dir + " already exists")
    else:
        os.makedirs(output_dir)
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        
    # Create directories (in disk/memory) for each time running code
    writer_train = tf.summary.FileWriter(train_dir, graph=sess.graph)
    writer_val = tf.summary.FileWriter(val_dir)
    
    # Train and evaluate loop
    epoch_pbar = tqdm(range(n_epochs)) # to see the progress
    for epoch in epoch_pbar:
        # 1. TRAIN STEP
        # to calculate mIoU accuracy for each epoch
        sess.run(iou_metric_reset_ops)
        train_loss = 0.0
        num_batches = 0
        global_step_val = 0
        
        # TODO write image, label, prediction summary to watch on tensorboard
        # save 1 image - label -prediction to display on tensorboard for 
        # every 10 global steps
        for images, labels in train_batches_fn(): 
            
            # calculate speed: global step per second
            start = time.time()
            predict_val, loss_val, global_step_val, learning_rate_val, _, _ = \
                    sess.run([predict_op, loss, global_step, learning_rate, 
                              train_op, iou_op], 
                             feed_dict= {input_images: images,
                                         correct_labels: labels,
                                         training: True}
                             )
            end = time.time()
            
            
            if global_step_val % 10 == 0:
                image = np.expand_dims(images[0,:,:,:], axis=0)
                
                origin_label = labels[0,:,:,:]
                origin_label = np.argmax(origin_label, axis=-1)
                label = np.zeros((origin_label.shape[0], origin_label.shape[1],
                                  3), dtype=np.uint8)
                label[origin_label == 1] = [255, 0, 0]
                label[origin_label == 2] = [0, 255, 0]
                label[origin_label == 3] = [0, 0, 255]
                label = np.expand_dims(label, axis = 0)
                
                prediction_label = predict_val["labels"][1,:,:]
                prediction = np.zeros((prediction_label.shape[0], 
                                       prediction_label.shape[1], 3),
                                        dtype=np.uint8)
                prediction[prediction_label == 1] = [255, 0, 0]
                prediction[prediction_label == 2] = [0, 255, 0]
                prediction[prediction_label == 3] = [0, 0, 255]
                prediction = np.expand_dims(prediction, axis = 0)
                if params["debug"]:
                    print("label shape:", label.shape)
                    print("prediction shape:", prediction.shape)
                    
                image_summary, label_summary, prediction_summary = \
                    sess.run([summary["image_summary"], 
                              summary["label_summary"],
                              summary["prediction_summary"]],
                        feed_dict = {summary["image"]: image,                                        
                                     summary["correct_label"]: label,
                                     summary["prediction_label"]: prediction 
                                         })
         
                writer_train.add_summary(image_summary, global_step_val)
                writer_train.add_summary(label_summary, global_step_val)
                writer_train.add_summary(prediction_summary, global_step_val)
                
            step_per_sec_val = 1.0 / (end - start)
            
            # output to console
            epoch_pbar.write(
                "Epoch: %03d, global_step: %6d, train_loss: %.4f"
                % (epoch, global_step_val, loss_val)
                )
            train_loss += loss_val
            num_batches += 1

        train_iou = sess.run(iou)
        train_loss /= num_batches
        
        # write summary to disk to display on tensorboard
        
        loss_summary, iou_summary, lr_summary, speed_summary = \
            sess.run([summary["loss_summary"], summary["iou_summary"],
                      summary["lr_summary"], summary["speed_summary"]],
                     feed_dict={summary["loss"]: train_loss,
                               summary["iou"]: train_iou,
                               summary["learning_rate"]: learning_rate_val,
                               summary["speed"]: step_per_sec_val}
                    )
        writer_train.add_summary(loss_summary, global_step_val)
        writer_train.add_summary(iou_summary, global_step_val)
        writer_train.add_summary(lr_summary, global_step_val)
        writer_train.add_summary(speed_summary, global_step_val)
        writer_train.flush()
 
        # 2. EVALUATE STEP
        # to calculate mIoU accuracy for each epoch
        sess.run(iou_metric_reset_ops)  
        val_loss = 0.0
        num_batches = 0
        for images, labels in val_batches_fn(batch_size):
            loss_val, _ = sess.run([loss, iou_op], 
                                   feed_dict= {
                                           input_images: images,
                                           correct_labels: labels,
                                           training: True
                                           }
                                   )
            val_loss += loss_val
            num_batches += 1

        val_iou = sess.run(iou)
        val_loss /= num_batches
        
        # write summary to disk to display on tensorboard
        loss_summary, iou_summary = \
            sess.run([summary["loss_summary"], summary["iou_summary"]],
                     feed_dict={summary["loss"]: val_loss,
                                summary["iou"]: val_iou}
                    )
        writer_val.add_summary(loss_summary, global_step_val)
        writer_val.add_summary(iou_summary, global_step_val)
        writer_val.flush()
        
        # output to console
        epoch_pbar.write(
            "Epoch %03d: loss: %.4f mIoU: %.4f val_loss: %.4f val_mIoU: %.4f"
            % (epoch, train_loss, train_iou, val_loss, val_iou)
            )
        # TODO SAVE CHECKPOINT: save latest or highest val meanIoU?
        # -- save both



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
    # TODO: monitor params (config) and save them using sacred library
    # TODO: load params from a separate file
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
                    "weight_decay": 1e-06
                    },
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
            "debug" : True,
            
            }
    
    ##################################
    ### DEFINE COMPUTATION GRAPH #####
    ##################################
    # 1. Set up params' values (configuration) from dictionary params
    image_shape = params["image_shape"]
    num_classes = params["model_params"]["n_classes"]
    model_params = ModelParams(**params['model_params'])
#    training_params = TrainingParams.from_dict(params['training_params'])
#    model_params = params['model_params']
#    training_params = params['training_params']
    
    # 2. Create input tensor for computation graphs
    input_images = tf.placeholder(
        tf.float32,
        shape=[None, image_shape[0], image_shape[1], 3])
    correct_labels = tf.placeholder(
        tf.float32,
        shape=[None, image_shape[0], image_shape[1], num_classes],
        name='correct_labels')
    
    training = tf.placeholder(tf.bool, name='training')    # for batch norm layers 
    
    # 3. Build computation graph
    logits = inference_resnet_v1_50(input_images,
                                    model_params,
                                    model_params.n_classes,
                                    use_batch_norm=model_params.batch_norm,
                                    weight_decay=model_params.weight_decay,
                                    is_training=training
                                    )
    predict_op, train_op, loss, metrics, global_step, learning_rate = \
            build_estimator(logits, correct_labels, params)
    
    # 4. Build summary to display on tensorboard
    loss_val_placeholder = tf.placeholder(tf.float32)
    iou_val_placeholder = tf.placeholder(tf.float32)
    lr_val_placeholder = tf.placeholder(tf.float32)
    speed_val_placeholder = tf.placeholder(tf.float32)
    
    image_placeholder = \
        tf.placeholder(tf.float32,
                       shape=[None, image_shape[0], image_shape[1], 3],
                       name='summary/image')
    correct_label_placeholder = \
        tf.placeholder(tf.float32,
                       shape=[None, image_shape[0], image_shape[1], 3],
                       name='summary/label')
    prediction_label_placeholder = \
        tf.placeholder(tf.float32,
                       shape=[None, image_shape[0], image_shape[1], 3],
                       name='summary/prediction')

    summary = {"loss" : loss_val_placeholder,
               "iou" : iou_val_placeholder,
               "learning_rate" : lr_val_placeholder,
               "speed" : speed_val_placeholder,
               "image" : image_placeholder,
               "correct_label" : correct_label_placeholder,
               "prediction_label" : prediction_label_placeholder,
               "loss_summary": tf.summary.scalar("loss", loss_val_placeholder),
               "iou_summary": tf.summary.scalar("iou", iou_val_placeholder),
               "lr_summary": tf.summary.scalar("learning_rate", 
                                               lr_val_placeholder),
               "speed_summary": tf.summary.scalar("global_step/sec", 
                                                    speed_val_placeholder),
               "image_summary": 
                   tf.summary.image("input/image", image_placeholder, 1),
               "label_summary": 
                   tf.summary.image("input/label", correct_label_placeholder, 
                                    1),
               "prediction_summary": 
                   tf.summary.image("output/prediction", 
                                    prediction_label_placeholder, 1)
               }

    
    ##################################
    ### CREATE AND RUN SESSION #######
    ##################################

    with tf.Session() as sess:        
        # TODO: load back checkpoint to continue training and evaluating 
        # INITIALIZE ALL VARIABLES
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        # LOAD PRETRAINED WEIGHTS INTO GRAPH OPS BELONGING TO RESNET BASE MODEL
#        key_restore_model = 'resnet_v1_50'
#        pretrained_restorer = \
#                tf.train.Saver(var_list= [v for v in tf.global_variables()
#                                            if key_restore_model in v.name])
#        pretrained_restorer.restore(sess, model_params.pretrained_model_file)
        
        
        # TRAIN AND EVALUATE LEARNING MODEL
        train_and_evaluate(sess, input_images, correct_labels, training,
                           predict_op, train_op, loss, metrics, learning_rate,
                           global_step, summary, params
                           )
        
        # TODO: load back the checkpoint that is good for production and
        # export it as a savedmodel for production -- separate file
        
        # TODO: load back the saved model (computation graph with 
        # weights/variables) and predict on test data or user's data 
        # -- separate file
        


if __name__ == '__main__':
    run()
