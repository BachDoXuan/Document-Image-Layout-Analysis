#import os.path
import os
#import shutil
import tensorflow as tf
#from keras import backend as K
#import helper
import warnings
from distutils.version import LooseVersion
#from seg_mobilenet import SegMobileNet
#import project_tests as tests
from tqdm import tqdm
import numpy as np
#from IPython import embed
from augmentation import rotate_both, flip_both, blur_both, illumination_change_both  # noqa
from dh_segment.network.model import inference_resnet_v1_50
from dh_segment.utils import ModelParams, TrainingParams, class_to_label_image
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
#    classes_file = params['classes_file']  
    
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
    return predict_op, train_op, loss, metrics, global_step, learning_rate, \
            optimizer


# tests.test_optimize(optimize)


def train_and_evaluate(sess, input_images, correct_labels, training,
                       predict_op, train_op, loss, metrics, learning_rate,
                       global_step, train_summary, val_summary,
                       summary_input, params):
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
    n_classes = params["model_params"]["n_classes"]
    training_params = TrainingParams.from_dict(params['training_params'])
    classes_file = params["classes_file"]
#    model_params = ModelParams(**params['model_params'])
    
    # SET UP TRAIN DATA, VAL DATA
    train_batches_fn = input_helper.input_fn(
                            input_data = os.path.join(params["train_data"], "images"),
                            input_label_dir = os.path.join(params["train_data"], "labels"),
                            num_epochs=training_params.evaluate_every_epoch,
                            batch_size=training_params.batch_size,
                            data_augmentation=training_params.data_augmentation,
                            make_patches=training_params.make_patches,
                            image_summaries=True,
                            params=params,
                            num_threads=32)
    val_batches_fn = input_helper.input_fn(
                            input_data = os.path.join(params["eval_data"], "images"),
                            input_label_dir= os.path.join(params["eval_data"], "labels"),
                            batch_size=1,
                            data_augmentation=False,
                            make_patches=False,
                            image_summaries=True,
                            params=params,
                            num_threads=32)
    
    # Create writers for training step and evaluation step
    # Write time stamp into log directory
    now = datetime.now()
    datetime_str = now.strftime("Date_%Y_%m_%d_Time_%H_%M_%S")
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
    train_writer = tf.summary.FileWriter(train_dir, graph=sess.graph)
    val_writer = tf.summary.FileWriter(val_dir)
    
    # Train and evaluate loop
    epoch_pbar = tqdm(range(n_epochs)) # to see the progress
    for epoch in epoch_pbar:
        # 1. TRAIN STEP
        train_loss = 0.0
        num_batches = 0
        global_step_val = 0
        
        # create iterator to get images and labels
        next_images, next_labels = train_batches_fn()
        while True:
            try:
                # to calculate mIoU accuracy for each batch
                sess.run(iou_metric_reset_ops)
                images, labels = sess.run([next_images, next_labels])
                
                # Convert label (shape [batch_size, height, width]) into
                # of shape [batch_size, height, width, 3]
                class_eye = np.eye(n_classes, dtype = np.uint8)
                one_hot_labels = class_eye[labels, :]
                
                start = time.time()
                predict_val, loss_val, global_step_val, learning_rate_val, _, _ = \
                        sess.run([predict_op, loss, global_step, learning_rate, 
                                  train_op, iou_op], 
                                 feed_dict= {input_images: images,
                                             correct_labels: one_hot_labels,
                                             training: True}
                                 )
                end = time.time()
                                    
                speed_val = 1.0 / (end - start)
                train_iou_val = sess.run(iou)
                
                # write summary to disk to display on tensorboard   
                s_image_shape = tf.cast(tf.shape(images)[1:3] / 3, tf.int32)
                           
                s_images = tf.image.resize_images(images, s_image_shape)
                s_correct_labels = \
                    tf.image.resize_images(
                        class_to_label_image(labels, classes_file),
                        s_image_shape
                        )
                s_prediction_labels = \
                    tf.image.resize_images(
                        class_to_label_image(predict_val["labels"],
                                                         classes_file),
                                    s_image_shape
                                    )
                s_images_val, s_correct_labels_val, s_prediction_labels_val = \
                    sess.run([s_images, s_correct_labels, s_prediction_labels])
                    
                train_summary_val = \
                    sess.run(train_summary,
                             feed_dict={
                                summary_input["images"]: s_images_val,
                                summary_input["correct_labels"]: 
                                    s_correct_labels_val,
                                summary_input["prediction_labels"]: 
                                    s_prediction_labels_val,
                                summary_input["learning_rate"]: 
                                    learning_rate_val,
                                summary_input["loss"]: loss_val,
                                summary_input["iou"]: train_iou_val,
                                summary_input["speed"]: speed_val
                                }
                            )
                train_writer.add_summary(train_summary_val, global_step_val)
                train_writer.flush()
                
                # write result to console
                epoch_pbar.write(
                    "Epoch: %03d | global_step: %6d | train_loss: %.4f |" +
                    "train_iou: %.4f"
                    % (epoch, global_step_val, loss_val, train_iou_val)
                    )
                
                train_loss += loss_val
                num_batches += 1
            except tf.errors.OutOfRangeError:
                break

        # calculate train_loss over current epoch
        train_loss /= num_batches


        # 2. EVALUATE STEP
        # to calculate mIoU accuracy for each epoch
        sess.run(iou_metric_reset_ops)  
        val_loss = 0.0
        num_batches = 0
        has_s_image = False # for image summary
        next_images, next_labels = val_batches_fn()
        while True:
            try:
                images, labels = sess.run([next_images, next_labels])
                # Convert label (shape [batch_size, height, width]) into
                # of shape [batch_size, height, width, 3]
                class_eye = np.eye(n_classes, dtype = np.uint8)
                one_hot_labels = class_eye[labels, :]
                
                predict_val, loss_val, _ = \
                        sess.run([predict_op, loss, iou_op], 
                                 feed_dict= {
                                       input_images: images,
                                       correct_labels: one_hot_labels,
                                       training: False
                                       }
                                 )
                        
                if not has_s_image:
                    s_images_input = images
                    s_labels_input = labels
                    s_predictions_input = predict_val["labels"]
                val_loss += loss_val
                num_batches += 1
            except tf.errors.OutOfRangeError:
                break
        val_iou_val = sess.run(iou)
        val_loss /= num_batches
        
        # write summary to disk to display on tensorboard
        s_image_shape = tf.cast(tf.shape(s_images_input)[1:3] / 3, tf.int32)
                   
        s_images = tf.image.resize_images(s_images_input, s_image_shape)
        s_correct_labels = \
            tf.image.resize_images(
                class_to_label_image(s_labels_input, classes_file),
                s_image_shape)
        s_prediction_labels = \
            tf.image.resize_images(
                class_to_label_image(s_predictions_input, classes_file),
                s_image_shape)
        s_images_val, s_correct_labels_val, s_prediction_labels_val = \
            sess.run([s_images, s_correct_labels, s_prediction_labels])
        
        val_summary_val = \
            sess.run(val_summary,
                     feed_dict={
                                summary_input["images"]: s_images_val,
                                summary_input["correct_labels"]: 
                                    s_correct_labels_val,
                                summary_input["prediction_labels"]: 
                                    s_prediction_labels_val,
                                summary_input["loss"]: val_loss,
                                summary_input["iou"]: val_iou_val
                                }
                    )
        val_writer.add_summary(val_summary_val, global_step_val)
        val_writer.flush()
        
        # output to console for current epoch
        epoch_pbar.write(
            "Epoch %03d | train_loss: %.4f |" +
            "val_loss: %.4f | val_mIoU: %.4f"
            % (epoch, train_loss, val_loss, val_iou_val)
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
            "image_shape" : [896, 576],
            "learning_rate_val" : 0.001,
            "debug" : False,
            
            }
    debug = params["debug"]
    ##################################
    ### DEFINE COMPUTATION GRAPH #####
    ##################################
    # 1. Set up params' values (configuration) from dictionary params
#    image_shape = params["image_shape"]
    num_classes = params["model_params"]["n_classes"]
    model_params = ModelParams(**params['model_params'])
#    training_params = TrainingParams.from_dict(params['training_params'])
#    model_params = params['model_params']
#    training_params = params['training_params']
#    classes_file = params["classes_file"]
    
    # 2. Create input tensor for computation graphs
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    correct_labels = tf.placeholder(tf.float32, 
                                    shape=[None, None, None, num_classes],
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

    key_restore_model = 'resnet_v1_50'
    pretrained_restorer = \
                tf.train.Saver(var_list= [v for v in tf.global_variables()
                                            if key_restore_model in v.name])
    if debug:
        with open("resnet_v1_50_vars_before.txt", "w") as var_file:
            for v in tf.global_variables():      
                if key_restore_model in v.name:
                    var_file.write("{}\n".format(v.name))
                
    predict_op, train_op, loss, metrics, global_step, learning_rate, \
        optimizer = build_estimator(logits, correct_labels, params)
    
    # 4. Build summary to display on tensorboard
    #   Tensorboard: works like a web-server, use data from summary to display
    s_images = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                              name = "summary_images")
    s_correct_labels = \
        tf.placeholder(tf.float32, shape=[None, None, None, None],
                       name = "summary_correct_labels")
    s_prediction_labels = \
        tf.placeholder(tf.float32, shape=[None, None, None, None],
                       name = "summary_prediction_labels")
    s_learning_rate = tf.placeholder(tf.float32)
    s_loss = tf.placeholder(tf.float32)
    s_iou = tf.placeholder(tf.float32)
    s_speed = tf.placeholder(tf.float32)
    
    s_image_output = \
        tf.summary.image('input/image', s_images, max_outputs=1)
    s_label_output = \
        tf.summary.image('output/label', s_correct_labels, max_outputs=1)
    s_prediction_output = \
        tf.summary.image('output/prediction', s_prediction_labels, 
                         max_outputs=1)

    s_learning_rate_output = \
        tf.summary.scalar("learning_rate", s_learning_rate)
    s_loss_output = tf.summary.scalar("loss", s_loss)
    s_iou_output = tf.summary.scalar("meanIoU", s_iou)
    s_speed_output = tf.summary.scalar("speed:batch_step_per_sec", s_speed)
    
    train_summary = [s_image_output, s_label_output, s_prediction_output,
                      s_learning_rate_output, s_loss_output, s_iou_output,
                      s_speed_output]
    val_summary = [s_image_output, s_label_output, s_prediction_output,
                   s_loss_output, s_iou_output]
    summary_input = {"images": s_images,
                     "correct_labels": s_correct_labels,
                     "prediction_labels": s_prediction_labels,
                     "learning_rate": s_learning_rate,
                     "loss": s_loss,
                     "iou": s_iou,
                     "speed": s_speed
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
        if debug:      
            with open("global_vars.txt", "w") as var_file:
                for v in tf.global_variables():
                    var_file.write("{}\n".format(v.name))
                    
            with open("resnet_v1_50_vars_after.txt", "w") as var_file:
                for v in tf.global_variables():      
                    if key_restore_model in v.name:
                        var_file.write("{}\n".format(v.name))

                
        pretrained_restorer.restore(sess, model_params.pretrained_model_file)
        
        
        # TRAIN AND EVALUATE LEARNING MODEL
        train_and_evaluate(
                sess, input_images, correct_labels, training,
                predict_op, train_op, loss, metrics, learning_rate,
                global_step, train_summary, val_summary, summary_input, params
                )
        
        # TODO: load back the checkpoint that is good for production and
        # export it as a savedmodel for production -- separate file
        
        # TODO: load back the saved model (computation graph with 
        # weights/variables) and predict on test data or user's data 
        # -- separate file
        


if __name__ == '__main__':
    run()
