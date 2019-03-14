#!/usr/bin/env python3
# MODIFY CARND TO SEMANTICALLY SEGMENT PRImA DATASET FOR LAYOUT ANALYSIS
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
#import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
	 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
			 tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train \
				  your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
	"""
	Load Pretrained VGG Model into TensorFlow.
	:param sess: TensorFlow Session
	:param vgg_path: Path to vgg folder, containing "variables/" and 
		"saved_model.pb"
	:return: Tuple of Tensors from VGG model 
	    (image_input, keep_prob,layer3_out, layer4_out, layer7_out)
	"""
    # TODO: Implement function
	# Use tf.saved_model.loader.load to load the model and weights
	tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
	
	# Get Tensors to be returned from graph
	graph = tf.get_default_graph()
	image_input = graph.get_tensor_by_name('image_input:0')
	keep_prob = graph.get_tensor_by_name('keep_prob:0')
	layer3 = graph.get_tensor_by_name('layer3_out:0')
	layer4 = graph.get_tensor_by_name('layer4_out:0')
	layer7 = graph.get_tensor_by_name('layer7_out:0')
	
	return image_input, keep_prob, layer3, layer4, layer7

#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  
	Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer ???
	# ??? 1x1 convolution means convoluting through the depth, if image has 3
	# color channels, 1X1 convolution will convolute through color channels
    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, 
							name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that \
	# we can add skip connection with 4th layer
    fcn9 = tf.layers.conv2d_transpose(fcn8, 
									  filters=layer4.get_shape().as_list()[-1],
									  kernel_size=4, strides=(2, 2), 
									  padding='SAME', name="fcn9")

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, 
									   filters = \
									      layer3.get_shape().as_list()[-1], 
									   kernel_size=4, strides=(2, 2), 
									   padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, 
									   filters=num_classes,
									   kernel_size=16, strides=(8, 8), 
									   padding='SAME', name="fcn11")

    return fcn11
    
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a 
	# class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))
    
    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			logits=logits, labels=correct_label_reshaped[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")
    
    # The model implements this operation to find the weights/parameters that 
	# would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
			loss_op, name="fcn_train_op")
    
    return logits, train_op, loss_op
    
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, 
			 cross_entropy_loss, input_image, correct_label, 
			 keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  
		Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    keep_prob_value = 0.5
    learning_rate_value = 0.001
    for epoch in range(epochs):
        # Create function to get batches
        total_loss = 0 # we will eliminate this one
        for X_batch, gt_batch in get_batches_fn(batch_size):

            loss, _ = sess.run([cross_entropy_loss, train_op], 
							   feed_dict = {input_image: X_batch, 
							      correct_label: gt_batch,
								  keep_prob: keep_prob_value, 
								  learning_rate:learning_rate_value})

            total_loss += loss;
            
        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()
        
#tests.test_train_nn(train_nn)


def run():
	# Clear old variables
	tf.reset_default_graph()
	num_classes = 6
	# We resize PRImA dataset into 320x224 images for our model
	image_shape = (320, 224)  
	data_dir = './Data'
	train_dir = './Data/train/'
	train_gt_dir = './Data/train_gt/'
	dev_dir = './Data/dev/'
	runs_dir = './runs'
	#   tests.test_for_kitti_dataset(data_dir)
	EPOCHS = 40
	BATCH_SIZE = 16
#	DROPOUT = 0.75
	correct_label = tf.placeholder(tf.float32, [None, image_shape[0], 
												image_shape[1], num_classes])
	learning_rate = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)
	
	# Download pretrained vgg model
	helper.maybe_download_pretrained_vgg(data_dir)

	# OPTIONAL: Train and Inference on the cityscapes dataset instead of 
	# the Kitti dataset.
	# You'll need a GPU with at least 10 teraFLOPS to train on.
	#  https://www.cityscapes-dataset.com/

	with tf.Session() as session:
		# Path to vgg model
		vgg_path = os.path.join(data_dir, 'vgg')
		
		# Create function to get batches
		get_batches_fn = helper.gen_batch_function(train_dir, train_gt_dir, 
											 image_shape, num_classes)
		
		# OPTIONAL: Augment Images for better results
		#  https://datascience.stackexchange.com/questions/5224/ \
		#  how-to-prepare-augment-images-for-neural-network
		
		# TODO: Build NN using load_vgg, layers, and optimize function
		
		# Returns the three layers, keep probability and input layer from \ 
		# the vgg architecture
		image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, 
															vgg_path)
		
		# The resulting network architecture from adding a decoder on top of \ 
		# the given vgg model
		model_output = layers(layer3, layer4, layer7, num_classes)    
		
		# Build the output logits operation, training operation and 
		# cost operation to be used
		# - logits: each row represents a pixel, each column a class
		# - train_op: function used to get the right parameters to the model 
		#              to correctly label the pixels
        # - cross_entropy_loss: function outputting the cost which we 
		#           are minimizing, lower cost should yield higher accuracy
		logits, train_op, cross_entropy_loss = optimize(model_output, 
												  correct_label, learning_rate, 
												  num_classes)
		
		# Initialize all variables
		session.run(tf.global_variables_initializer())
		session.run(tf.local_variables_initializer())
		
		print("Model build successful, starting training")
		
		# TODO: Train NN using the train_nn function
		# Train the neural network
		train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn, 
                 train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)
		
		# TODO: Save inference data using helper.save_inference_samples
		# Run the model with the dev images and save each painted output 
		# image (roads painted green) - we will modify this for layout analysis
		helper.save_inference_samples(runs_dir, dev_dir, session, image_shape, 
								logits, keep_prob, image_input)
		
		# TensorBoard: save the computation graph to a TensorBoard summary 
		# file as follows:
#		writer = tf.summary.FileWriter('.')
#		writer.add_graph(tf.get_default_graph())
#		writer.flush()
		
		
		print("All done!")
		# OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
