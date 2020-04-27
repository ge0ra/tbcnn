import os
import math
import random
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import net
from deformations import elastically_deform_image_2d
import progress

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Remove tf annoying logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def augment(image, label, size):
    "Augments image and returns the augmented tensor"

    if random.random() > 0.2:
        # Augment with elastic deformations
        image = elastically_deform_image_2d(image[:,:,0], 2, 32)
        # Go back to initial image shape
        image = image.reshape(image.shape + (1,))
    
    # Assume image is square
    max_displacement = image.shape[0] - size
    displacement_x = int(random.random() * max_displacement)
    displacement_y = int(random.random() * max_displacement)

    image = image[
        displacement_y:displacement_y+size,
        displacement_x:displacement_x+size
    ]

    return image, label

def plot_roc(name, labels, predictions, e, **kwargs):
    fp, tp, threshold = roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    #plt.xlim([-0.5,20])
    #plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(os.getcwd() + '/results/graphs/' + str(e) + '.png')

def train_net(training, test, size=512, epochs=400, batch_size=4, logging_interval=5, run_name=None, load_checkpoint=False, save_checkpoint=False):
    """Train network using the given training and test data.
    """

    if run_name is None:
        run_name = datetime.now().strftime(r'%Y-%m-%d_%H:%M')

    training_images, training_labels = training
    test_images, test_labels = test

    # Crop center from test images
    border = (test_images.shape[1] - size) // 2
    test_images = test_images[:,border:border+size, border:border+size]

    print()

    epoch_size = int(math.ceil(training_images.shape[0] / batch_size))

    # Use tensorflow Dataset API to improve the performances of the training set
    # Shuffle, augment and created batches for each epoch
    training_set = (
        tf.data.Dataset.from_tensor_slices((training_images, training_labels))
            # .shuffle(buffer_size=training_images.shape[0])
            .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=training_images.shape[0]))
            .map(lambda im, lab: tf.py_func(augment, [im, lab, size], [im.dtype, lab.dtype]), num_parallel_calls=4)
            .batch(batch_size)
            .prefetch(1)
    )
    
    next_training = training_set.make_one_shot_iterator().get_next()

    # Create network
    inp_var, labels_var, output = net.generate_network(size)
    error_fn, train_fn, metrics = net.generate_functions(inp_var, labels_var, output)

    print('Parameter number: {}'.format( np.sum([np.prod(v.shape) for v in tf.trainable_variables()]) ))

    # Create tensorboard summaries
    metrics_summary = progress.create_metrics_summary(metrics)

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        currentDirectory = os.getcwd()

        if not load_checkpoint:
            # Initialize weights
            sess.run(tf.global_variables_initializer())
        else:
            # Restore variables from disk.
            saver.restore(sess, currentDirectory + "/checkpoints/model.ckpt")
            print("Model restored.")

        # Initialite tensorboard
        progress.init_run(run_name)

        # Training loop
        for e in range(epochs):
            start = time.time()

            # Initialize accuracy calculation
            sess.run(tf.local_variables_initializer())

            # Get needed functions
            accuracy_fn, accuracy_update = metrics['accuracy']
            auc_fn, auc_update = metrics['AUC']

            for b in range(epoch_size):
                batch_imgs, batch_labs = sess.run(next_training)

                # Train
                sess.run([train_fn, accuracy_update, auc_update], {
                    'input:0': batch_imgs,
                    'labels:0': batch_labs,
                })

                # Provide some feedback
                print('Batch {} / {}'.format(b + 1, epoch_size), end='\r')

            # Compute metrics
            accuracy = sess.run(accuracy_fn)
            auc = sess.run(auc_fn)
            computed_classes = []

            if True:
                # Every logging_interval epochs compute and save results on the test set

                # Reset accuracy and auc for the test set
                sess.run(tf.local_variables_initializer())

                # Accuracy on test
                for ti, (img, lab) in enumerate(zip(test_images, test_labels)):
                    sess.run([accuracy_update, auc_update], {
                        'input:0': img.reshape(1, size, size, -1),
                        'labels:0': [lab],
                    })

                    computed_classes.append(sess.run(output['classes'][0], {
                        'input:0': img.reshape(1, size, size, -1),
                        'labels:0': [lab],
                    }))

                    print('Test image {} / {}'.format(ti + 1, len(test_images)), end='\r')

                # Compute test metrics
                test_accuracy = sess.run(accuracy_fn)
                test_auc = sess.run(auc_fn)

                plot_roc("Test Baseline", test_labels, computed_classes, e, color=colors[0], linestyle='--')

                # Collect summaries for tensorboard
                summ_data = sess.run(metrics_summary, {
                    'training_accuracy:0': accuracy,
                    'training_AUC:0': auc,
                    'test_accuracy:0': test_accuracy,
                    'test_AUC:0': test_auc,
                })
                # Write summaries to disk
                progress.add_summary(summ_data, e)

            elapsed = time.time() - start
            # Print progress
            print(
                'Epoch {:>3} | Time: {:>3.0f} s | Acc: {:>5.3f} (Test: {:>5.3f}) | AUC: {:>5.3f} (Test: {:>5.3f})'
                    .format(e, elapsed, accuracy, test_accuracy, auc, test_auc)
            )
        
        if save_checkpoint:
            # Save the variables to disk.
            save_path = saver.save(sess, currentDirectory + "/checkpoints/model.ckpt")