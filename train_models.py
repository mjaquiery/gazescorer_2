import os.path
import time

import pandas
import tensorflow as tf
import numpy as np
import random

"""
Construct ML model to categorize gaze direction in Columbia Gaze Data Set.
We run a full set of tests for each strategy, so we train with all but one example and then test on that example,
for each example in the dataset.
"""

for strategy in [r"diff", r"default", r"diff_threshold", r"default_threshold"]:
    start_time = time.time()
    random.seed(20230512)

    input_dir = os.path.join(r"./.output/numpy/", strategy)
    output_dir = os.path.join(r"./.output/models", strategy)

    class_names = ["left", "right"]

    # Load the labels
    labels = np.load(os.path.join(input_dir, "labels.npy"))
    # Convert the labels to integers
    labels = np.array([0 if label == "left" else 1 for label in labels])
    eye_images = np.load(os.path.join(input_dir, "images.npy"))

    results = pandas.DataFrame(columns=["prob_left", "predicted_class", "actual_class"])

    # for i in range(len(eye_images)):
    #     training_images = np.delete(eye_images, i, axis=0)
    #     training_labels = np.delete(labels, i, axis=0)
    # 
    #     # Train the model
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.Flatten(input_shape=eye_images.shape[1:]),
    #         tf.keras.layers.Dense(128, activation='relu'),
    #         tf.keras.layers.Dense(2)  # 2 classes: left and right
    #     ])
    # 
    #     model.compile(
    #         optimizer='adam',
    #         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #         metrics=['accuracy', 'mse']
    #     )
    # 
    #     model.fit(training_images, training_labels, epochs=1, verbose=0)
    # 
    #     # Test model fit on test data
    #     test_images = np.array([eye_images[i]])
    #     test_labels = np.array([labels[i]])
    # 
    #     test_loss, test_acc, test_mse = model.evaluate(test_images,  test_labels, verbose=0)
    # 
    #     print(f'Test accuracy: {test_acc}, MSE: {test_mse}')
    # 
    #     probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    #     predictions = probability_model.predict(test_images)
    # 
    #     # Save the results
    #     results.loc[i] = [predictions[0][0], class_names[np.argmax(predictions[0])], class_names[test_labels[0]]]
    # 
    # results.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    # Save a version of the model with all training data
    # Train the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=eye_images.shape[1:]),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)  # 2 classes: left and right
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', 'mse']
    )

    model.fit(eye_images, labels, epochs=16, verbose=0)

    model.save(os.path.join(output_dir, "model.h5"))

    print(f"Completed for input {input_dir}. Time elapsed: {time.time() - start_time} seconds")
