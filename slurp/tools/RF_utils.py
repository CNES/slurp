#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from skimage.filters.rank import maximum
from skimage.morphology import square
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from slurp.tools import io_utils


def print_feature_importance(classifier, layers):
    """Compute feature importance."""
    feature_names = ["R", "G", "B", "NIR", "NDVI", "NDWI"] + layers

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    std = np.std(
        [tree.feature_importances_ for tree in classifier.estimators_], axis=0
    )

    print("Feature ranking:")
    for idx in indices:
        print(
            "  %4s (%f) (std=%f)"
            % (feature_names[idx], importances[idx], std[idx])
        )


def train_classifier(classifier, x_samples, y_samples):
    """Create and train classifier on samples."""
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(
        x_samples, y_samples, test_size=0.2, random_state=42
    )
    classifier.fit(x_train, y_train)
    print("Train time :", time.time() - start_time)

    # Compute accuracy on train and test sets
    x_train_prediction = classifier.predict(x_train)
    x_test_prediction = classifier.predict(x_test)

    print("Accuracy on train set :", accuracy_score(y_train, x_train_prediction))
    print("Accuracy on test set :", accuracy_score(y_test, x_test_prediction))


def save_indexes(filename, element_idxs, other_idxs, shape, crs, transform, rpc, colormap):
    """Save points used for learning into a file."""
    img = np.zeros(shape, dtype=np.uint8)

    for row, col in element_idxs:
        img[row, col] = 1

    for row, col in other_idxs:
        img[row, col] = 2

    img_dilat = maximum(img, square(5))
    io_utils.save_image(img_dilat, filename, crs, transform, 0, rpc, colormap)
