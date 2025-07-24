#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of SLURP
# (see https://github.com/CNES/slurp).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Useful functions for Random Forest implementation"""
import time
import logging

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("slurp")

def print_feature_importance(classifier, layers):
    """Compute feature importance."""
    feature_names = ["R", "G", "B", "NIR", "NDVI", "NDWI"] + layers

    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    std = np.std(
        [tree.feature_importances_ for tree in classifier.estimators_], axis=0
    )

    logger.info("Feature ranking:")
    for idx in indices:
        logger.info(
            f" {feature_names[idx]:4s} ({importances[idx]:f}) (std={std[idx]:f})"
        )


def train_classifier(classifier, x_samples, y_samples):
    """Create and train classifier on samples."""
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(
        x_samples, y_samples, test_size=0.2, random_state=42
    )
    classifier.fit(x_train, y_train)
    logger.info("Train time : %s", str(time.time() - start_time))

    # Compute accuracy on train and test sets
    x_train_prediction = classifier.predict(x_train)
    x_test_prediction = classifier.predict(x_test)

    logger.info(
        "Accuracy on train set : %s", str(accuracy_score(y_train, x_train_prediction))
    )
    logger.info("Accuracy on test set : %s", str(accuracy_score(y_test, x_test_prediction)))
