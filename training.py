from argparse import Namespace
import json
import os
from typing import Iterable

import tensorflow as tf

from dataloader import ADNIDataLoader
from models.models import MODELS


def build_model(model_name: str,
                optimizer: str | tf.keras.optimizers.Optimizer,
                loss: str | tf.keras.losses.Loss,
                metrics: Iterable[tf.keras.metrics.Metric],
                model_params=None) -> tf.keras.Model:

    if model_params is None:
        model_params = {}

    model_class = MODELS[model_name]

    model = model_class(**model_params)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build()

    print(model.summary())
    return model


def train(namespace: Namespace) -> None:
    results_dir = os.path.join(namespace.results_path, namespace.training_id)
    os.makedirs(results_dir, exist_ok=False)

    early_stopping = tf.keras.callbacks.EarlyStopping(**early_stopping_params) # TODO: change definition to namespace arguments
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(results_dir, "log.csv"))

    callbacks = [early_stopping, csv_logger]

    model = build_model() # TODO: complete definition with namespace arguments
    print(f"Start training of model {namespace.training_id}.")

    loader = ADNIDataLoader()  # TODO: complete definition with namespace arguments
    validation_loader = ADNIDataLoader()  # TODO: complete definition with namespace arguments

    history = model.fit(loader, validation_data=validation_loader, epochs=namespace.epochs, callbacks=callbacks)

    print(f"Training of model {namespace.training_id} finished.")

    model.save(os.path.join(results_dir, "model.tf"), save_format="tf")
    with open(os.path.join(results_dir, "history.json"), "w") as file:
        json.dump(history.history, file, indent=4)
