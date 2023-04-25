from argparse import Namespace
import json
import os
from typing import Iterable, Union

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from dataloader import MRIDataLoader
from models.models import MODELS


def build_model(model_name: str,
                optimizer: Union[str, tf.keras.optimizers.Optimizer],
                loss: Union[str, tf.keras.losses.Loss],
                metrics: Iterable[tf.keras.metrics.Metric],
                model_params=None) -> tf.keras.Model:

    if model_params is None:
        model_params = {}

    model_class = MODELS[model_name]

    model = model_class(**model_params)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build(input_shape=(None, 128, 128, 3))

    print(model.summary())
    return model


def train(namespace: Namespace) -> None:
    results_dir = os.path.join(namespace.results_path, namespace.training_id)
    os.makedirs(results_dir, exist_ok=False)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy")
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(results_dir, "log.csv"))

    callbacks = [early_stopping, csv_logger]

    model = build_model("ConvNet",
                        optimizer="adam",
                        loss="categorical_crossentropy",
                        metrics=[tf.keras.metrics.AUC(), "accuracy"])
    
    print(f"Start training of model {namespace.training_id}.")
    
    metadata = pd.read_csv(os.path.join(namespace.data_path, "metadata.csv"))
    X_train, X_val, _, _ = train_test_split(metadata["name"], metadata["label"], 
                                            stratify=metadata["label"], random_state=namespace.random_seed)

    loader = MRIDataLoader(data_path=os.path.join(namespace.data_path, "preprocessed"), metadata_path=os.path.join(namespace.data_path, "metadata.csv"),
                           patients=X_train, batch_size=256, verbose=False)
    validation_loader = MRIDataLoader(data_path=os.path.join(namespace.data_path, "preprocessed"), metadata_path=os.path.join(namespace.data_path, "metadata.csv"),
                                      patients=X_val, shuffle_all=False, shuffle_batch=False, batch_size=256, verbose=False)

    history = model.fit(loader, validation_data=validation_loader, epochs=namespace.epochs, callbacks=callbacks)

    print(f"Training of model {namespace.training_id} finished.")

    model.save(os.path.join(results_dir, "model.tf"), save_format="tf")
    with open(os.path.join(results_dir, "history.json"), "w") as file:
        json.dump(history.history, file, indent=4)
