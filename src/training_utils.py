import os
from functools import partial

import tensorflow as tf
import wandb


LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "1"))


class LoggingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs):
        if batch % LOGGING_STEPS == 0:
            wandb.log({**logs, "lr": self.model.optimizer.learning_rate}, commit=True)

    def on_test_end(self, logs):
        wandb.log(logs, commit=False)

    def on_epoch_end(self, epoch, logs):
        wandb.log({**logs, "epoch": epoch}, commit=False)

    def on_train_end(self, logs):
        print("########## TRAINING FINISHED ##########")
        print("Final LOGGING:\n", logs)
        print("#######################################")


def fetch_callbacks(args):
    def scheduler(epoch, lr, transition_epoch):
        multiplier = 1 if epoch < transition_epoch else lr * tf.math.exp(-0.1)
        return lr * multiplier

    scheduler = partial(scheduler, transition_epoch=args.transition_epoch)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    logging_callback = LoggingCallback()
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.ckpt_path,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=False,
        save_freq="epoch",
    )

    return [ckpt_callback, logging_callback, lr_callback]


def is_tpu_available():
    return len(tf.config.list_logical_devices("TPU")) > 0


def is_gpu_available():
    return len(tf.config.list_physical_devices("GPU")) > 0


class LocalTPUClusterResolver(tf.distribute.cluster_resolver.TPUClusterResolver):
    """LocalTPUClusterResolver."""

    def __init__(self):
        self._tpu = ""
        self.task_type = "worker"
        self.task_id = 0

    def master(self, task_type=None, task_id=None, rpc_layer=None):
        return None

    def cluster_spec(self):
        return tf.train.ClusterSpec({})

    def get_tpu_system_metadata(self):
        return tf.tpu.experimental.TPUSystemMetadata(
            num_cores=8,
            num_hosts=1,
            num_of_cores_per_host=8,
            topology=None,
            devices=tf.config.list_logical_devices(),
        )

    def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
        return {"TPU": 8}
