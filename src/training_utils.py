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
