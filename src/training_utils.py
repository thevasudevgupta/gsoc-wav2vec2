from functools import partial

import tensorflow as tf
import wandb


class LoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logging_steps):
        super().__init__()
        self.logging_steps = logging_steps

    def on_train_batch_end(self, batch, logs):
        if batch % self.logging_steps == 0:
            wandb.log({**logs, "lr": self.model.optimizer.learning_rate}, commit=True)

    def on_test_end(self, logs):
        wandb.log(logs, commit=False)

    def on_epoch_end(self, epoch, logs):
        wandb.log({**logs, "epoch": epoch}, commit=False)


def fetch_callbacks(args, is_stage2=False):
    def scheduler(epoch, lr, lr1, lr2, transition_epochs):
        return lr1 if epoch <= transition_epochs else lr2

    callbacks = []

    if is_stage2:
        scheduler = partial(scheduler, lr1=args.stage2_lr1, lr2=args.stage2_lr2, transition_epochs=args.stage2_transition_epochs)
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
        ckpt_path = args.ckpt_path + "_stage2/tf_model"
    else:
        ckpt_path = args.ckpt_path + "_stage1/tf_model"

    logging_callback = LoggingCallback(args.logging_steps)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=False,
        save_freq="epoch",
    )

    callbacks.extend([ckpt_callback, logging_callback])
    return callbacks


def is_tpu_available():
    return len(tf.config.list_logical_devices("TPU")) > 0


def is_gpu_available():
    return len(tf.config.list_physical_devices("GPU")) > 0
