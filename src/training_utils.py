from functools import partial

import tensorflow as tf
import wandb


class TrainingCallback(tf.keras.callbacks.Callback):

    def __init__(self, logging_steps, trainable_transition_epoch):
        super().__init__()
        self.logging_steps = logging_steps
        self.trainable_transition_epoch = trainable_transition_epoch

    def on_train_batch_end(self, batch, logs):
        if batch % self.logging_steps == 0:
            wandb.log({**logs, "lr": self.model.optimizer.learning_rate}, commit=True)

        print("parameters", len(self.model.trainable_variables))

    def on_test_end(self, logs):
        wandb.log(logs, commit=False)

    def on_epoch_end(self, epoch, logs):
        wandb.log({**logs, "epoch": epoch}, commit=False)
        if epoch == self.trainable_transition_epoch:
            print("#######################################")
            print("Freezing feature extractor layer & training rest of model")
            self.model.trainable = True
            self.model.freeze_feature_extractor()
            self.model.summary()
            print("#######################################")

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

    training_callback = TrainingCallback(args.logging_steps, args.trainable_transition_epoch)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.ckpt_path,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=False,
        save_freq="epoch",
    )

    return [ckpt_callback, training_callback, lr_callback]


def is_tpu_available():
    return len(tf.config.list_logical_devices("TPU")) > 0


def is_gpu_available():
    return len(tf.config.list_physical_devices("GPU")) > 0
