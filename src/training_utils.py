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

    def on_test_end(self, logs):
        wandb.log(logs, commit=False)

    def on_epoch_end(self, epoch, logs):
        wandb.log({**logs, "epoch": epoch}, commit=False)
        if epoch == self.trainable_transition_epoch:
            print("######## FREEZING ########")
            self.model.trainable = True
            for i in range(len(self.model.layers[0].layers) - 2):
                self.model.layers[0].layers[i].trainable = False
                print(self.model.layers[0].layers[i])
            self.model.summary()
            print("#######################################")


def fetch_callbacks(args):
    def scheduler(epoch, lr, lr1, lr2, lr3, transition_epoch1, transition_epoch2):
        if epoch <= transition_epoch1:
            lr = lr1
        elif epoch <= transition_epoch2:
            lr = lr2
        else:
            lr = lr3
        return lr

    scheduler = partial(scheduler, lr1=args.lr1, lr2=args.lr2, lr3=args.lr3, transition_epoch1=args.transition_epoch1, transition_epoch2=args.transition_epoch2)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    training_callback = TrainingCallback(args.logging_steps, args.trainable_transition_epoch)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.ckpt_path,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_freq="epoch",
    )

    return [ckpt_callback, training_callback, lr_callback]


def is_tpu_available():
    return len(tf.config.list_logical_devices("TPU")) > 0


def is_gpu_available():
    return len(tf.config.list_physical_devices("GPU")) > 0
