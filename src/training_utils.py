import tensorflow as tf
from functools import partial

import tensorflow as tf
from wandb.keras import WandbCallback


def fetch_callbacks(args):
    def scheduler(epoch, lr, transition_epoch):
        if epoch < transition_epoch:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    # using wandb for real-time logging during training
    wandb_callback = WandbCallback(
        monitor="val_loss",
        mode="min",
        save_model=False,
    )

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.save_path,
        save_weights_only=False,
        monitor="val_loss",
        save_freq="epoch",
        verbose=1,
    )

    scheduler = partial(scheduler, transition_epoch=args.transition_epoch)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    callbacks = [wandb_callback, ckpt_callback, lr_callback]
    return callbacks


class LocalTPUClusterResolver(
    tf.distribute.cluster_resolver.TPUClusterResolver):
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
        devices=tf.config.list_logical_devices())

  def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
    return {"TPU": 8}


def is_tpu_available():
    return len(tf.config.list_logical_devices('TPU')) > 0

def is_gpu_available():
    return len(tf.config.list_physical_devices("GPU")) > 0
