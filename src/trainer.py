# __author__ = "Vasudev Gupta"
# __author_email__ = "7vasudevgupta@gmail.com"

"""Simple Trainer for handling TensorFlow Model Training"""

from dataclasses import asdict, dataclass

import tensorflow as tf
import wandb
from tqdm.auto import tqdm

@dataclass
class Trainer:
    lr: float
    max_epochs: int = 2

    logging_steps: int = 10
    eval_steps: int = 10

    use_tpu: bool = False

    save_steps: int = 10
    ckpt_dir: str = "checkpoints" # None

    # wandb args
    project_name: str = "gsoc-wav2vec2"
    run_name: str = "finetuning"

    def setup(self, model, optimizer, restore_ckpt=False):

        if self.use_tpu:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            self.strategy = tf.distribute.TPUStrategy(resolver)
            with self.strategy.scope():
                self.model = model
                self.optimizer = optimizer
        else:
            self.strategy = None
            self.model = model
            self.optimizer = optimizer

        self.logger = wandb.init(
            project_name=self.project_name, run_name=self.run_name, config=asdict(self)
        )

        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(checkpoint, self.ckpt_dir, max_to_keep=5)

        if restore_ckpt:
            self.restore_checkpoint(checkpoint)

    def train(self, tr_dataset_fn, val_dataset_fn):

        if self.strategy is not None:
            tr_dataset = self.strategy.distribute_datasets_from_function(tr_dataset_fn)
            val_dataset = self.strategy.distribute_datasets_from_function(val_dataset_fn)
        else:
            tr_dataset = tr_dataset_fn()
            val_dataset = val_dataset_fn()

        for epoch in range(self.max_epochs):
            pbar = tqdm(enumerate(tr_dataset), desc=f"Running epoch-{epoch}")
            total_loss = tf.constant(0, dtype=tf.float32)
            for i, batch in pbar:
                speech, labels = batch
                loss = self.training_step(speech, epoch)
                total_loss += loss

                val_loss = None
                if (i + 1) % self.eval_steps == 0:
                    val_loss = self.evaluate(val_dataset)

                if (i + 1) % self.logging_steps == 0:
                    self.logger.log(
                        {"step": i + 1, "tr_loss": total_loss.numpy() / (i + 1), "val_loss": val_loss},
                        commit=True,
                    )

                if (i + 1) % self.save_steps == 0:
                    self.save_checkpoint()

            self.logger.log({"epoch": epoch}, commit=False)

    def training_step(self, batch):
        if self.strategy is not None:
            return self.strategy.run(self._training_step, args=(batch, ))
        return self._training_step(batch)

    # @tf.function(autograph=True, jit_compile=True)
    def _training_step(self, batch):
        with tf.GradientTape as gtape:
            outputs = self.model(batch, training=True)
            loss = outputs["loss"]
        gradients = gtape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(*zip(gradients, self.model.trainable_variables))
        return loss

    def validation_step(self, batch):
        if self.strategy is not None:
            return self.strategy.run(self._validation_step, args=(batch, ))
        return self._validation_step(batch)

    # @tf.function(autograph=True, jit_compile=True)
    def _validation_step(self, batch):
        outputs = self.model(batch, training=False)
        return outputs["loss"]

    def evaluate(self, val_dataset, epoch):
        pbar = tqdm(enumerate(val_dataset), desc=f"Running epoch-{epoch}")
        total_loss = tf.constant(0, dtype=tf.float32)
        for i, batch in pbar:
            speech, labels = batch
            loss = self.validation_step(speech)
            total_loss += loss
        return total_loss / (i + 1)

    def save_checkpoint(self):
        self.manager.save()
        print("CHECKPOINTED")

    def restore_checkpoint(self, checkpoint):
        status = checkpoint.restore(self.manager.latest_checkpoint)
        print("CHECKPOINT RESTORE STATUS -", status)
