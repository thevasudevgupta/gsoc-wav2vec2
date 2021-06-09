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
    max_epochs: int = 1

    logging_steps: int = 10

    # wandb args
    project_name: str = "gsoc-wav2vec2"
    run_name: str = "finetuning"

    def setup(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.logger = wandb.init(
            project_name=self.project_name, run_name=self.run_name, config=asdict(self)
        )

    def train(self, tr_dataset, val_dataset):

        for epoch in range(self.max_epochs):
            pbar = tqdm(enumerate(tr_dataset), desc=f"Running epoch-{epoch}")
            total_loss = tf.constant(0, dtype=tf.float32)
            for i, batch in pbar:
                loss = self.train_on_batch(batch, epoch)
                total_loss += loss

                if (i + 1) % self.logging_steps == 0:
                    self.logger.log(
                        {"step": i + 1, "tr_loss": total_loss.numpy() / (i + 1)},
                        commit=True,
                    )

            eval_loss = self.evaluate(val_dataset)
            self.logger.log(
                {
                    "eval_loss": eval_loss.numpy(),
                    "epoch": epoch,
                },
                commit=False,
            )

    # @tf.function(autograph=True, jit_compile=True)
    def train_on_batch(self, batch):
        # TODO: checkout padding_mask
        with tf.GradientTape as gtape:
            outputs = self.model(batch, training=True)
            loss = outputs["loss"]
        gradients = gtape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(*zip(gradients, self.model.trainable_variables))
        return loss

    # @tf.function(autograph=True, jit_compile=True)
    def evaluate_on_batch(self, batch):
        # TODO: checkout padding_mask
        outputs = self.model(batch, training=False)
        return outputs["loss"]

    def evaluate(self, val_dataset, epoch):
        pbar = tqdm(enumerate(val_dataset), desc=f"Running epoch-{epoch}")
        total_loss = tf.constant(0, dtype=tf.float32)
        for i, batch in pbar:
            loss = self.evaluate_on_batch(batch)
            total_loss += loss
        return total_loss / (i + 1)

    def save_training_state(self, save_dir: str):
        raise NotImplementedError

    def load_training_state(self, load_dir: str):
        raise NotImplementedError
