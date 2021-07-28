
from typing import List

import tensorflow as tf
import tfx.v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

from ..wav2vec2 import Wav2Vec2ForCTCTrainer, CTCLoss

MODEL_ID = "vasudevgupta/tf-wav2vec2-100h"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
SEQLEN = 246000


def _input_fn(
    files: List[str],
    data_accessor: tfx.components.DataAccessor,
    schema: schema_pb2.Schema,
):
    dataset = data_accessor.tf_default_factory(
        files,
        tfxio.TensorFlowDatasetOptions(batch_size=1),
        schema=schema,
    )
    # TODO: pad & batch
    # dataset = 

    return dataset


def run_fn(fn_args: tfx.components.FnArgs):

    schema: schema_pb2.Schema

    tr_data = _input_fn(fn_args.train_files, fn_args.data_accessor, schema)
    val_data = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema)

    model = Wav2Vec2ForCTCTrainer.from_pretrained(MODEL_ID)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss_fn=CTCLoss(model.config, (BATCH_SIZE, SEQLEN), division_factor=BATCH_SIZE),
    )

    model.fit(
        tr_data,
        validation_data=val_data,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
    )

    tf.saved_model.save(model, fn_args.serving_model_dir)
