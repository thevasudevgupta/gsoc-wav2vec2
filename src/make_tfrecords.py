import argparse
import os

import tensorflow as tf

from data_utils import (
    LABEL_DTYPE,
    SPEECH_DTYPE,
    LibriSpeechDataLoader,
    LibriSpeechDataLoaderArgs,
)
from tqdm.auto import tqdm


def create_tfrecord(speech_tensor, label_tensor):
    speech_tensor = tf.cast(speech_tensor, SPEECH_DTYPE)
    label_tensor = tf.cast(label_tensor, LABEL_DTYPE)

    speech_bytes = tf.io.serialize_tensor(speech_tensor).numpy()
    label_bytes = tf.io.serialize_tensor(label_tensor).numpy()

    feature = {
        "speech": tf.train.Feature(bytes_list=tf.train.BytesList(value=[speech_bytes])),
        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "CLI to convert .flac dataset into .tfrecords format"
    )
    parser.add_argument(
        "--data_dir", default="../data/librispeech/test-clean", type=str
    )
    parser.add_argument("-d", "--tfrecord_dir", default="test-clean", type=str)
    parser.add_argument("-n", "--num_shards", default=1, type=int)

    args = parser.parse_args()
    os.makedirs(args.tfrecord_dir, exist_ok=True)

    data_args = LibriSpeechDataLoaderArgs(data_dir=args.data_dir)
    dataloader = LibriSpeechDataLoader(data_args)
    dataset = dataloader.build_and_fetch_dataset()

    speech_stats, label_stats = [], []

    # shards the TFrecords into several files (since overall dataset size is approx 280 GB)
    # this will help TFRecordDataset to read shards in parallel from several files
    # Docs suggest to keep each shard around 100 MB in size, so choose num_shards accordingly
    num_records_to_skip = num_records_to_take = len(dataloader) // args.num_shards
    for i in range(args.num_shards):
        file_name = os.path.join(args.tfrecord_dir, f"{args.tfrecord_dir}-{i}.tfrecord")
        # last shard may have extra elements
        if i == args.num_shards - 1:
            num_records_to_take += len(dataloader) % args.num_shards
        with tf.io.TFRecordWriter(file_name) as writer:
            iterable_dataset = dataset.skip(num_records_to_skip * i).take(
                num_records_to_take
            )
            for speech, label in tqdm(
                iterable_dataset,
                total=num_records_to_take,
                desc=f"Preparing {file_name} ... ",
            ):
                speech, label = tf.squeeze(speech), tf.squeeze(label)
                speech_stats.append(len(speech))
                label_stats.append(len(label))
                tf_record = create_tfrecord(speech, label)
                writer.write(tf_record)
    print(f"Total {len(dataloader)} tfrecords are sharded in `{args.tfrecord_dir}`")
    print("############# Data Stats #############")
    print(
        {
            "speech_min": min(speech_stats),
            "speech_mean": sum(speech_stats) / len(speech_stats),
            "speech_max": max(speech_stats),
            "label_min": min(label_stats),
            "label_mean": sum(label_stats) / len(label_stats),
            "label_max": max(label_stats),
        }
    )
    print("######################################")
