import argparse
import os

import tensorflow as tf

from data_utils import LABEL_DTYPE, SPEECH_DTYPE, LibriSpeechDataLoader, LibriSpeechDataLoaderArgs
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
    parser.add_argument("--data_dir", default="../data/LibriSpeech/dev-clean", type=str)
    parser.add_argument("-d", "--tfrecord_dir", default="dev-clean", type=str)
    parser.add_argument("-n", "--num_shards", default=1, type=int)

    args = parser.parse_args()
    os.makedirs(args.tfrecord_dir, exist_ok=True)

    data_args = LibriSpeechDataLoaderArgs(data_dir=args.data_dir)
    dataloader = LibriSpeechDataLoader(data_args)
    dataset = dataloader.build_and_fetch_dataset()

    # shards the TFrecords into several files (since overall dataset size is approx 280 GB)
    # this will help TFRecordDataset to read shards in parallel from several files
    # Docs suggest to keep each shard around 100 MB in size, so choose num_shards accordingly
    num_records_per_file = len(dataloader) // args.num_shards
    file_names = [
        os.path.join(args.tfrecord_dir, f"{args.tfrecord_dir}-{i}.tfrecord")
        for i in range(args.num_shards)
    ]
    writers = [tf.io.TFRecordWriter(file_name) for file_name in file_names]

    # following loops runs in O(n) time (assuming n = num_samples & for every tfrecord prepartion_take = O(1))
    i, speech_stats, label_stats = 0, [], []
    pbar = tqdm(dataset, total=len(dataloader), desc=f"Preparing {file_names[i]} ... ")
    for j, inputs in enumerate(pbar):
        speech, label = inputs
        speech, label = tf.squeeze(speech), tf.squeeze(label)
        speech_stats.append(len(speech))
        label_stats.append(len(label))
        tf_record = create_tfrecord(speech, label)

        writers[i].write(tf_record)
        if (j + 1) % num_records_per_file == 0:
            if i == len(file_names) - 1:
                # last file will have extra samples
                continue
            writers[i].close()
            i += 1
            pbar.set_description(f"Preparing {file_names[i]} ... ")
    writers[-1].close()

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
