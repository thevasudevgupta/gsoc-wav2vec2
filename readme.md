# Wav2Vec2 (GSoC'21)

In this repositary, I have implemented **Wav2Vec2** model (from paper: [**wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**](https://arxiv.org/abs/2006.11477)) in **TensorFlow 2.0** as a part of [**GSoC**](https://summerofcode.withgoogle.com/) project.

## Notebooks

| Notebook | Description |
|-------------------------------------------|-------------------------------------------|
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech-evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Wav2Vec2 **evaluation on LibriSpeech dataset** using original fine-tuned checkpoint |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2-inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook shows a small demo of how to use Wav2Vec2 for inference on **Automatic Speech Recognition** task. |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_saved_model_finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates the workflow for finetuning `Wav2Vec2 saved-model` (from TensorFlow Hub) for `Speech -> Text` task. Extra head is appended over the top of pre-trained model and whole architecture is wrapped in `tf.keras.Model`. Further, data-processing pipeline is explained in-detail along with few features of `tf.data.Dataset` API. |

## Using this Repository

### Setting Up

```shell
# install & setup TensorFlow first
pip3 install tensorflow

# install other requirements of this project using the following command:
pip3 install -qr requirements.txt

# switch to code directory for further steps
cd src
```

### Preparing dataset

```shell
# It's better to convert speech, text into TFRecords format
# and save it in GCS buckets for online streaming during training
# To make tfrecords out of the dataset downloaded from the official link, run the following:
python3 make_tfrecords.py --data_dir <source-directory> \
-d <target-directory> \
-n <no-of-tfrecords-shards>

# after successful completion of above command, transfer dataset into GCS bucket like this:
gsutil cp -r <target-directory> gs://<GCS-bucket-name>
```

### Model training

```shell
# running following command will initiate training:
# this command will work on multiple cloud-TPUs, multiple GPUs, single GPU
python3 main.py

# for training on Colab Notebooks, you need to export env variable first
# this flag will ensure that colab notebook connects to TPUs & TPUs become visible to TensorFlow
ON_COLAB_TPU=true python3 main.py
```

### Running tests

```shell
# first install `torch` & `transformers`
pip3 install torch transformers

# run this from the root of this repository
pytest -sv tests
```

## Mentors

* [Sayak Paul](https://github.com/sayakpaul)
* [Morgan Roff](https://github.com/MorganR)
* [Jaeyoun Kim](https://github.com/jaeyounkim)
