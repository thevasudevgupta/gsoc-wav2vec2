# Wav2Vec2 (GSoC'21)

This repository presents an implementation of the **Wav2Vec2** model [1] in **TensorFlow 2.0** as a part of [**GSoC'21**](https://summerofcode.withgoogle.com/) project.

## Notebooks

The repository comes with shiny Colab Notebooks. Below you can find a list of them. Spin them up and don't forget to have fun!

| Notebook | Description |
|------------|-------------|
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_saved_model_finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook gives you a template to fine-tune a pre-trained Wav2Vec2 SavedModel |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_onnx.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates conversion of Wav2Vec2 model to ONNX model and compares the latency of ONNX exported model & TF model on CPU |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech_evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates Wav2Vec2 evaluation (without any padding) on LibriSpeech data |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech_saved_model_evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates Wav2Vec2 SavedModel evaluation (with constant padding upto 246000 length) on LibriSpeech data |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2-inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook shows a small demo of how to use Wav2Vec2 for inference for ASR task |

## Checkpoints

The original model checkpoints are provided in PyTorch. But you can find the equivalent TensorFlow `SavedModel` on [TensorFlow Hub](https://tfhub.dev/vasudevgupta7/wav2vec2/1). Below is a summary:

| Checkpoint | TF `SavedModel` | Description |
|------------|-------------|-------------|
| [🤗Hub](https://hf.co/vasudevgupta/gsoc-wav2vec2) | [TFHub](https://tfhub.dev/vasudevgupta7/wav2vec2/1) | This checkpoint is TensorFlow's equivalent of [pre-trained Wav2Vec2](facebook/wav2vec2-base) by Facebook. PyTorch weights are converted into TensorFlow using [`convert_torch_to_tf.py`](src/convert_torch_to_tf.py) |
| [🤗Hub](https://hf.co/vasudevgupta/gsoc-wav2vec2-960h) | [TFHub](https://tfhub.dev/vasudevgupta7/wav2vec2-960h/1) | This checkpoint is TensorFlow's equivalent of [fine-tuned Wav2Vec2](facebook/wav2vec2-base-960h) by Facebook. PyTorch weights are converted into TensorFlow using [`convert_torch_to_tf.py`](src/convert_torch_to_tf.py) |

To know about how to run the conversion process for obtaining the TensorFlow `SavedModel` keep reading. 

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
python3 make_tfrecords.py \
--data_dir <source-directory> \
-d <target-directory> \
-n <no-of-tfrecords-shards>
```

```shell
# create a GCS bucket
gsutil mb -l <BUCKET-LOCATION> gs://<GCS-BUCKET-NAME>

# transfer tfrecords to GCS bucket
gsutil cp -r <target-directory> gs://<GCS-BUCKET-NAME>
```

### Model training

The following command will fine-tune wav2vec2 model on single/multiple GPUs or Colab/Kaggle TPUs:

```shell
python3 main.py
```

For training on Cloud TPUs, run the following command:

```shell
# export `TPU_NAME` env variable first
# this flag will ensure that your VM connects to the specified TPUs & TPUs become visible to TensorFlow
TPU_NAME=<tpu-name> python3 main.py
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

## References

[1] Baevski, Alexei, et al. “Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.” ArXiv:2006.11477 [Cs, Eess], Oct. 2020. arXiv.org, http://arxiv.org/abs/2006.11477.
