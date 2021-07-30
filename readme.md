# Wav2Vec2 (GSoC'21)

This repository presents an implementation of the **Wav2Vec2** model [1] in **TensorFlow 2.0** as a part of [**GSoC'21**](https://summerofcode.withgoogle.com/) project.

## Notebooks

The repository comes with shiny Colab Notebooks. Below you can find a list of them. Spin them up and don't forget to have fun!

| Notebook | Description |
|------------|-------------|
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_saved_model_finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook gives you a template to fine-tune a pre-trained Wav2Vec2 SavedModel |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech-evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates Wav2Vec2 evaluation on LibriSpeech data |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2-inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook shows a small demo of how to use Wav2Vec2 for inference for ASR task |

## Checkpoints

The original model checkpoints are provided in PyTorch. But you can find the equivalent TensorFlow `SavedModel` on [TensorFlow Hub](https://tfhub.dev/vasudevgupta7/wav2vec2/1). Below is a summary.

| Checkpoint | TF `SavedModel` | Description |
|------------|-------------|-------------|
| [HuggingFace-Hub](https://hf.co/vasudevgupta/gsoc-wav2vec2) | [TFHub](https://tfhub.dev/vasudevgupta7/wav2vec2/1) | This checkpoint is TensorFlow's equivalent of pre-trained Wav2Vec2 by Facebook. PyTorch weights are converted into TensorFlow using [`convert_torch_to_tf.py`](src/convert_torch_to_tf.py) |

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

# after successful completion of above command, transfer dataset into GCS bucket like this:
gsutil cp -r <target-directory> gs://<GCS-bucket-name>
```

### Model training

The following command will fine-tune wav2vec2 model on single/multiple GPUs:

```shell
python3 main.py
```

For training on Cloud/Colab TPUs, run the following command:

```shell
# export `ON_TPU` env variable first
# this flag will ensure that your VM connects to TPUs & TPUs become visible to TensorFlow
ON_TPU=true python3 main.py
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
