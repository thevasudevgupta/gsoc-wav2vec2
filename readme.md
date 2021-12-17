![GSoC](assets/gsoc.png)

This repository presents an implementation of the **Wav2Vec2** model [1] in **TensorFlow 2.0** as a part of [**Google Summer of Code**](https://summerofcode.withgoogle.com/).

For a quick demo, please check out [this](https://hf.co/spaces/vasudevgupta/GOOGLE_SUMMER_OF_CODE). You can find the final report of the project [here](https://vasudevgupta7.github.io/gsoc-wav2vec2/assets/final_report).

## Notebooks

The repository comes with shiny Colab Notebooks. Below you can find a list of them. Spin them up and don't forget to have fun!

| Notebook | Description |
|------------|-------------|
| [**`tensorflow/hub`**](https://www.tensorflow.org/hub/tutorials/wav2vec2_saved_model_finetuning) | This notebook gives you a template to fine-tune a pre-trained Wav2Vec2 SavedModel |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_onnx.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates conversion of TF Wav2Vec2 model to ONNX and compares the latency of ONNX exported model & TF model on CPU |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech_evaluation_WER_3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates Wav2Vec2 evaluation (without any padding) on LibriSpeech data |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech_evaluation_WER_6.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates Wav2Vec2 SavedModel evaluation (with constant padding upto 246000 length) on LibriSpeech data |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2-inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook shows a small demo of how to use Wav2Vec2 for inference for ASR task |

## Checkpoints

Below is a summary of checkpoints obtained during the project:

| 🤗Hub Checkpoint | TFHub `SavedModel` | Description |
|------------------|--------------------|-------------|
| [`gsoc-wav2vec2`](https://hf.co/vasudevgupta/gsoc-wav2vec2) | [`wav2vec2`](https://tfhub.dev/vasudevgupta7/wav2vec2/1) | This checkpoint is TensorFlow's equivalent of [pre-trained Wav2Vec2](https://hf.co/facebook/wav2vec2-base) by Facebook. PyTorch weights are converted into TensorFlow using [`convert_torch_to_tf.py`](src/convert_torch_to_tf.py) |
| [`gsoc-wav2vec2-960h`](https://hf.co/vasudevgupta/gsoc-wav2vec2-960h) | [`wav2vec2-960h`](https://tfhub.dev/vasudevgupta7/wav2vec2-960h/1) | This checkpoint is TensorFlow's equivalent of [fine-tuned Wav2Vec2](https://hf.co/facebook/wav2vec2-base-960h) by Facebook. PyTorch weights are converted into TensorFlow using [`convert_torch_to_tf.py`](src/convert_torch_to_tf.py) |
| [`finetuned-wav2vec2-960h`](https://hf.co/vasudevgupta/finetuned-wav2vec2-960h) | - | This checkpoint is obtained by fine-tuning Wav2Vec2 model on 960h of LibriSpeech dataset during my GSoC tenure. You can reproduce training by running [`main.py`](src/main.py) on TPU v3-8 |
| [`gsoc-wav2vec2-robust`](https://hf.co/vasudevgupta/gsoc-wav2vec2-robust) | [`wav2vec2-robust`](https://tfhub.dev/vasudevgupta7/wav2vec2-robust) | This checkpoint is TensorFlow's equivalent of [pre-trained Wav2Vec2-robust](https://hf.co/facebook/wav2vec2-large-robust) by Facebook. PyTorch weights are converted into TensorFlow using [`convert_torch_to_tf.py`](src/convert_torch_to_tf.py) |
| [`gsoc-wav2vec2-xlsr-53`](https://hf.co/vasudevgupta/gsoc-wav2vec2-xlsr-53) | [`wav2vec2-xlsr-53`](https://tfhub.dev/vasudevgupta7/wav2vec2-xlsr-53) | This checkpoint is TensorFlow's equivalent of [pre-trained Wav2Vec2-xlsr-53](https://hf.co/facebook/wav2vec2-large-xlsr-53) by Facebook. PyTorch weights are converted into TensorFlow using [`convert_torch_to_tf.py`](src/convert_torch_to_tf.py) |

To know more about the process of obtaining the first two checkpoints, please check out [this section](#running-conversion-script) and to know about the process of getting the last checkpoint, please check out [this section](#reproducing-this-project).

## Using this Repository

Install `Wav2Vec2` model from this repository using the `pip` command:

```shell
# this will install the wav2vec2 package
pip3 install git+https://github.com/vasudevgupta7/gsoc-wav2vec2@main
```

You can use the fine-tuned checkpoints (from 🤗 Hub) like this: 

```python
from wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Config

config = Wav2Vec2Config()
model = Wav2Vec2ForCTC(config)
# now use this model like any other TF model

# incase you are interested in already trained model, use `.from_pretrained` method
model_id = "finetuned-wav2vec2-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_id)
```

Additionally, you can use the `SavedModel` from TFHub like this:

```python
import tensorflow_hub as Hub

model_url = "https://tfhub.dev/vasudevgupta7/wav2vec2-960h/1"
model = hub.KerasLayer(model_url)

# use this `model`, just like any other TF SavedModel
```

Please checkout the notebooks referred to in this repository for more information on using the `Wav2Vec2` model.

## Reproducing this project

### Setting Up

```shell
# install & setup TensorFlow first
pip3 install tensorflow==2.5
# Only `TF==2.5` is tested for now!

# install other requirements of this project using the following command:
pip3 install -qr requirements.txt
sudo apt-get install libsndfile1-dev

# switch to code directory for further steps
cd src
```

For using TPUs, it's essential to store model weights and datasets in the GCS bucket so that TPU can access them directly from there. Hence we will create 2 GCS buckets - one for checkpointing and the other for storing LibriSpeech tfrecords.

```shell
# these bucket names will be required to run the training script later
export DATA_BUCKET_NAME="gsoc-librispeech-us"
export CKPT_BUCKET_NAME="gsoc-checkpoints-us"

# create GCS buckets
gsutil mb gs://${DATA_BUCKET_NAME}
gsutil mb gs://${CKPT_BUCKET_NAME}
```

### Preparing dataset

Now we will download the LibriSpeech dataset from the official website & convert them into tfrecords using [`make_tfrecords.py`](src/make_tfrecords.py). Finally, we will export all the tfrecords to the GCS bucket.

```shell
# possible values are `dev-clean`, `train-clean-100`, `train-clean-360`, `train-other-500`, `test-clean`
# you will have to follow same steps for all the configurations (specified above).
export DATA_SPLIT=dev-clean

wget https://www.openslr.org/resources/12/${DATA_SPLIT}.tar.gz
tar -xf ${DATA_SPLIT}.tar.gz

python3 make_tfrecords.py --data_dir LibriSpeech/${DATA_SPLIT} -d ${DATA_SPLIT} -n 50

# transfer tfrecords to GCS bucket
gsutil cp -r ${DATA_SPLIT} gs://<DATA_BUCKET_NAME>/${DATA_SPLIT}
```

Now your GCS bucket (`DATA_BUCKET_NAME`) should look like this:

    .
    |- ${DATA_SPLIT}
        |- ${DATA_SPLIT}-0.tfrecord
        |- ${DATA_SPLIT}-1.tfrecord
        .
        .

Follow the above steps for all other data splits. You need to change the `DATA_SPLIT` environment variable.

### Model training

Since you have installed everything and GCS buckets are configured, we need to run one command to initiate training.

Note: Following commands assume that you already have exported `DATA_BUCKET_NAME` & `CKPT_BUCKET_NAME` environment variables.

The following command will fine-tune the wav2vec2 model on single/multiple GPUs or Colab/Kaggle TPUs:

```shell
python3 main.py
```

For training on Cloud TPUs, run the following command:

```shell
# export `TPU_NAME` environment variable first
# this flag will ensure that your VM connects to the specified TPUs & TPUs become visible to TensorFlow
TPU_NAME=<tpu-name> python3 main.py
```

## Running Conversion script

You can convert original PyTorch checkpoints (from Facebook) using the conversion script available in this repository.

```shell
python3 convert_torch_to_tf.py \
--hf_model_id facebook/wav2vec2-base \ # HuggingFace Hub ID of the model you want to convert
--with_lm_head # Whether to use `Wav2Vec2ForCTC` or `Wav2Vec2Model` from this repository
```

## Running tests

```shell
# first install `torch` & `transformers`
pip3 install torch transformers

# run this from the root of this repository
pytest -sv tests
```

## Acknowledgement

* [Sayak Paul](https://github.com/sayakpaul), [Morgan Roff](https://github.com/MorganR), [Jaeyoun Kim](https://github.com/jaeyounkim) for mentoring me throughout the project.
* [TensorFlow team](https://www.tensorflow.org) & [TRC](https://sites.research.google/trc/) for providing access to TPUs during my GSoC tenure.

## References

[1] Baevski, Alexei, et al. "Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." ArXiv:2006.11477 [Cs, Eess], Oct. 2020. arXiv.org, http://arxiv.org/abs/2006.11477.

## End Notes

Please create an issue if you encounter any problems while using this project. Don't forget to 🌟 this repository if you like this work.
