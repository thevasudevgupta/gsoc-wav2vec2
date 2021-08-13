![GSoC](assets/gsoc.png)

This repository presents an implementation of the **Wav2Vec2** model [1] in **TensorFlow 2.0** as a part of [**Google Summer of Code**](https://summerofcode.withgoogle.com/).

For a quick demo of the project, check out this: https://huggingface.co/spaces/vasudevgupta/GOOGLE_SUMMER_OF_CODE

## Notebooks

The repository comes with shiny Colab Notebooks. Below you can find a list of them. Spin them up and don't forget to have fun!

| Notebook   | Description |
|------------|-------------|
| <a href="https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/wav2vec2_saved_model_finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook gives you a template to fine-tune a pre-trained Wav2Vec2 SavedModel |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_onnx.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates conversion of TF Wav2Vec2 model to ONNX and compares the latency of ONNX exported model & TF model on CPU |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech_evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates Wav2Vec2 evaluation (without any padding) on LibriSpeech data |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech_saved_model_evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates Wav2Vec2 SavedModel evaluation (with constant padding upto 246000 length) on LibriSpeech data |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2-inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook shows a small demo of how to use Wav2Vec2 for inference for ASR task |

## Checkpoints

The original model checkpoints are provided in PyTorch. But you can find the equivalent TensorFlow `SavedModel` on [TensorFlow Hub](https://tfhub.dev/vasudevgupta7/wav2vec2/1). Below is a summary:

| Checkpoint | TF `SavedModel` | Description |
|------------|-------------|-------------|
| [🤗Hub](https://hf.co/vasudevgupta/gsoc-wav2vec2) | [TFHub](https://tfhub.dev/vasudevgupta7/wav2vec2/1) | This checkpoint is TensorFlow's equivalent of [pre-trained Wav2Vec2](facebook/wav2vec2-base) by Facebook. PyTorch weights are converted into TensorFlow using [`convert_torch_to_tf.py`](src/convert_torch_to_tf.py) |
| [🤗Hub](https://hf.co/vasudevgupta/gsoc-wav2vec2-960h) | [TFHub](https://tfhub.dev/vasudevgupta7/wav2vec2-960h/1) | This checkpoint is TensorFlow's equivalent of [fine-tuned Wav2Vec2](facebook/wav2vec2-base-960h) by Facebook. PyTorch weights are converted into TensorFlow using [`convert_torch_to_tf.py`](src/convert_torch_to_tf.py) |
| [🤗Hub](https://hf.co/vasudevgupta/finetuned-wav2vec2-960h) | - | This checkpoint is obtained by fine-tuning Wav2Vec2 model on 960h of LibriSpeech dataset during my GSoC tenure. You can reproduce training by running [`main.py`](src/main.py) on TPU v3-8 |

To know about how we obtained the above checkpoints, please checkout [this section](##-Reproducing-this-project).

## Using this Repository

`Wav2Vec2` model from this repository can be installed using the `pip` command:

```shell
# this will install the wav2vec2 package
pip3 install git+https://github.com/vasudevgupta7/gsoc-wav2vec2@main
```

```python
from wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Config

config = Wav2Vec2Config()
model = Wav2Vec2ForCTC(config)
# now use this model like any other TF model

# incase you are interested in already trained model, use `.from_pretrained` method
model_id = "finetuned-wav2vec2-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_id)
```

Please checkout the notebooks referred to in this repository for more information on how to use the `Wav2Vec2` model.

## Reproducing this project

### Setting Up

```shell
# install & setup TensorFlow first
pip3 install tensorflow

# install other requirements of this project using the following command:
pip3 install -qr requirements.txt
sudo apt-get install libsndfile1-dev

# switch to code directory for further steps
cd src
```

For using TPUs, it's important to store model weights and datasets in the GCS bucket so that TPU can access them directly from there. Hence we will create 2 GCS buckets - one for checkpointing and the other for storing LibriSpeech tfrecords.

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

Follow the above steps for all other data splits. You just need to change the `DATA_SPLIT` environment variable.

### Model training

Now since everything is installed and GCS buckets are configured, we just need to run one command to initiate training.

Note: Following commands assumes that you have exported `DATA_BUCKET_NAME` & `CKPT_BUCKET_NAME` environment variables already.

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

[1] Baevski, Alexei, et al. “Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.” ArXiv:2006.11477 [Cs, Eess], Oct. 2020. arXiv.org, http://arxiv.org/abs/2006.11477.

## End Notes

Please create an issue in case you encountered any issues while using this project. Don't forget to 🌟 this repository if you liked my work.
