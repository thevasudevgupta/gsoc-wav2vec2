# GSoC'21 @ TensorFlow

This repositary hosts my work on the project with [TensorFlow](https://github.com/tensorflow/tensorflow) as a part of [GSoC'21](https://summerofcode.withgoogle.com/).

| Mentors | [Jaeyoun Kim](https://github.com/jaeyounkim), [Morgan Roff](https://github.com/MorganR), [Sayak Paul](https://github.com/sayakpaul) |
|---------|---------|

## Notebooks

| Description                               | Link                                      |
|-------------------------------------------|-------------------------------------------|
| Wav2Vec2 **evaluation on LibriSpeech dataset** using original fine-tuned checkpoint | <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech-evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| TensorFlow Wav2Vec2 for **Automatic Speech Recognition** | <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2-inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Using this Repositary

**Setting up**

```shell
# install & setup tensorflow first
pip3 install tensorflow

# note: Cloud TPUs (version: v2-alpha) relies on special version of TensorFlow
# so you should not install TensorFlow yourself there.

# install other requirements of this project using following command:
pip3 install -qr requirements.txt

# switch to code directory for further steps
cd src
```

**Preparing Dataset**

```shell
# It's better to convert speech, text into TFRecords format
# and save it in GCS buckets for online streaming during training
# In order to make tfrecords out of dataset downloaded from official link, run following:
python3 make_tfrecords.py --data_dir=<directory-where-you-downloaded> \
-d=<target-directory-where-tfrecords-should-be-sharded> \
-n=<no-of-tfrecords-shard>

# after successful completion of above command, transfer dataset into GCS bucket like this:
gsutil cp -r <dir-which-you-want-to-copy> gs://<GCS-bucket-name>
```

**Training Model**

```shell
# running following command will initiate training:
# this command will work on multiple cloud-TPUs, multiple GPUs, single GPU
python3 main.py

# for training on Colab Notebooks, you need to export env variable first
# this flag will ensure that colab notebook connects to TPUs & TPUs become visible to TensorFlow
ON_COLAB_TPU=true python3 main.py
```

**Running tests**

Tests ensures that any update to the code will not make the model diverge from the original implementation of `Wav2Vec2`.

```shell
# first install `torch` & `transformers`
pip3 install torch transformers

# run this from root of this repositary
pytest -sv tests
```
