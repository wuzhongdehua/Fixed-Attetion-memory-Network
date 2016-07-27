# *MovieQA End2End memory network*

## Dependencies
This code is written in python. To use it you will need:
* Python 2.7
* [Tensorflow 0.8](https://www.tensorflow.org/)
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)
* [Hickle](https://github.com/telegraphic/hickle)
* [Optional] [tqdm](https://pypi.python.org/pypi/tqdm) for progress bar

## Settings
You might need to change some paths in configs.py depending on where you stored the data.
In configs.py, you should change line number 16,17,18 to your own path.
For example,
```bash
flags.DEFINE_string("path_to_data_dump", '/data/seil/w2v_plot_official.hkl', 'your data dump path')
flags.DEFINE_string("path_to_shortcut", './MN_shortcut', 'your data dump path')
flags.DEFINE_string("patt_to_summaries", './summaries', 'your data dump path')
```
also you need to change line number 12 in main.py, your servers own # of GPUs.
```bash
device_count={'GPU':4})) as sess
```


## Getting started
If you are done above settings, we could start training by
```bash
python main.py
```
We wrote best hyperparameter in our training in configs.py,
but you could change these parameters.

## Changes from [End2End Memory Network](http://arxiv.org/abs/1503.08895) and [MovieQA](http://arxiv.org/abs/1512.02902)
We have got best training result at nhops = 3 or 5, also we used *Early stopping*, *batch normalization* and Embedding Demension 512 on our model.
Memory network is very hard to training because of overfitting, So we recommend monitoring training result carefully.
For this, you could need [tensorboard](https://www.tensorflow.org/versions/r0.9/how_tos/summaries_and_tensorboard/index.html) in summary dir by
```bash
tensorboard --logdir='./'
```
Also, where we initialize parameters is critical on our training result.
It is recommended that trying at least 10 random parameters.
