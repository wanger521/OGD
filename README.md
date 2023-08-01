# Softmax regression

## Environment

* python 3.7
* numpy 1.21.4
* matplotlib 2.2.3

## Files

* `MainRunModel.py`: code of main run model for mnist, start from here.

* `RunArtificial.py`: code of main run model for synthetic, start from here.

* `./Models`: code of robust bounded aggregation rules

* `Attacks.py`: Different Byzantine attacks, include gaussian attacks, sign-flipping attacks, sample-duplicating attacks

* `Config.py`: Configurations of these rules under Byzantine attack for mnist experiment. All hyper parameters like learning rate and penalty parameter can be tuned here

* `Config0.py`: Configurations of these rules without attack. All hyper parameters like learning rate and penalty parameter can be tuned here

* `Config_artificial.py`: Configurations of these rules under Byzantine attack for synthetic experiment. All hyper parameters like learning rate and penalty parameter can be tuned here

* `Config0_artificial.py`: Configurations of these rules without attack for synthetic experiment. All hyper parameters like learning rate and penalty parameter can be tuned here

* `draw/draw_all.py`: Plot the curve of mnist experiment results

* `draw/draw_artificial.py`: Plot the curve of synthetic experiment results


* `LoadMnist.py`: Load MNIST dataset

* `FatherModel.py`: Solver of softmax regression, includes the calculation functions of loss, regret, variance and accuracy.

## Results

* The results of experiment are stored in `./results` directory
* The picture of experiment are stored in `./picture` directory
* The meaning of the suffix:
  '' : without Byzantine attacks,  '-gs': gaussian attacks,
  '-sf': sign-flipping attacks,  '-hd': sample-duplicating attacks under non-iid data

## Dataset

* Download [MNIST](http://yann.lecun.com/exdb/mnist/) dataset in `./datasets/MNIST`

