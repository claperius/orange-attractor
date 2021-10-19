#!/usr/bin/env python3

################################################################################
#
# Mikolaj Sitarz 2021
# Apache License 2.0
#
# Demonstration code for article https://orange-attractor.eu/?p=537
#
################################################################################

import mnist 
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import logging
from numpy.random import default_rng


TSHIRT_INTEGER_VALUE = 0

# initialize pseudorandom number generator
rng = default_rng(seed=31415)

log = logging.getLogger(__name__)


def get_representation(y):
    '''Assign 1 for t-shirt and -1 for any other clothes. The input values are as follows:
    0 	T-shirt/top
    1 	Trouser
    2 	Pullover
    3 	Dress
    4 	Coat
    5 	Sandal
    6 	Shirt
    7 	Sneaker
    8 	Bag
    9 	Ankle boot
    '''
    return (y == TSHIRT_INTEGER_VALUE).astype(np.int32) * 2 - 1


def normalize(x):
    'normalize input vectors'
    nrm = Normalizer()
    nrm.fit(x)
    return nrm.transform(x)


def get_data():
    '''return fashion mnist dataset https://github.com/zalandoresearch/fashion-mnist
    new labels for data are:
    -1  for "non t-shirt"
     1  for "t-shirt"
    '''
    x, y, x_test, y_test = mnist.fashion_mnist('FASHION_MNIST')
    y = get_representation(y)
    y_test = get_representation(y_test)

    # reshape images data into single column vector
    x = x.reshape((len(x), -1))
    x_test = x_test.reshape(len(x_test), -1)

    # normalize data
    x = normalize(x)
    x_test = normalize(x_test)

    return x, y, x_test, y_test


def tshirtize_set(x, y):
    '''Return the set containing 50% tshirts and 50% randomly chosen other clothes.
    The method uses all the tshirts available in the input set'''
    tshirt_idxs = np.where(y == 1)[0]
    non_tshirt_idxs = np.where(y == -1)[0]

    non_thshirt_choice_idxs = rng.choice(non_tshirt_idxs, len(tshirt_idxs))

    x_test_tshirts = x[tshirt_idxs]
    x_test_non_thisrts = x[non_thshirt_choice_idxs]

    y_test_tshirts = y[tshirt_idxs]
    y_test_non_thsirts = y[non_thshirt_choice_idxs]

    x_result = np.concatenate((x_test_tshirts, x_test_non_thisrts))
    y_result = np.concatenate((y_test_tshirts, y_test_non_thsirts))

    return x_result, y_result


def predict_and_analyze(fit, x, y, set_name):

    n = len(y)
    y_predicted = fit.predict(x)

    # convert predictions to {-1, 1}
    y_predicted = (y_predicted > 0).astype(np.int32) * 2 - 1

    diff = y - y_predicted
    overpredicted = (diff < 0).astype(np.int32).sum()  # sum of all overpredicted samples (true_value=-1, predicted_value=1)
    underpredicted = (diff > 0).astype(np.int32).sum()  # sum of all underpredicted samples (true_value=1, predicted_value=-1)

    acc = accuracy_score(y, y_predicted)
    f1 = f1_score(y, y_predicted)

    log.info(f"\n\n********** {set_name} set **********")
    log.info(f"overprediced   = {overpredicted:<2} / {n} ({100*overpredicted/n:>.2f}%)")
    log.info(f"underprediced  = {underpredicted:<2} / {n} ({100*underpredicted/n:>.2f}%)")
    log.info(f"accuracy       = {acc:.4f}")
    log.info(f"f1             = {f1:.4f}")
    log.info("\n")



def main():
    logging.basicConfig(format='%(message)s', level=logging.DEBUG, force=True)

    # fetch the Fashion MNIST dataset
    x, y, x_test, y_test = get_data()

    # construct 50/50 tshirt/non-tshirt sets
    x, y = tshirtize_set(x, y)
    x_test, y_test = tshirtize_set(x_test, y_test)
    log.info(f"training set size: {len(x)}, test set size: {len(x_test)}")

    # fit and analyze result
    fit = linear_model.LinearRegression().fit(x, y)
    predict_and_analyze(fit, x, y, "Train")
    predict_and_analyze(fit, x_test, y_test, "Test")



if __name__ == '__main__':
    main()
