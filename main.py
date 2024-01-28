"""
the purpose of this is to create a simple neural net, which distinguishes
between very simple patterns, roughly like the following examples
1000
1000
0000
0000   -> should say TL (top left)

0011
0010
0000
0000   -> should say TR (top right)

0000
0000
1100
0000  -> should say BL (bottom left)

0000
0000
0001
0010  -> should say BR (bottom-right)

----
backpropagation
C = cost function = SUM (over all input) (i=output neuron index) OF (a_i - y)^2 = SUM_i(C_i)
so C_0 = (a_0_L - y_0)^2   ...but we drop the _0 index, understanding that we're talking about a specific neuron
C = (a_L - y)^2
    dC/da_L = 2(a_L - y)
a_l = simgoid(z_L)
    da_L/dz_L = sigmoid'(z_L)
z_L = w_L * a_(L-1) + b_L
    dz_L/dw_L = a_(L-1)
    dz_L/db_L = 1
so dC/dW = dC/da * da/dz * dz/dw_L
...so how much does C (meaning C_0) change if we change

...but why do it like this? I mean C_0 = (a_0_L - y0) ^2 => (omitting the _0) C = (a_L -y)^2
    => C = (sigmoid(z_L) - y)^2 => C = (sigmoid(w_L *a_(L-1) + b_L) -y)^2
    => dC/dw_L = ... kind of complicated to do it in one shot indeed
    dC/dw_L = f'f + ff' = 2ff'
    dC/dw_L = 2(sigmoid(w_L * a_(L-1) + b_L) - y) * (sigmoid(w_L * a_(L-1) + b_L)'
    dC/dw_L = 2(sigmoid(w_L * a_(L-1) + b_L) - y) * sigmoid'(w_L * a_(L-1) + b_L) * a_(L-1)

    dC/db_L = dC/da * da_L/dz_L * dz_L/db_l = 2(a_L - y) * sigmoid'(z_L) = 2(sigmoid(z_L)-y) * sigmoid'(w_L * a_(L-1) + b_L)
    dC/db_L = 2(sigmoid(w_L * a_(L-1) + b_L) - y) * sigmoid'(w_L * a_(L-1) + b_L)
        ...dC/dw_L and dC/db_L say how much the cost function varies if w_L and b_L vary
        ...so let's change w_L and b_L with -that amount, and see what happens

---
... ok, so now if we have multiple neurons, in our layers, let's calculate the dC/dw_L and dC/db_L again
----

so what are we gonna do about this?
1. create a formal model
2. create data as an example
3. train the formal model, obtaining the weights and biases of the network => the model
3.1. create a cost function
4. test the model
"""
import itertools
import math
from collections import Counter

import numpy as np
import random


def join_matrixes(tl, tr, bl, br):
    result = []
    for list1, list2 in itertools.chain(zip(tl, tr), zip(bl, br)):
        result.append(list1 + list2)
    return np.array(result)


def to_base2(num):
    result = []
    while num != 0:
        result.append(num % 2)
        num = num // 2
    result.reverse()
    if len(result) < 4:
        result = [0] * (4 - len(result)) + result
    return result


def create_data_training():
    zero_matrix = [[0, 0], [0, 0]]
    for i in range(1, 16):
        a, b, c, d = to_base2(i)
        my_matrix = [[a, b], [c, d]]
        yield join_matrixes(my_matrix, zero_matrix, zero_matrix, zero_matrix).reshape(16), 'TL'
        yield join_matrixes(zero_matrix, my_matrix, zero_matrix, zero_matrix).reshape(16), 'TR'
        yield join_matrixes(zero_matrix, zero_matrix, my_matrix, zero_matrix).reshape(16), 'BL'
        yield join_matrixes(zero_matrix, zero_matrix, zero_matrix, my_matrix).reshape(16), 'BR'


class Model:
    """
    16 input neurons whose activations are simply the value in the input data
    0 hidden layers
    1 output layer with 4 neurons
    """

    def __init__(self, random_init=True, zero_init=False, by_hand=False):
        # n0 = w0_0 * a0 + w0_1 * a1 + ... +w0_15 *a15 + b0
        # n1 = w1_0*a0 + w1_1*a1 + ... w1_15*a15 +b1
        # ...
        # n3 = w3_0*a0 + ... + w3_15*a15 + b3
        # ...so our weights are
        if random_init:
            self.weights = np.array([[random.random() * 2 - 1 for _ in range(16)] for _ in range(4)])
            self.biases = np.array([random.random() * 32 - 16 for _ in range(4)])
            return
        if by_hand:
            self.weights = np.array([
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            ])
            self.biases = np.array([0, 0, 0, 0])

    def train(self, data_and_label_set):
        dC_over_db_i = np.zeros(4)
        dC_over_dw_i_j = np.zeros((4, 16))

        # C = Sum(a_i -yi)^2 over all of the data, so in our case it's C_0 = SUM(a_0 - y_0)^2 over all of the data
        for data_and_label in data_and_label_set:
            data, label = data_and_label

            activations, sigmoid_derived, z_i = self._get_calculation_artefacts(data)
            expected_outcomes = np.array([
                int(label == 'TL'),
                int(label == 'TR'),
                int(label == 'BL'),
                int(label == 'BR'),
            ])

            # Vlad calculated this, and it's the best chance we have of being right!
            # dC/db_i = 2 SUM(over all data) OF {sigmoid( [SUM(over j=0,15) OF w_i_j * a_(L-1)_j] + b_i ) -y_i(d)}
            #               * sigmoid'([SUM(j=0,15) OF w_i_j * a_(L-1)_j] + b_i )
            # SUM(over all data)
            dC_over_db_i_partial = 2 * (activations - expected_outcomes) * sigmoid_derived
            dC_over_db_i += dC_over_db_i_partial

            # Vlad calculated this, and it's the best chance we have of being right!
            # dC/dw_i_j = 2 SUM(over all data) {sigmoid(SUM(over j=0,15) OF [w_i_j * a_(L-1)_j] + b_i) - y_i(d)}
            #               * sigmoid'(SUM(over j=0,15) OF [w_i_j * a_(L-1)_j] + b_i) * a_(L-1)_j
            dC_over_dw_i_j_partial = np.outer(dC_over_db_i_partial, data)
            dC_over_dw_i_j += dC_over_dw_i_j_partial

        # todo - adjust weights and biases
        self.weights -= dC_over_dw_i_j
        self.biases -= dC_over_db_i

    def estimate(self, data):
        resulting_activations, _, __ = self._get_calculation_artefacts(data)
        # how do we map again the output neurons to T/B X L/R ? ...however we want
        max_activation = resulting_activations[0]
        idx_activation = 0
        for idx, activation in enumerate(resulting_activations):
            if activation > max_activation:
                max_activation = activation
                idx_activation = idx
        return {0: 'TL', 1: 'TR', 2: 'BL', 3: 'BR'}.get(idx_activation)

    def _get_calculation_artefacts(self, data):
        # how would we do it "by hand", for it to be properly vectorized? 1/(1+e^(-x))
        z_i = self.weights @ data + self.biases
        sig, sig_derived = sigmoid_and_derivative(z_i)

        # activations, sigmoid, sigmoid'
        return sig, sig_derived, z_i

    def precision_for(self, data_and_labels_set):
        # todo - calc cost and precision
        cost = 0
        precision_hits = 0
        precision_total = 0
        c = Counter()
        estimates = []
        for data, label in data_and_labels_set:
            estimation = self.estimate(data)
            estimates.append(estimation)
            c.update([estimation])
            if estimation == label:
                precision_hits += 1
            precision_total += 1
        return Stats(precision_hits/(precision_total or 1))


def sigmoid_and_derivative(data_array):
    ones = np.ones(len(data_array))
    minus_data = -data_array
    exponential = np.exp(minus_data)
    one_plus_exponential = exponential + ones

    sigmoid = ones / one_plus_exponential
    sigmoid_derived = exponential / np.square(one_plus_exponential)
    return sigmoid, sigmoid_derived


class Stats:
    def __init__(self, precision):
        self.precision = precision

def main():
    artisan_model = Model(by_hand=True, random_init=False)
    model = Model()
    training_data_set = list(create_data_training())

    stats = model.precision_for(training_data_set)
    print("initial precision: ", stats.precision)

    for round in range(100_000):
        model.train(training_data_set)
        stats = model.precision_for(training_data_set)
        print("precision after round {} of training : {}".format(round, stats.precision))


if __name__ == '__main__':
    main()
