"""
inputs: digits 0 to 9
outputs: binary representations

structure: 10 inputs, 4 outputs


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
--------------
TODOS:
1. [v] make matrix be NxN not just 4x4
2. [v] don't feed all the permutations to the model
3. make the training data more confusing and add more noise

-----
Observations:
for 4x4 matrixes, it trains in 3-4 turns
"""
import itertools
import math
import time
from collections import Counter

import numpy as np
import random


def join_matrixes(tl, tr, bl, br, num_noise_bits=0):
    result = []
    for list1, list2 in itertools.chain(zip(tl, tr), zip(bl, br)):
        result.append(list1 + list2)
    while num_noise_bits > 0:
        candidate_index = math.floor(random.random() * len(result) * len(result))
        candidate_x = math.floor(random.random() * len(result))
        candidate_y = math.floor(random.random() * len(result))
        if result[candidate_x][candidate_y] == 0:
            result[candidate_x][candidate_y] = 1
            num_noise_bits -= 1
    return np.array(result)


def to_base2(num, num_elems):
    result = []
    while num != 0:
        result.append(num % 2)
        num = num // 2
    result.reverse()
    if len(result) < num_elems:
        result = [0] * (num_elems - len(result)) + result
    return result


def split_into_square_matrix(numbers, edge_size):
    if edge_size ** 2 != len(numbers):
        raise ValueError("The length of the list is not a perfect square.")

    square_matrix = []
    for i in range(0, len(numbers), edge_size):
        row = numbers[i:i + edge_size]
        square_matrix.append(row)

    return square_matrix


class Trainer:
    def create_data_training(self):
        for x in range(10):
            result = [0]*10
            result[x] = 1
            yield np.array(result), np.array(self.to_bin(x))

    @staticmethod
    def to_bin(x):
        return list(int(e) for e in bin(x)[2:].zfill(4))


class Model:
    """
    10 inputs
    no hidden layer
    4 outputs
    """

    def __init__(self, random_init=True):
        if random_init:
            self.weights = np.array([[random.random() * 2 - 1 for _ in range(10)] for _ in range(4)])
            self.biases = np.array(
                [random.random() * 2 * 0.5 - 0.5 for _ in range(4)])
            return

    def train(self, data_and_label_set):
        dC_over_db_i = np.zeros(4)
        dC_over_dw_i_j = np.zeros((4, 10))

        # C = Sum(a_i -yi)^2 over all of the data, so in our case it's C_0 = SUM(a_0 - y_0)^2 over all of the data
        for data_and_label in data_and_label_set:
            data, binary_label = data_and_label

            activations, sigmoid_derived, z_i = self._get_calculation_artefacts(data)
            expected_outcomes = binary_label

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

        self.weights -= dC_over_dw_i_j
        self.biases -= dC_over_db_i

    def estimate(self, data):
        resulting_activations, _, __ = self._get_calculation_artefacts(data)
        return np.fromiter((round(e) for e in resulting_activations), int)

    def _get_calculation_artefacts(self, data):
        # how would we do it "by hand", for it to be properly vectorized? 1/(1+e^(-x))
        z_i = self.weights @ data + self.biases
        sig, sig_derived = sigmoid_and_derivative(z_i)

        # activations, sigmoid, sigmoid'
        return sig, sig_derived, z_i

    def get_stats_for_data(self, data_and_labels_set):
        precision_hits = 0
        precision_total = 0
        # c = Counter()
        estimates = []
        for data, label in data_and_labels_set:
            estimation = self.estimate(data)
            estimates.append(estimation)
            # c.update([estimation])
            if np.array_equal(estimation, label):
                precision_hits += 1
            precision_total += 1
        return Stats(precision_hits / (precision_total or 1))


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


def main(desired_precision):
    t0 = time.time()
    np.set_printoptions(precision=3, suppress=True)


    round = 0
    model = Model()
    trainer = Trainer()
    training_data_set = list(trainer.create_data_training())
    test_data_set = list(trainer.create_data_training())
    rounds_it_took_to_find_the_solution = []
    global_round = 0
    previous_precision = -1
    rounds_with_current_precision = -1
    while True:
        global_round += 1

        round += 1
        model.train(training_data_set)
        stats = model.get_stats_for_data(test_data_set)
        if stats.precision != previous_precision:
            previous_precision = stats.precision
            rounds_with_current_precision = 1
        else:
            rounds_with_current_precision += 1

        if stats.precision >= desired_precision:
            for data, label in training_data_set:
                num = int('0b'+''.join(str(e) for e in label), 2)
                print(f"num: {num}, expected: {label}, result: {model.estimate(data)}")
            rounds_it_took_to_find_the_solution.append(round)
            model = Model()
            round = 0
            previous_precision = -1
            rounds_with_current_precision = -1

        if round > 10_000:

            rounds_it_took_to_find_the_solution.append(1_000_000_000_000)
            model = Model()
            round = 0
            previous_precision = -1
            rounds_with_current_precision = -1

        if rounds_with_current_precision > 100:
            rounds_it_took_to_find_the_solution.append(1_000_000_000_000)
            model = Model()
            round = 0
            previous_precision = -1
            rounds_with_current_precision = -1

        if len(rounds_it_took_to_find_the_solution) > 2 and False:
            t1 = time.time()
            good_solutions = len([e for e in rounds_it_took_to_find_the_solution if e < 1e12])
            print(
                f'prec.:{desired_precision}, '
                f'rounds:{global_round}, rps={global_round/(t1-t0):.2f}, '
                f'attempts: {len(rounds_it_took_to_find_the_solution)}, '
                f'good: {100 * good_solutions / len(rounds_it_took_to_find_the_solution):.2f}%, '
                f'percentiles: '
                f'{np.percentile(rounds_it_took_to_find_the_solution, [10, 30, 50, 70, 90, 99, 100])}')


if __name__ == '__main__':
    main(desired_precision=1.0)
