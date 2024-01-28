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
    def __init__(self, edge_size):
        self.edge_size = edge_size
        self.num_input_neurons = edge_size ** 2

    def create_data_training(self, keep_percentage=100, noise_percentage=10):
        zero_matrix = [[0] * (self.edge_size // 2)] * (self.edge_size // 2)
        for i in range(1, 2 ** self.edge_size):
            elems = to_base2(i, (self.edge_size // 2) ** 2)
            my_matrix = split_into_square_matrix(elems, self.edge_size // 2)
            number_of_ones = sum(elems)
            num_noise_bits = number_of_ones * noise_percentage // 100
            if random.random() * 100 < keep_percentage:
                yield join_matrixes(
                    my_matrix, zero_matrix, zero_matrix, zero_matrix, num_noise_bits
                ).reshape(self.num_input_neurons), 'TL'
            if random.random() * 100 < keep_percentage:
                yield join_matrixes(
                    zero_matrix, my_matrix, zero_matrix, zero_matrix, num_noise_bits
                ).reshape(self.num_input_neurons), 'TR'
            if random.random() * 100 < keep_percentage:
                yield join_matrixes(
                    zero_matrix, zero_matrix, my_matrix, zero_matrix, num_noise_bits
                ).reshape(self.num_input_neurons), 'BL'
            if random.random() * 100 < keep_percentage:
                yield join_matrixes(
                    zero_matrix, zero_matrix, zero_matrix, my_matrix, num_noise_bits
                ).reshape(self.num_input_neurons), 'BR'


class Model:
    """
    16 input neurons whose activations are simply the value in the input data
    0 hidden layers
    1 output layer with 4 neurons
    """

    def __init__(self, edge_size=4, random_init=True, zero_init=False, by_hand=False):
        # n0 = w0_0 * a0 + w0_1 * a1 + ... +w0_15 *a15 + b0
        # n1 = w1_0*a0 + w1_1*a1 + ... w1_15*a15 +b1
        # ...
        # n3 = w3_0*a0 + ... + w3_15*a15 + b3
        # ...so our weights are
        self.edge_size = edge_size

        if random_init:
            self.weights = np.array([[random.random() * 2 - 1 for _ in range(self.edge_size ** 2)] for _ in range(4)])
            self.biases = np.array(
                [random.random() * 2 * 0.5 - 0.5 for _ in range(4)])
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
        dC_over_dw_i_j = np.zeros((4, (self.edge_size ** 2)))

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

    def get_stats_for_data(self, data_and_labels_set):
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


def main(edge_size, desired_precision, noise_percentage):
    t0 = time.time()
    np.set_printoptions(precision=3, suppress=True)


    round = 0
    model = Model(edge_size)
    trainer = Trainer(edge_size)
    training_data_set = list(trainer.create_data_training(keep_percentage=80, noise_percentage=noise_percentage))
    test_data_set = list(trainer.create_data_training(keep_percentage=20, noise_percentage=noise_percentage))
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
            rounds_it_took_to_find_the_solution.append(round)
            model = Model(edge_size)
            round = 0
            previous_precision = -1
            rounds_with_current_precision = -1

        if round > 10_000:

            rounds_it_took_to_find_the_solution.append(1_000_000_000_000)
            model = Model(edge_size)
            round = 0
            previous_precision = -1
            rounds_with_current_precision = -1

        if rounds_with_current_precision > 100:
            rounds_it_took_to_find_the_solution.append(1_000_000_000_000)
            model = Model(edge_size)
            round = 0
            previous_precision = -1
            rounds_with_current_precision = -1

        if len(rounds_it_took_to_find_the_solution) > 2:
            t1 = time.time()
            good_solutions = len([e for e in rounds_it_took_to_find_the_solution if e < 1e12])
            print(
                f'{edge_size}x{edge_size}, '
                f'noise: {noise_percentage}%, '
                f'prec.:{desired_precision}, '
                f'rounds:{global_round}, rps={global_round/(t1-t0):.2f}, '
                f'attempts: {len(rounds_it_took_to_find_the_solution)}, '
                f'good: {100 * good_solutions / len(rounds_it_took_to_find_the_solution):.2f}%, '
                f'percentiles: '
                f'{np.percentile(rounds_it_took_to_find_the_solution, [10, 30, 50, 70, 90, 99, 100])}')


"""
PRECISION = 100%
for 4x4
4x4, after 14416 rounds, 2600 solution attempts, percentiles: [  1.     3.     4.     6.    12.    32.01 103.  ]


for 6x6, the number of rounds to finish training:
for edge size 6, after 14815, the percentiles: [1.00e+00 1.70e+00 3.25e+01 1.00e+12 1.00e+12 1.00e+12 1.00e+12]


for 8x8:
for edge size 8, after 4126 rounds, the percentiles: [1.830e+01 1.549e+02 1.000e+12 1.000e+12 1.000e+12 1.000e+12 1.000e+12]

 
for 10x10
for edge size 10, after 9273 rounds, the percentiles: [1.e+00 5.e+11 1.e+12 1.e+12 1.e+12 1.e+12 1.e+12]
-----------------
PRECISION = 0.95
4x4, prec.:0.95, rounds:14727, attempts: 3129, good: 100.00%, , percentiles: [ 1.    1.    3.    4.   11.   32.72 76.  ]
6x6, prec.:0.95, rounds:8715, attempts: 215, good: 80.93%, , percentiles: [1.e+00 1.e+00 1.e+00 4.e+00 1.e+12 1.e+12 1.e+12]
8x8, prec.:0.95, rounds:2117, solutions: 54, good: 81.48% attempts, percentiles: [1.e+00 1.e+00 1.e+00 1.e+00 1.e+12 1.e+12 1.e+12]
10x10, prec.:0.95, rounds:1158, attempts: 47, good: 78.72%, , percentiles: [1.e+00 1.e+00 1.e+00 1.e+00 1.e+12 1.e+12 1.e+12]
12x12, prec.:0.95, rounds:875, attempts: 18, good: 55.56%, , percentiles: [1.e+00 1.e+00 1.e+00 1.e+12 1.e+12 1.e+12 1.e+12]
------
WITH TEST DATA != TRAINING DATA
4x4, prec.:0.95, rounds:4870, attempts: 1677, good: 100.00%, , percentiles: [ 1.  1.  3.  4.  6. 11. 27.]
6x6, prec.:0.95, rounds:3906, attempts: 137, good: 78.10%, , percentiles: [1.e+00 1.e+00 1.e+00 3.e+00 1.e+12 1.e+12 1.e+12]
8x8, prec.:0.95, rounds:2544, attempts: 63, good: 74.60%, , percentiles: [1.e+00 1.e+00 1.e+00 1.e+01 1.e+12 1.e+12 1.e+12]
10x10, prec.:0.95, rounds:1318, attempts: 78, good: 84.62%, , percentiles: [1.e+00 1.e+00 1.e+00 1.e+00 1.e+12 1.e+12 1.e+12]
12x12, prec.:0.95, rounds:421, attempts: 19, good: 84.21%, percentiles: [1.e+00 1.e+00 1.e+00 1.e+00 1.e+12 1.e+12 1.e+12]
16x16, prec.:0.95, rounds:118, attempts: 3, good: 66.67%, percentiles: [1.0e+00 1.0e+00 1.0e+00 4.0e+11 8.0e+11 9.8e+11 1.0e+12]
-----
WITH  NOISE
4x4, noise: 10%, prec.:0.95, rounds:18352, attempts: 4939, good: 100.00%, percentiles: [ 1.  2.  3.  4.  7. 14. 24.]
6x6, noise: 10%, prec.:0.95, rounds:8131, attempts: 280, good: 79.64%, percentiles: [1.e+00 1.e+00 1.e+00 4.e+00 1.e+12 1.e+12 1.e+12]
8x8, noise: 10%, prec.:0.95, rounds:2465, attempts: 99, good: 81.82%, percentiles: [1.e+00 1.e+00 1.e+00 1.e+00 1.e+12 1.e+12 1.e+12]
10x10, noise: 10%, prec.:0.95, rounds:548, attempts: 40, good: 87.50%, percentiles: [1.e+00 1.e+00 1.e+00 1.e+00 1.e+12 1.e+12 1.e+12]

4x4, noise: 30%, prec.:0.95, rounds:12855, attempts: 4683, good: 100.00%, percentiles: [ 1.  1.  1.  3.  6. 13. 24.]
6x6, noise: 30%, prec.:0.95, rounds:5463, attempts: 71, good: 50.70%, percentiles: [1.0e+00 1.0e+00 9.3e+01 1.0e+12 1.0e+12 1.0e+12 1.0e+12]
10x10, noise: 30%, prec.:0.95, rounds:484, attempts: 41, good: 90.24%, percentiles: [1.e+00 1.e+00 1.e+00 1.e+00 1.e+00 1.e+12 1.e+12]

4x4, noise: 30%, prec.:1.0, rounds:6770, attempts: 1826, good: 100.00%, percentiles: [ 1.    1.    3.    4.    8.   18.75 40.  ]
6x6, noise: 30%, prec.:1.0, rounds:4292, attempts: 63, good: 52.38%, percentiles: [1.0e+00 1.0e+00 3.4e+01 1.0e+12 1.0e+12 1.0e+12 1.0e+12]
8x8, noise: 30%, prec.:1.0, rounds:1266, attempts: 19, good: 47.37%, percentiles: [1.e+00 1.e+00 1.e+12 1.e+12 1.e+12 1.e+12 1.e+12]
10x10, noise: 30%, prec.:1.0, rounds:1362, attempts: 17, good: 23.53%, percentiles: [1.e+00 1.e+12 1.e+12 1.e+12 1.e+12 1.e+12 1.e+12]
12x12, noise: 30%, prec.:1.0, rounds:1363, attempts: 14, good: 7.14%, percentiles: [1.e+12 1.e+12 1.e+12 1.e+12 1.e+12 1.e+12 1.e+12]
"""
if __name__ == '__main__':
    main(edge_size=8, desired_precision=1.0, noise_percentage=30)
