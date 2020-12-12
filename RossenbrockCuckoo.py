import math
import numpy
import random
import timeit


def func(X, n):
    # Objective function
    # for Rosenbrock Function minimum is X = [1, .. 1], F(X) = 0
    F = 0
    for i in range(n-1):
        f = (1 - X[i]) ** 2 + 100 * (X[i+1] - X[i] ** 2) ** 2
        F += f
    return math.fabs(F)


def levyFlight():
    # Levy flight
    return math.pow(random.uniform(0.0001, 0.9999), -1.0 / 3.0)


class Cuckoo:
    def __init__(self, nestCount, pa, stepSize, ranges, convergence, numVar):
        self.nestCount = nestCount  # counts of nest
        self.pa = pa  # Probability of discover
        self.stepSize = stepSize  # step size 'a'
        self.ranges = [ranges*-1, ranges]  # problem ranges
        self.convergence = convergence  # convergence rate [0~1]
        self.numVar = numVar  # number of variables

        self.generation = 0  # current generation
        self.nests = []  # nest set
        self.fitness = []  # fitness set
        self.populations = []  # 2-dimentional population set[nest, fitness]
        self.finder = 0  # first global optimal
        self.finderBools = False

    def search(self):
        # Init nests
        X = numpy.zeros(self.numVar)
        for i in range(self.nestCount):
            for j in range(self.numVar):
                X[j] = round(random.uniform(self.ranges[0], self.ranges[1]), 6)
            self.fitness.append(round(func(X, self.numVar), 4))
            self.nests.append(numpy.array([X]))

        populations = numpy.array([self.nests, self.fitness])
        populations = populations.T
        self.populations = populations[numpy.argsort(populations[:, 1])]
        print('Initialize the minimum that : \n', self.populations[0, 0], "\n", self.populations[0, 1])

        while 1:
            self.generation += 1
            populations = self.populations.copy()

            # Get a cuckoo randomly by levy flight
            cuckooNest = populations[random.randint(0, self.nestCount - 1), 0]
            similarity = levyFlight()
            for j in range(self.numVar):
                X[j] = round(cuckooNest[0][j] + self.stepSize * similarity, 6)
            cuckooNest = numpy.array([X])
            randomNestIndex = random.randint(0, self.nestCount - 1)

            # Evaluate and replace
            F = func(cuckooNest[0], self.numVar)
            if populations[randomNestIndex, 1] > F:
                populations[randomNestIndex, 0] = cuckooNest
                populations[randomNestIndex, 1] = round(F, 4)

            # Pa of worse nests are abandoned and new ones built
            for i in range(self.nestCount - int(self.pa * self.nestCount), self.nestCount):
                for j in range(self.numVar):
                    X[j] = round(random.uniform(self.ranges[0], self.ranges[1]), 6)
                populations[i, 0] = numpy.array([X])
                populations[i, 1] = round(func(X, self.numVar), 4)
            self.populations = populations[numpy.argsort(populations[:, 1])]

            if self.generation % 50000 == 0:
                print(self.generation, 'Generation of optimal solutions, : \n', self.populations[0, 1])
                print(self.populations[0, 0])
                print(self.populations)

            # ICS
            if self.generation % 500000 == 0:
                self.pa = round(self.pa / 1.1, 4)
                self.stepSize = round(self.stepSize / 1.1, 4)

            # satisfied?
            if self.populations[0, 1] == 0 and self.finderBools == False:
                self.finder = self.generation
                self.finderBools = True
            if self.populations[0, 1] == self.populations[int(self.nestCount * self.convergence), 1] or self.generation > 2000000:
                print("Minimum :", self.populations[0, 0], self.populations[0, 1])
                print("First convergence :", self.finder)
                print(self.generation, "Convergence complete")
                break


if __name__ == "__main__":
    print('Print number of variables in your function: ')
    n = int(input())
    # nest_count; probability_of_discover; step_size; ranges; convergence; number_of_variables
    cs = Cuckoo(20, 0.75, 1, 5, 0.2, n)
    start = timeit.default_timer()
    cs.search()
    stop = timeit.default_timer()
    print("Time :", stop - start)
