import numpy as numpy


def main():
    inf = open('C:/Users/Хэшэгто/Desktop/input.txt', 'r')
    coeffs = int(inf.readline().strip())
    inequalities = int(inf.readline().strip())
    print('Количество переменных: ', coeffs)
    print('Количество ограничений: ', inequalities)

    method = SymplexMethod()
    method.generateMatrix(coeffs, inequalities)

    line = inf.readline().strip()
    method.add_ObjectiveFunction(line, coeffs)

    for i in range(inequalities):
        line = inf.readline().strip()
        method.add_IneqFunctions(line, coeffs, i)
    inf.close()

    print(method.table)
    while method.iter == 0:
        method.check_Optimality(coeffs)
    for i in range(inequalities + 1):
        print(method.table[i][-1])


class SymplexMethod:
    iter = 0
    def generateMatrix(self, variables, constants):
        self.table = numpy.zeros((constants + 1, variables + constants + 1))
        self.signs = []

    def add_ObjectiveFunction(self, eq, co):
        s = eq.split()
        for i in range(co):
            self.table[0][i] = -int(s[i])

    def add_IneqFunctions(self, ineq, co, num):
        s = ineq.split()
        for i in range(co):
            self.table[num + 1][i] = s[i]
        self.table[num + 1][co + num] = 1
        self.table[num + 1][-1] = s[-1]
        self.signs += [s[-2]]

    def check_Optimality(self, co):
        min = [self.table[0][0], 0]
        for i in range(co):
            if self.table[0][i] < min[0] and self.table[0][i] < 0:
                min[0] = self.table[0][i]
                min[1] = i
        if min[0] >= 0:
            print('Текущее базисное решение оптимально!')
            self.iter = 1
        else:
            self.LeadingColumn = min[1]
            self.check_InfSolutions()

    def check_InfSolutions(self):
        p = 1
        for i in range(1, len(self.table)):
            if self.table[i][self.LeadingColumn] > 0:
                p = 0
            else:
                print('Значение задачи ЛП не ограничено!')
                self.iter = 2
        if p != 1:
            min = [self.table[1][-1] / self.table[1][self.LeadingColumn], 1]
            for i in range(1, len(self.table)):
                if self.table[i][self.LeadingColumn] > 0:
                    a = self.table[i][-1] / self.table[i][self.LeadingColumn]
                    if min[0] > a:
                        min[0] = a
                        min[1] = i

            self.LeadingString = min[1]
            self.tableConversion()

    def tableConversion(self):
        n = len(self.table[0])
        m = len(self.table)
        table = numpy.zeros((m, n))
        a = self.table[self.LeadingString][self.LeadingColumn]
        for i in range(m):
            if i != self.LeadingString:
                for j in range(n-1):
                    table[i][j] = self.table[i][j] - (self.table[self.LeadingString][j] / a * self.table[i][self.LeadingColumn])
                table[i][n-1] = self.table[i][n-1] - (self.table[self.LeadingString][n-1] / a * self.table[i][self.LeadingColumn])
            else:
                for j in range(n - 1):
                    table[i][j] = self.table[i][j] / a
                table[i][n-1] = self.table[i][n-1] / a
        print(table)
        self.table = table


if __name__ == "__main__":
    main()
