import random


# random.seed(42)


class Problem:
    def __init__(self, elements: list[int]):
        self.elements = elements
        self.t = self.sum_t()

    def __str__(self):
        return ", ".join([str(number) for number in self.elements])

    def check_problem(self):
        if not len(self.elements) >= 3:
            ...

        if not len(self.elements) % 3:
            ...

    def sum_t(self) -> float:
        return sum(self.elements)/len(self.elements)


class Solution:
    def __init__(self, problem: Problem):
        self.p = problem
        self.multiset: list[list[int]]
        self.make_multiset(self.p.elements.copy())
       
    def make_multiset(self, elements: list[int]):
        self.multiset = [
            elements[i: i+3]
            for i in range(0, len(elements), 3)
        ]

    def goal(self) -> float:
        return sum(
            [1 for triplet in self.multiset if sum(triplet) == self.p.t]
        )/len(self.multiset)

    def random_shuffle(self):
        new_solution = self.p.elements.copy()
        random.shuffle(new_solution)
        self.make_multiset(new_solution)

    def random_modify(self):
        midx1 = random.randint(0, len(self.multiset) - 1)
        midx2 = random.randint(0, len(self.multiset) - 1)
        while midx2 == midx1:
            midx2: int = random.randint(0, len(self.multiset) - 1)

        elem1 = random.randint(0, 2)
        elem2 = random.randint(0, 2)

        (self.multiset[midx1][elem1],
         self.multiset[midx2][elem2]) = (self.multiset[midx2][elem2],
                                         self.multiset[midx1][elem1])


p = Solution(Problem([1, 2, 3, 4, 5, 6]))
p.random_shuffle()
print(p.multiset)
p.random_modify()
print(p.multiset)
