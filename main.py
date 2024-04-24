import random
from itertools import permutations


class Problem:
    def __init__(self, elements: list[int]):
        self.elements = elements
        self.t = self.sum_t()

    def __str__(self):
        return ", ".join([str(number) for number in self.elements])

    def check_problem(self):
        if not len(self.elements):
            ...

        elif not len(self.elements) % 3:
            ...

    def sum_t(self) -> float:
        return sum(self.elements)//(len(self.elements)/3)

    def random_shuffle(self):
        random.shuffle(self.elements)

    def goal(self, elements: list[int]):
        triplets_sum: list[int] = [
            sum(self.elements[idx:idx+3])
            for idx in range(0, len(self.elements), 3)
            ]

        self.current_goal = sum(
            [
                1 if x == self.t else 0.5/abs(self.t - x)
                for x in triplets_sum
            ]
        )/len(self.elements)*3


class Solution:
    def __init__(self, problem: Problem):
        self.p = problem
        self.multiset: list[int]
        self.make_multiset(self.p.elements.copy())

        self.current_goal: float
        self.goal()

    def __str__(self):
        return ", ".join(
            [
                f"{self.multiset[idx:idx+3]}"
                for idx in range(0, len(self.multiset), 3)
            ]
        )

    def make_multiset(self, elements: list[int]):
        self.multiset = elements.copy()
        self.goal()

    def goal(self):
        triplets_sum: list[int] = [
            sum(self.multiset[idx:idx+3])
            for idx in range(0, len(self.multiset), 3)
            ]

        self.current_goal = sum(
            [
                1 if x == self.p.t else 0.5/abs(self.p.t - x)
                for x in triplets_sum
            ]
        )/len(self.multiset)*3

    def random_shuffle(self):
        new_solution = self.p.elements.copy()
        random.shuffle(new_solution)
        self.make_multiset(new_solution)

    def random_modify(self):
        midx1 = random.randint(0, len(self.multiset) - 1)
        midx2 = random.randint(0, len(self.multiset) - 1)

        while midx2//3 == midx1//3:
            midx2: int = random.randint(0, len(self.multiset) - 1)

        (self.multiset[midx1],
         self.multiset[midx2]) = (self.multiset[midx2],
                                  self.multiset[midx1])
        self.goal()

    def generate_neighbours(self):
        self.neighbours: list[list[int]] = []
        self.neighbours.append(self.multiset)

        for eidx1 in range(0, len(self.multiset)):
            for eidx2 in range(0, len(self.multiset)):
                if not (eidx1//3 == eidx2//3):
                    new_neighbour = self.multiset.copy()
                    (new_neighbour[eidx1], new_neighbour[eidx2]) = (
                        new_neighbour[eidx2], new_neighbour[eidx1])
                    if new_neighbour not in self.neighbours:

                        self.neighbours.append(new_neighbour)

    def best_neighbour(self):
        new_neighbour = Solution(Problem(self.multiset.copy()))

        for neighbour in self.neighbours:
            new_neighbour.make_multiset(neighbour)
            if new_neighbour.current_goal > self.current_goal:
                self.make_multiset(neighbour)
        return self.multiset


class Solver:
    def __init__(self, problem: Problem, seed: int | str | float | None, shuffle: bool):
        self.p = problem
        self.p.random_shuffle()
        random.seed(seed)

    def brute_force(self):
        solution = Solution(self.p)
        best_solution = Solution(self.p)
        perm = permutations(solution.multiset)

        for _ in range(1000000):
            solution.make_multiset(list(perm.__next__()))

            if best_solution.current_goal < solution.current_goal:
                best_solution.make_multiset(solution.multiset.copy())

            if best_solution.current_goal == 1.0:
                break

        return best_solution

    def deterministic_hill_climb(self):
        solution = Solution(self.p)

        for _ in range(5040):
            solution.generate_neighbours()
            solution.make_multiset(solution.best_neighbour())

            if solution.current_goal == 1:
                break

        return solution

    def random_hill_climb(self):
        solution = Solution(self.p)

        for _ in range(5040):
            solution.random_modify()
            solution.make_multiset(solution.multiset)

            if solution.current_goal == 1:
                break

        return solution


if __name__ == "__main__":
    s = Solver(Problem([1, 2, 3, 3, 6, 4, 8, 9, 9, 9, 10, 16]), 42, False)
    print(s.p)
    result = s.brute_force()
    print(result, result.current_goal)

    print(s.p)
    result = s.random_hill_climb()
    print(result, result.current_goal)

    print(s.p)
    result = s.deterministic_hill_climb()
    print(result, result.current_goal)
