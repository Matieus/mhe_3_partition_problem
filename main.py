import random
from itertools import permutations
from decorators import results, timer


class Problem:
    def __init__(self, elements: list[int]):
        self.elements = elements
        self.t = self.sum_t()
        self.m = int(len(self.elements)/3)
        self.sum = sum(self.elements)
        self.check_problem()

    def __str__(self):
        return ", ".join([str(number) for number in self.elements])

    def check_problem(self):
        if len(self.elements) < 6:
            raise ValueError("The problem length cannot be zero")

        if not len(self.elements) % 3 == 0:
            raise ValueError(
                "Invalid problem length. The length should be divisible by 3")

        if self.sum != self.t*self.m:
            print(
                f"Problem:  {self.elements}",
                "There is no partition such that for all triplets \
                    the sum of the elements in each triplet is equal to T",
                f"sum != m*T {sum(self.elements)} != {self.t*self.m}",
                sep="\n"
                )

    def sum_t(self) -> float:
        return sum(self.elements) // (len(self.elements) / 3)

    def random_shuffle(self):
        random.shuffle(self.elements)


class RandomProblem(Problem):
    def __init__(
            self, m: int,
            t: int,
            *,
            min_value: int | None = None,
            max_value: int | None = None,
            attempts: int = 20,
            seed: int = 42
            ):
        self.elements: list[int] = []
        self.m = m
        self.t = t

        self.min = min_value if isinstance(min_value, int) else int(self.t/4)
        self.max = max_value if isinstance(max_value, int) else int(self.t/2)
        self.attempts = attempts
        self.seed = seed
        random.seed(self.seed)

        self._generate_elements()
        self.sum = sum(self.elements)

        self.check_problem()

    def _generate_elements(self):
        for _ in range(self.m*3):
            self.elements.append(random.randint(self.min, self.max))

        for _ in range(self.attempts):
            idx = random.randint(0, len(self.elements) - 1)
            self.elements[idx] = random.randint(self.min, self.max)

            if sum(self.elements) / self.m == self.t:
                break


class Solution:
    def __init__(self, problem: Problem):
        self.p = problem
        self.multiset: list[int]
        self.make_multiset(self.p.elements.copy())

        self.current_goal = self.goal(self.multiset)

    def __str__(self):
        return ", ".join([str(number) for number in self.multiset])

    def __repr__(self) -> str:
        return ", ".join(
            [
                f"{self.multiset[idx:idx+3]}"
                for idx in range(0, len(self.multiset), 3)

                ]
        )

    def make_multiset(self, elements: list[int]):
        self.multiset = elements.copy()
        self.current_goal = self.goal(self.multiset)

    def goal(self, elements: list[int]) -> float:
        triplets_sum: list[int] = [
            sum(elements[idx: idx + 3])
            for idx in range(0, len(elements), 3)
        ]

        return sum(
                [
                    1 if x == self.p.t
                    else 0.5 / abs(self.p.t - x)
                    for x in triplets_sum
                    ]
                    ) / self.p.m

    def random_shuffle(self):
        random.shuffle(self.multiset)
        self.current_goal = self.goal(self.multiset)

    def random_modify(self):
        midx1 = random.randint(0, len(self.multiset) - 1)
        midx2 = random.randint(0, len(self.multiset) - 1)

        while midx2 // 3 == midx1 // 3:
            midx2: int = random.randint(0, len(self.multiset) - 1)

        (self.multiset[midx1], self.multiset[midx2]) = (
            self.multiset[midx2],
            self.multiset[midx1],
        )
        self.current_goal = self.goal(self.multiset)

    def neighbours_generator(self):
        yield self.multiset
        for eidx1 in range(0, len(self.multiset)):
            for eidx2 in range(0, len(self.multiset)):
                if not (eidx1 // 3 == eidx2 // 3):
                    new_neighbour = self.multiset.copy()
                    (new_neighbour[eidx1],
                     new_neighbour[eidx2]) = (
                        new_neighbour[eidx2],
                        new_neighbour[eidx1],
                    )
                    yield new_neighbour

    def best_neighbour(self) -> list[int]:
        """
        The function will search for the best neighbor
        of the current solution from generator

        Parameters:

        Returns:
        Solution: best neighbour for current solution
        """
        best_neighbour = self.multiset
        best_goal = self.goal(best_neighbour)

        for neighbour in self.neighbours_generator():
            neighbour_goal = self.goal(neighbour)

            if neighbour_goal > best_goal:
                best_neighbour = neighbour
                best_goal = neighbour_goal

        return best_neighbour

    def simplified_neighbors_generator(self):
        yield self.multiset
        for eidx in range(0, len(self.multiset)):
            new_neighbour = self.multiset.copy()
            (new_neighbour[0], new_neighbour[eidx]) = (
                new_neighbour[eidx],
                new_neighbour[0],
            )
            yield new_neighbour

    def best_simplified_neighbour(self):
        best_neighbour = self.multiset
        best_goal = self.goal(best_neighbour)

        for neighbour in self.simplified_neighbors_generator():
            neighbour_goal = self.goal(neighbour)

            if neighbour_goal > best_goal:
                best_neighbour = neighbour
                best_goal = neighbour_goal

        return best_neighbour


class Solver:
    def __init__(
        self,
        problem: Problem | RandomProblem,
        *,
        seed: int | str | float | None = None,
        shuffle: bool = True,
        iterations: int = 362_880,
        stop_on_best_solution: bool = True
    ):

        self.p: Problem = problem
        self.seed: int | str | float | None = seed
        self.shuffle: bool = shuffle
        self.iterations: int = iterations
        self.stop_on_best_solution: bool = stop_on_best_solution

        if seed:
            random.seed(seed)
        if shuffle:
            self.p.random_shuffle()

    def _stopper(self, goal: float) -> bool:
        return goal == 1.0 and self.stop_on_best_solution

    @results("BRUTE FORCE")
    @timer
    def brute_force(self) -> Solution:
        solution = Solution(self.p)

        best_solution = Solution(self.p)
        perm = permutations(solution.multiset)

        for _ in range(self.iterations):
            solution.make_multiset(list(perm.__next__()))

            if best_solution.current_goal < solution.current_goal:
                best_solution.make_multiset(solution.multiset.copy())

            if self._stopper(best_solution.current_goal):
                print(f"{'stopped in:':.>15} {_} iterations")
                break
        return best_solution

    @results("DETERMINISTIC HILL CLIMB")
    @timer
    def deterministic_hill_climb(self) -> Solution:
        best_solution = Solution(self.p)

        for _ in range(self.iterations):
            best_solution.make_multiset(
                best_solution.best_neighbour())

            if self._stopper(best_solution.current_goal):
                print(f"{'stopped in:':.>15} {_} iterations")
                break

        return best_solution

    @results("RANDOM HILL CLIMB")
    @timer
    def random_hill_climb(self) -> Solution:
        solution = Solution(self.p)
        best_solution = Solution(self.p)

        for _ in range(self.iterations):
            solution.random_modify()

            if best_solution.current_goal <= solution.current_goal:
                best_solution.make_multiset(solution.multiset)

            if self._stopper(best_solution.current_goal):
                print(f"{'stopped in:':.>15} {_} iterations")
                break

        return best_solution

    @results("TABU SEARCH")
    @timer
    def tabu_search(self) -> Solution:
        solution = Solution(self.p)
        best_solution = Solution(self.p)

        tabu_set: set[tuple[int, ...]] = set()
        tabu_set.add(tuple(solution.multiset))

        for _ in range(self.iterations):

            solution.make_multiset(
                best_solution.best_neighbour())

            if tuple(solution.multiset) in tabu_set:
                print(f"Ate my tail... in {_}")
                return best_solution

            if solution.current_goal > best_solution.current_goal:
                best_solution.make_multiset(solution.multiset.copy())

            tabu_set.add(tuple(solution.multiset.copy()))

            if self._stopper(best_solution.current_goal):
                print(f"{'stopped in:':.>15} {_} iterations")
                break

        return best_solution


if __name__ == "__main__":
    s = Solver(
        Problem([12, 28, 10, 10, 15, 25, 31, 1, 18, 22, 23, 5, 40, 4, 6, 33, 7, 10, 10, 30, 10, 25, 24, 1, 48, 1, 1, 42, 2, 6, 40, 4, 6, 21, 21, 8]),
        shuffle=True,
        stop_on_best_solution=True,
        )

    # result: Solution = s.brute_force()

    result: Solution = s.random_hill_climb()

    # result: Solution = s.deterministic_hill_climb()

    result: Solution = s.tabu_search()
