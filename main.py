import random
from itertools import permutations
from typing import Any, Callable, TypeVar


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
            raise ValueError("Invalid problem length. The length should be divisible by 3")

        if self.sum != self.t*self.m:
            raise ValueError(
                f"sum != m*T {sum(self.elements)} != {self.t*self.m}"
                )

    def sum_t(self) -> float:
        return sum(self.elements) // (len(self.elements) / 3)

    def random_shuffle(self):
        random.shuffle(self.elements)


class Solution:
    def __init__(self, problem: Problem):
        self.p = problem
        self.multiset: list[int]
        self.make_multiset(self.p.elements.copy())

        self.current_goal: float = 0
        self.goal()

    def __str__(self):
        return ", ".join([str(number) for number in self.multiset])

    def __repr__(self) -> str:
        return ", ".join(
            [f"{self.multiset[idx:idx+3]}" for idx in range(0, len(self.multiset), 3)]
        )

    def make_multiset(self, elements: list[int]):
        self.multiset = elements.copy()
        self.goal()

    def goal(self):
        triplets_sum: list[int] = [
            sum(self.multiset[idx : idx + 3]) for idx in range(0, len(self.multiset), 3)
        ]

        self.current_goal = (
            sum([1 if x == self.p.t else 0.5 / abs(self.p.t - x) for x in triplets_sum])
            / len(self.multiset)
            * 3
        )

    def random_shuffle(self):
        new_solution = self.p.elements.copy()
        random.shuffle(new_solution)
        self.make_multiset(new_solution)

    def random_modify(self):
        midx1 = random.randint(0, len(self.multiset) - 1)
        midx2 = random.randint(0, len(self.multiset) - 1)

        while midx2 // 3 == midx1 // 3:
            midx2: int = random.randint(0, len(self.multiset) - 1)

        (self.multiset[midx1], self.multiset[midx2]) = (
            self.multiset[midx2],
            self.multiset[midx1],
        )
        self.goal()

    def generate_neighbours(self):
        self.neighbours: list[list[int]] = []
        self.neighbours.append(self.multiset)

        for eidx1 in range(0, len(self.multiset)):
            for eidx2 in range(0, len(self.multiset)):
                if not (eidx1 // 3 == eidx2 // 3):
                    new_neighbour = self.multiset.copy()
                    (new_neighbour[eidx1], new_neighbour[eidx2]) = (
                        new_neighbour[eidx2],
                        new_neighbour[eidx1],
                    )
                    if new_neighbour not in self.neighbours:

                        self.neighbours.append(new_neighbour)

    def best_neighbour(self):
        new_neighbour = Solution(Problem(self.multiset.copy()))
        best_neighbour = Solution(Problem(self.multiset.copy()))

        for neighbour in self.neighbours:
            new_neighbour.make_multiset(neighbour)
            if new_neighbour.current_goal > best_neighbour.current_goal:
                best_neighbour.make_multiset(neighbour)
        return best_neighbour.multiset


T = TypeVar("T")


def results(name: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            print("-" * 32, name if name else func.__name__, sep="\n")

            result = func(*args, **kwargs)
            print(f"{'problem:':.>12} {result.__getattribute__('p')}")
            print(f"{'result:':.>12} {result!r}")
            print(f"{'goal:':.>12} {result.__getattribute__('current_goal')}")
            print(
                f"{'sum-T:':.>12}",
                f"{result.__getattribute__('p').__getattribute__('t')}",
            )
            print(f"{'seed:':.>12} {args[0].__getattribute__('seed')}")

            return result

        return wrapper

    return decorator


class Solver:
    def __init__(
        self,
        problem: Problem,
        *,
        seed: int | str | float | None = None,
        shuffle: bool = True,
        iterations: int = 720,
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

    @results("BRUTE FORCE")
    def brute_force(self) -> Solution:
        solution = Solution(self.p)

        best_solution = Solution(self.p)
        perm = permutations(solution.multiset)

        for permutation in perm:
            solution.make_multiset(list(permutation))

            if best_solution.current_goal < solution.current_goal:
                best_solution.make_multiset(solution.multiset.copy())

            if best_solution.current_goal and self.stop_on_best_solution:
                break
        return best_solution

    @results("DETERMINISTIC HILL CLIMB")
    def deterministic_hill_climb(self) -> Solution:
        solution = Solution(self.p)

        for _ in range(self.iterations):
            solution.generate_neighbours()
            solution.make_multiset(solution.best_neighbour())

            if solution.current_goal and self.stop_on_best_solution:
                break

        return solution

    @results("RANDOM HILL CLIMB")
    def random_hill_climb(self) -> Solution:
        solution = Solution(self.p)

        for _ in range(self.iterations):
            solution.random_modify()
            solution.make_multiset(solution.multiset)

            if solution.current_goal and self.stop_on_best_solution:
                break

        return solution

    @results("TABU SEARCH")
    def tabu_search(self) -> Solution:
        solution = Solution(self.p)
        best_solution = Solution(self.p)

        tabu_set: list[list[int]] = []
        tabu_set.append(solution.multiset)

        for _ in range(self.iterations):
            solution.generate_neighbours()
            print(len(solution.neighbours))
            solution.neighbours = [
                neighbour for neighbour in solution.neighbours
                if neighbour not in tabu_set]

            if len(solution.neighbours) == 0:
                print("Ate my tail...", best_solution)
                return best_solution

            solution.make_multiset(solution.best_neighbour())
            if solution.current_goal >= best_solution.current_goal:
                best_solution.make_multiset(solution.multiset)

            if solution.multiset not in tabu_set:
                tabu_set.append(solution.multiset)

            if best_solution.current_goal and self.stop_on_best_solution:
                break

        return best_solution


if __name__ == "__main__":
    s = Solver(Problem([1, 2, 3, 4, 5, 7]), seed=42, shuffle=True, stop_on_best_solution=True)

    result: Solution = s.brute_force()

    result: Solution = s.random_hill_climb()

    result: Solution = s.deterministic_hill_climb()

    result: Solution = s.tabu_search()
