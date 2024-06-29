import random
from itertools import permutations
from decorators import results, timer
from math import exp
import argparse


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

    def __format__(self, f: str):
        if f == "indent":
            return ",".join(
                [
                    f"\n{'':>15} {self.multiset[idx:idx+3]}"
                    for idx in range(0, len(self.multiset), 3)

                    ]
            )
        if f == "indent_triplets_goal":
            return ",".join(
                [
                    f"\n{'':>16}" + f"{str(self.multiset[idx:idx+3]):<20} {sum(self.multiset[idx:idx+3])}"
                    for idx in range(0, len(self.multiset), 3)

                    ]
            )
        return self.__repr__()

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
        stop_on_best_solution: bool = True,
        temp_0: float | None = None
    ):

        self.p: Problem = problem
        self.seed: int | str | float | None = seed
        self.shuffle: bool = shuffle
        self.iterations: int = iterations
        self.stop_on_best_solution: bool = stop_on_best_solution

        if seed:
            random.seed(seed)
        if temp_0:
            self.temp_0: float = 1000.0
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

        tabu_list: list[list[int]] = []
        tabu_list.append(list(solution.multiset))
        best_goal = best_solution.current_goal

        for _ in range(self.iterations):
            for neighbour in best_solution.neighbours_generator():
                if neighbour not in tabu_list:
                    neighbour_goal = solution.goal(neighbour)

                    if neighbour_goal > best_goal:
                        best_neighbour = neighbour
                        best_goal = neighbour_goal

                        solution.make_multiset(best_neighbour)

            if solution.multiset not in tabu_list:
                tabu_list.append(solution.multiset)

            if solution.current_goal > best_solution.current_goal:
                best_solution.make_multiset(solution.multiset.copy())

            if self._stopper(best_solution.current_goal):
                print(f"{'stopped in:':.>15} {_} iterations")
                break

        return best_solution

    @results("SIM ANNEALING")
    @timer
    def sim_annealing(self) -> Solution:
        solution = Solution(self.p)
        best_solution = Solution(self.p)
        new_solution = Solution(self.p)

        for _ in range(1, self.iterations):
            new_solution.random_modify()

            if solution.current_goal >= new_solution.current_goal:
                solution.make_multiset(new_solution.multiset.copy())

                if solution.current_goal >= best_solution.current_goal:
                    best_solution.make_multiset(new_solution.multiset.copy())

            else:
                temp_i  = lambda i: self.temp_0 / i
                p = exp(-abs(new_solution.current_goal - solution.current_goal) / temp_i(_))

                if random.uniform(0, 1) < p:
                    solution.make_multiset(new_solution.multiset.copy())

            if self._stopper(best_solution.current_goal):
                print(f"{'stopped in:':.>15} {_} iterations")
                break

        return best_solution


def allowed_methods(method: str):
    methods = [k for k in Solver.__dict__ if not k.startswith("_")]
    print(methods)
    if method not in methods:
        raise argparse.ArgumentTypeError(f"not allowed method: {', '.join(method)}")

    return method


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="3-partition problem solver")

    parser.add_argument(
        '--shuffle', type=bool, help="shuffle problem", required=False)
    parser.add_argument(
        '--stop_on_best_solution', type=bool, help="Stop on found best solution (goal=1.0)", required=False)

    parser.add_argument(
        '--iterations', type=int, help="Set number of iterations", default=362_880)

    parser.add_argument(
        '--seed', type=int, help="Set seed for problem", required=False)

    parser.add_argument('--method', type=allowed_methods, help="name of method to run", required=True)


    args = parser.parse_args()

    s = Solver(
        Problem([12, 10, 15, 25, 3, 22, 25, 8, 14, 24, 23, 19]),
        shuffle=args.shuffle,
        stop_on_best_solution=args.stop_on_best_solution,
        iterations=args.iterations
        )

    results = s.__getattribute__(args.method)
    results()


"""

    # result: Solution = s.brute_force()

    # result: Solution = s.random_hill_climb()

    # result: Solution = s.deterministic_hill_climb()

    # result: Solution = s.tabu_search()
    # result = s.sim_annealing()

"""