from typing import Any, Callable, TypeVar
from time import perf_counter
import csv

T = TypeVar("T")


def save_results(data: dict[str, Any]):
    with open("results.csv", mode='a', newline='') as f:
        write_csv = csv.DictWriter(f, fieldnames=["method", "problem", "problem_size",  "goal", "time", "iterations"], delimiter=";")
        if f.tell() == 0:
            write_csv.writeheader()

        write_csv.writerow(data)


def results(name: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            print("-" * 32, name if name else func.__name__, sep="\n")

            start_time = perf_counter()
            f = func(*args, **kwargs)
            end_time = perf_counter()
            total_time = end_time - start_time

            results = {
                "method": name,
                "problem": f.__getattribute__('p'),
                "problem_size": f.__getattribute__('p').__getattribute__('m'),
                "goal": f.__getattribute__('current_goal'),
                "time": total_time,
                "iterations": f.__getattribute__('stopped_in')
                }

            print(f"{'stopped in:':.>15} {f.__getattribute__('stopped_in')} iterations")
            print(f"{'timer:':.>15} {total_time:.6f} seconds")
            print(f"{'problem:':.>15} {f.__getattribute__('p')}")
            print(f"{'result:':.>15} {f:indent_triplets_goal}")
            print(f"{'goal:':.>15} {f.__getattribute__('current_goal')}")

            print(
                f"{'sum-T:':.>15}",
                f"{f.__getattribute__('p').__getattribute__('t')}",
            )
            print(
                f"{'m:':.>15}",
                f"{f.__getattribute__('p').__getattribute__('m')}",
            )
            print(f"{'seed:':.>15} {args[0].__getattribute__('seed')}")

            save_results(results)

            return f

        return wrapper

    return decorator
