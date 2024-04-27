from typing import Any, Callable, TypeVar


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
