from typing import Generic, Iterable, Mapping, TypeVar

T = TypeVar("T")


class Registry(Generic[T], Mapping[str, T]):
    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(self, name=None):
        def return_fn(f):
            registered_name = name or f.__name__
            if registered_name in self._registry:
                raise KeyError(
                    f"{registered_name} already registered in registry {self._name}"
                )
            self._registry[registered_name] = f
            return f

        return return_fn

    def __repr__(self) -> str:
        return f"Registry({self._name})"

    def __iter__(self) -> Iterable[str]:
        return iter(self._registry)

    def __getitem__(self, key: str) -> T:
        return self._registry[key]()

    def __len__(self) -> int:
        return len(self._registry)
