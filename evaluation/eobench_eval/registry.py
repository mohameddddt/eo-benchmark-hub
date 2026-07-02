from eobench_eval.base import Evaluator

_REGISTRY: dict[str, Evaluator] = {}


def register(evaluator: Evaluator) -> Evaluator:
    _REGISTRY[evaluator.task_type] = evaluator
    return evaluator


def get_evaluator(task_type: str) -> Evaluator:
    try:
        return _REGISTRY[task_type]
    except KeyError:
        raise KeyError(
            f"No evaluator registered for task type '{task_type}'. "
            f"Available: {sorted(_REGISTRY)}"
        ) from None
