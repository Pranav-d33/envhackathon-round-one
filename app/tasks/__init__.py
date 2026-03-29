from app.tasks.task1 import CPUSpikeTask
from app.tasks.task2 import DBConnectionPoolTask
from app.tasks.task3 import CascadingFailureTask

TASK_REGISTRY = {
    "task1": CPUSpikeTask(),
    "task2": DBConnectionPoolTask(),
    "task3": CascadingFailureTask(),
}

__all__ = ["TASK_REGISTRY", "CPUSpikeTask", "DBConnectionPoolTask", "CascadingFailureTask"]
