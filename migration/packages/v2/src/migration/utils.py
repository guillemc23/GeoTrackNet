import asyncio
import logging
import time
from typing import Any, Callable

from rich.console import Console
from rich.logging import RichHandler

# Configure the logging

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %X]",
    handlers=[RichHandler(console=console)],
)

# Create a logger instance
logger = logging.getLogger("rich")


def get_duration_str(start: float) -> str:
    """Get human readable duration string from start time."""
    duration = time.time() - start
    if duration > 1:
        duration_str = f"{duration:,.3f}s"
    elif duration > 1e-3:
        duration_str = f"{round(duration * 1e3)}ms"
    elif duration > 1e-6:
        duration_str = f"{round(duration * 1e6)}us"
    else:
        duration_str = f"{duration * 1e9}ns"
    return duration_str


def timed(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator log test start and end time of a function.

    :param fn: Function to decorate
    :return: Decorated function

    Example:
        >>> @timed
        >>> def test_fn():
        >>>     time.sleep(1)
        >>> test_fn()

    """

    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        # print(f'Running {fn.__name__}...')
        ret = fn(*args, **kwargs)
        duration_str = get_duration_str(start)
        logger.info(f"Finished {fn.__name__} in {duration_str}")
        return ret

    async def wrapped_fn_async(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        # print(f'Running {fn.__name__}...')
        ret = await fn(*args, **kwargs)
        duration_str = get_duration_str(start)
        logger.info(f"Finished {fn.__name__} in {duration_str}")
        return ret

    if asyncio.iscoroutinefunction(fn):
        return wrapped_fn_async
    else:
        return wrapped_fn
