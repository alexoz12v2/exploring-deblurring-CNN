import time
from contextlib import contextmanager
from multiprocessing import Value

class AtomicBool:
    def __init__(self, manager, initial=False):
        self._value = Value("b", initial)  # 'b' for boolean
        self._shutdown = Value("b", False)
        self._lock = manager.Lock()

    def load(self) -> bool:
        with self._lock:
            return self._value.value

    def store(self, value: bool) -> None:
        with self._lock:
            self._value.value = value

    def should_close(self) -> bool:
        with self._lock:
            return self._shutdown.value

    def request_close(self) -> None:
        with self._lock:
            self._shutdown.value = True

    def set(self):
        self.store(True)

    def reset(self):
        self.store(False)

    def compare_and_swap(self, expected: bool, value: bool) -> bool:
        with self._lock:
            if self._value.value == expected:
                self._value.value = value
                return True
            else:
                return False


@contextmanager
def lock_bool(mtx: AtomicBool, *, timeout_millis: float):
    start_time = time.time()
    timeout_secs = timeout_millis / 1000
    acquired = False  # Track whether the lock was successfully acquired
    try:
        while not acquired:
            if mtx.compare_and_swap(False, True):
                acquired = True
            elif time.time() - start_time > timeout_secs:
                yield False  # Lock not acquired within timeout
                return  # Exit the context manager without releasing
            else:
                time.sleep(0.0001)
        yield True  # Lock successfully acquired
    finally:
        if acquired:  # Only release the lock if it was acquired
            if not mtx.compare_and_swap(True, False):
                raise ValueError("Something went wrong while releasing the lock")

