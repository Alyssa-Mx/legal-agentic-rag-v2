import logging
import queue
import threading
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class AsyncTaskQueue:
    """
    后台线程任务队列。
    主链路先返回答案，长期记忆写入等耗时任务在后台线程执行，不阻塞主回复延迟。
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._handlers: Dict[str, Callable] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[AsyncWorker] 后台任务线程已启动")

    def register(self, task_name: str, handler: Callable) -> None:
        """注册任务处理器。"""
        self._handlers[task_name] = handler

    def enqueue(self, task_name: str, payload: Dict[str, Any]) -> None:
        """将任务入队（非阻塞）。"""
        self._queue.put({"task_name": task_name, "payload": payload})
        logger.debug("[AsyncWorker] 任务入队: %s", task_name)

    def _run(self) -> None:
        while True:
            try:
                task = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            name = task["task_name"]
            handler = self._handlers.get(name)
            if not handler:
                logger.warning("[AsyncWorker] 未找到处理器: %s", name)
                self._queue.task_done()
                continue

            try:
                handler(task["payload"])
                logger.debug("[AsyncWorker] 任务完成: %s", name)
            except Exception as e:
                logger.error("[AsyncWorker] 任务 %s 执行失败: %s", name, e, exc_info=True)
            finally:
                self._queue.task_done()


async_worker = AsyncTaskQueue()
