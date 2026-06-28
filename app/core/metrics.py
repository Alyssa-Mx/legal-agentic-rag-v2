import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MetricsRecorder:
    """
    简单的内存指标收集器。
    记录每轮对话的关键指标，支持汇总统计。
    生产环境可替换为写入 Prometheus / ClickHouse 等。
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def record(self, data: Dict[str, Any]) -> None:
        entry = {"_ts": time.time(), **data}
        self._records.append(entry)
        logger.info(
            "[Metrics] turns=%d | kb=%s | fallback=%s | repaired=%s | hallucination_risk=%s",
            len(self._records),
            data.get("used_kb"),
            data.get("kb_fallback_triggered"),
            data.get("answer_revised"),
            data.get("hallucination_risk", "n/a"),
        )

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def summary(self) -> Dict[str, Any]:
        if not self._records:
            return {}
        total = len(self._records)
        return {
            "total_turns": total,
            "kb_usage_rate": round(
                sum(1 for r in self._records if r.get("used_kb")) / total, 3
            ),
            "kb_fallback_rate": round(
                sum(1 for r in self._records if r.get("kb_fallback_triggered")) / total, 3
            ),
            "answer_repair_rate": round(
                sum(1 for r in self._records if r.get("answer_revised")) / total, 3
            ),
            "high_hallucination_rate": round(
                sum(1 for r in self._records if r.get("hallucination_risk") == "high") / total, 3
            ),
        }


metrics = MetricsRecorder()
