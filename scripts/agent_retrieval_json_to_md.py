"""将 evaluate_agent_retrieval 输出的 JSON 转为可读 Markdown 报告。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ARTICLE_ID_PATTERN = re.compile(r"article_id\s*=\s*(\d+)")


def load_corpus(path: Path) -> Dict[int, Dict[str, str]]:
    corpus: Dict[int, Dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            aid = int(row["id"])
            content = (row.get("content") or "").replace("\\n", "\n").strip()
            corpus[aid] = {
                "name": row.get("name") or "",
                "content": content,
            }
    return corpus


def trunc_text(text: str, n: int = 30) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[:n] + "…"


def fmt_article(aid: int, corpus: Dict[int, Dict[str, str]], n: int) -> str:
    info = corpus.get(aid)
    if not info:
        return f"`{aid}`（语料中未找到）"
    name = info["name"]
    snippet = trunc_text(info["content"], n)
    return f"`{aid}` {name} — {snippet}"


def fmt_ids(ids: List[int], corpus: Dict[int, Dict[str, str]], n: int) -> List[str]:
    return [f"- {fmt_article(aid, corpus, n)}" for aid in ids]


def fmt_args(args: Dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        if k == "article_ids" and isinstance(v, list):
            parts.append(f"{k}={v}")
        elif isinstance(v, str) and len(v) > 60:
            parts.append(f'{k}="{v[:60]}…"')
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def args_key(name: str, args: Dict[str, Any]) -> Tuple[str, str]:
    return name, json.dumps(args, sort_keys=True, ensure_ascii=False)


def build_returned_lookup(tool_calls_flat: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[int]]:
    lookup: Dict[Tuple[str, str], List[int]] = {}
    for tc in tool_calls_flat:
        lookup[args_key(tc.get("name", ""), tc.get("args") or {})] = tc.get("returned_ids") or []
    return lookup


def ids_from_preview(preview: str) -> List[int]:
    return [int(x) for x in ARTICLE_ID_PATTERN.findall(preview or "")]


def parse_trace_steps(trace_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 trace_events 解析为按 ReAct 回合分组的步骤（含并行标记）。"""
    steps: List[Dict[str, Any]] = []
    i = 0
    while i < len(trace_events):
        ev = trace_events[i]
        if ev.get("event") != "agent_step":
            i += 1
            continue
        tool_calls = ev.get("tool_calls") or []
        if not tool_calls:
            i += 1
            continue
        observations: List[Dict[str, Any]] = []
        i += 1
        while i < len(trace_events) and trace_events[i].get("event") == "tool_observation":
            observations.append(trace_events[i])
            i += 1
        steps.append({
            "step": ev.get("step"),
            "budget_left": ev.get("budget_left"),
            "phase": ev.get("phase"),
            "reason": (ev.get("reason") or "").strip(),
            "t_ms": ev.get("t_ms"),
            "tool_calls": tool_calls,
            "observations": observations,
            "parallel": len(tool_calls) > 1,
        })
    return steps


def match_observation(
    tc: Dict[str, Any],
    observations: List[Dict[str, Any]],
    used: set,
) -> Optional[Dict[str, Any]]:
    name = tc.get("name", "")
    args = tc.get("args") or {}
    for idx, obs in enumerate(observations):
        if idx in used:
            continue
        if obs.get("tool") == name and (obs.get("args") or {}) == args:
            used.add(idx)
            return obs
    for idx, obs in enumerate(observations):
        if idx in used:
            continue
        if obs.get("tool") == name:
            used.add(idx)
            return obs
    return None


def render_trace_trajectory(
    q: Dict[str, Any],
    corpus: Dict[int, Dict[str, str]],
    snippet_chars: int,
) -> List[str]:
    trace_events = q.get("trace_events") or []
    if not trace_events:
        return []

    steps = parse_trace_steps(trace_events)
    if not steps:
        return []

    returned_lookup = build_returned_lookup(q.get("tool_calls") or [])
    lines = ["**工具调用轨迹**（按 ReAct 回合；同回合多调用 = 并行）", ""]

    for st in steps:
        mode = "并行" if st["parallel"] else "串行"
        n_calls = len(st["tool_calls"])
        phase = st.get("phase") or ""
        phase_s = f"，phase={phase}" if phase else ""
        lines.append(
            f"#### 回合 {st['step']} · {mode}"
            f"（{n_calls} 个调用，budget_left={st.get('budget_left', '?')}{phase_s}，"
            f"t={st.get('t_ms', '?')}ms）"
        )
        lines.append("")
        if st["reason"]:
            lines.append(f"> Reason: {st['reason']}")
            lines.append("")

        used_obs: set = set()
        for j, tc in enumerate(st["tool_calls"], 1):
            name = tc.get("name", "?")
            args = tc.get("args") or {}
            lines.append(f"{j}. `{name}`({fmt_args(args)})")

            obs = match_observation(tc, st["observations"], used_obs)
            returned = returned_lookup.get(args_key(name, args), [])
            if not returned and obs:
                returned = ids_from_preview(obs.get("preview") or "")

            if returned:
                lines.append("   - 返回法条：")
                for aid in returned:
                    lines.append(f"     - {fmt_article(aid, corpus, snippet_chars)}")
            elif name == "submit_ranking":
                aids = args.get("article_ids") or []
                if aids:
                    lines.append("   - 提交法条：")
                    for aid in aids:
                        lines.append(f"     - {fmt_article(aid, corpus, snippet_chars)}")
            elif obs:
                preview = trunc_text((obs.get("preview") or "").replace("\\n", " "), 80)
                elapsed = obs.get("elapsed_ms", "?")
                lines.append(f"   - 返回：{preview}（{elapsed}ms）")
            lines.append("")

        if st["parallel"] and len(st["observations"]) > 1:
            lines.append(
                f"*本回合 {len(st['observations'])} 条 observation 并行返回，"
                f"完成顺序见 elapsed_ms*"
            )
            lines.append("")

    return lines


def render_flat_trajectory(
    q: Dict[str, Any],
    corpus: Dict[int, Dict[str, str]],
    snippet_chars: int,
) -> List[str]:
    tool_calls = q.get("tool_calls") or []
    if not tool_calls:
        return []

    lines = [
        "**工具调用轨迹**（扁平列表，无法区分并行/串行；需 `--trace` 重跑）",
        "",
    ]
    for j, tc in enumerate(tool_calls, 1):
        name = tc.get("name", "?")
        args = tc.get("args") or {}
        lines.append(f"{j}. `{name}`({fmt_args(args)})")
        returned = tc.get("returned_ids") or []
        if returned:
            lines.append("   - 返回法条：")
            for aid in returned:
                lines.append(f"     - {fmt_article(aid, corpus, snippet_chars)}")
        elif name == "submit_ranking":
            aids = args.get("article_ids") or []
            if aids:
                lines.append("   - 提交法条：")
                for aid in aids:
                    lines.append(f"     - {fmt_article(aid, corpus, snippet_chars)}")
        lines.append("")
    return lines


def render_report(data: dict, corpus: Dict[int, Dict[str, str]], snippet_chars: int) -> str:
    lines: List[str] = []
    label = data.get("label") or data.get("variant") or "Agent Retrieval"
    cfg = data.get("config") or {}
    summary = data.get("summary") or {}

    lines += [
        f"# {label}",
        "",
        "## 配置",
        "",
        f"- chat_model: `{cfg.get('chat_model', 'N/A')}`",
        f"- max_steps: {cfg.get('max_steps', 'N/A')}",
        f"- default_k: {cfg.get('default_k', 'N/A')}",
        f"- trunc_chars: {cfg.get('trunc_chars', 'N/A')}",
        f"- rerank: {cfg.get('rerank', 'N/A')}",
        f"- started_at: {cfg.get('started_at', 'N/A')}",
        "",
        "## 汇总指标",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
    ]
    for key in (
        "n", "hit@1", "hit@3", "hit@5", "recall@5", "mrr", "ndcg@5",
        "raw_hit@5", "submit_rate", "avg_tool_calls", "avg_kb_calls",
        "lat_avg", "lat_p50", "lat_p95", "errors",
    ):
        if key in summary:
            val = summary[key]
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.4f} |")
            else:
                lines.append(f"| {key} | {val} |")

    lines += ["", "## 逐题详情", ""]

    for i, q in enumerate(data.get("per_question") or [], 1):
        qid = q.get("query_id", "?")
        hit5 = (q.get("metrics") or {}).get("hit@5", "?")
        submitted = "✓" if q.get("submitted") else "✗"
        lines += [
            f"### {i}. Q{qid}（hit@5={hit5}，submit={submitted}）",
            "",
            f"**问题**：{q.get('question', '')}",
            "",
        ]

        gold_ids = q.get("gold_ids") or []
        gold_names = q.get("gold_names") or []
        if gold_ids:
            lines.append("**金标法条**：")
            if gold_names and len(gold_names) == len(gold_ids):
                for aid, name in zip(gold_ids, gold_names):
                    snippet = trunc_text((corpus.get(aid) or {}).get("content", ""), snippet_chars)
                    lines.append(f"- `{aid}` {name} — {snippet}")
            else:
                lines.extend(fmt_ids(gold_ids, corpus, snippet_chars))
            lines.append("")

        sub = q.get("submitted_ranking") or []
        if sub:
            lines.append("**提交排序**：")
            lines.extend(fmt_ids(sub, corpus, snippet_chars))
            lines.append("")
            reason = (q.get("submit_reason") or "").strip()
            if reason:
                lines.append(f"**提交理由**：{reason}")
                lines.append("")

        raw = q.get("raw_retrieved_ids") or []
        if raw:
            lines.append(f"**召回池**（{len(raw)} 条，去重顺序）：")
            lines.extend(fmt_ids(raw, corpus, snippet_chars))
            lines.append("")

        if q.get("trace_events"):
            lines.extend(render_trace_trajectory(q, corpus, snippet_chars))
            lines.append("")
        else:
            lines.extend(render_flat_trajectory(q, corpus, snippet_chars))

        m = q.get("metrics") or {}
        lines.append(
            f"*latency={q.get('latency_s', '?')}s | "
            f"tool_calls={q.get('tool_call_count', '?')} | "
            f"kb_calls={q.get('kb_call_count', '?')} | "
            f"mrr={m.get('mrr', '?')} | recall@5={m.get('recall@5', '?')}*"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent retrieval JSON → Markdown")
    parser.add_argument(
        "--input",
        default="results/agent_retrieval/agent_v2.json",
        help="输入 JSON 路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 Markdown 路径（默认同名 .md）",
    )
    parser.add_argument(
        "--corpus",
        default="data/lecoqa/corpus.jsonl",
        help="法条语料路径",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=30,
        help="法条正文截取字数",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".md")
    corpus_path = Path(args.corpus)

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    corpus = load_corpus(corpus_path)
    md = render_report(data, corpus, args.snippet_chars)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path} ({len(md):,} chars, {len(data.get('per_question') or [])} questions)")


if __name__ == "__main__":
    main()
