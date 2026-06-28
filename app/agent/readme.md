graph.py
  └─ 使用 AgentNodes（来自 nodes.py）注册所有节点

nodes.py（AgentNodes 类）
  ├─ 持有 router.py（QueryRouter）
  ├─ 持有 rewriter.py（QueryRewriter）
  ├─ 持有 inspector.py（AnswerInspector）
  ├─ 持有 context_resolver.py（ContextResolver）
  └─ 持有 memory/manager.py（MemoryManager）

state.py
  └─ 被 graph.py 和 nodes.py 共同使用，本身无依赖