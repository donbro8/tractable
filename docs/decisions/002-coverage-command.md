# 002 — Coverage Command File-Set Correction

The TASK-3.3.4 acceptance criterion listed `tractable/graph/mutations.py` as an explicit
`--include` target in the `coverage report` command. That file does not exist — the mutation
helpers (`apply_mutations`, `_create_entity`, `_create_edge`, `_check_for_orphaned_entities`)
are methods on `FalkorDBTemporalGraph` inside `tractable/graph/temporal_graph.py`. Because
the non-existent file was silently excluded from the report, the 80% threshold was being
evaluated against a smaller file-set than intended (7 files instead of 8).

**Original command (TASK-3.3.4 AC1):**
```
coverage report --fail-under=80 --include="tractable/graph/mutations.py,tractable/graph/temporal_graph.py,tractable/agent/tools/code_editor.py,tractable/agent/nodes/execute.py,tractable/agent/nodes/review.py,tractable/reactivity/webhook_receiver.py,tractable/reactivity/ingestion_pipeline.py,tractable/reactivity/notification_router.py"
```
Result: exits 0 (93% across 7 files — `mutations.py` silently absent from report).

**Corrected command:**
```
coverage report --fail-under=80 --include="tractable/graph/temporal_graph.py,tractable/agent/tools/code_editor.py,tractable/agent/nodes/execute.py,tractable/agent/nodes/review.py,tractable/reactivity/webhook_receiver.py,tractable/reactivity/ingestion_pipeline.py,tractable/reactivity/notification_router.py"
```
Result: exits 0 (93% across the same 7 files, which correctly represents all mutation logic).

No additional tests were needed — the corrected command already passes at 93%, well above the 80%
threshold. TASK-3a.2.1 (extracting `mutations.py`) proceeds as a structural alignment task only;
coverage target validity is not a driver for that extraction.
