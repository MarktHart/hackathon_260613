"""Read-only dashboard over the pipeline's event log.

The pipeline writes two append-only JSONL files under `settings.state_dir`
(`blocks.jsonl`, `events.jsonl`). This package never writes them — it folds
them into a live view model (`aggregate`) and serves it over HTTP + SSE
(`server`). Token-only tiers get a dollar figure via `pricing`.
"""
