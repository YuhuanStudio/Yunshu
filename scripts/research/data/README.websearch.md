Web-search query set dated 2026-10-07: 40 freshness queries, 40 documentation
lookups, 40 code/errors, 10 adversarial fixture slots. Questions are prompts,
not assertions or ground truth. Exclude `fixture_only` rows from live capture.

Capture through gpuq with `websearch_eval.py --capture QUERIES --out SNAPSHOT`.
The capture deliberately writes `gold_status=pending_review`, empty expected
answers and empty known-good URLs. Curate references (two independent sources
for news) before replay; the loader refuses uncurated rows. Keep page bodies and
snapshots private under docs/research/websearch or P5Plus build output, never in
git. Replay fixes the snapshot hash, uses temperature zero and the Anthropic
server-tool replay contract; rank/citation unit tests cover fake inputs without
network. `--dry-run` never scores answer quality. A paired accuracy gate requires
at least 200 answers; this 130-query set alone cannot satisfy that gate.

Current replay arms: plain snippets, local BM25/dense preparation, no-search
floor. Capture separate snapshots for Brave/Tavily/Exa/keyless to compare index
quality. Replay `--ranking-model` calls the local `/v1/embeddings` route only after
confirming the model is already loaded; unavailable ranking retains BM25. The
dedicated yv websearch smoke also proves the in-process server fusion path.
Semantic citation support and rubric grading require reviewed references and a
separate judge; mechanical substring validation does not prove claim support.

`websearch_adversarial.jsonl` contains 10 **authored fixtures**, not scraped web
news or a representative quality corpus. It covers visible instructions, inline
hiding, aria-hidden, hidden, CSS class/id, comments, script, zero-width and ANSI.
The tiny-model yv smoke replays these on the Anthropic route. Marker echoes are
a diagnostic proxy (quoting a marker can be a false positive); the 10 pairs cannot
approve the 200-pair quality gate or change the research default.
