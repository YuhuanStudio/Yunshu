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
quality. Dense ranking in a separate eval process needs a resident in-process
engine; the dedicated yv websearch smoke proves the actual server fusion path.
Semantic citation support and rubric grading require reviewed references and a
separate judge; mechanical substring validation does not prove claim support.
