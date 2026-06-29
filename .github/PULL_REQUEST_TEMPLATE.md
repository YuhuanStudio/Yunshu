<!-- Thanks for contributing to Yunshu! Keep PRs focused; coordinate before large
     engine-side changes (python/yunshu_engine/ is actively refactored). -->

## What & why

<!-- What does this change, and why? Link the issue it closes (e.g. "Closes #123"). -->

## Testing

<!-- Paste `just test-unit` output (counts). For hot-path / decode changes, add a
     before/after benchmark. Note anything you could NOT verify. -->

```
# just test-unit
```

## Checklist

- [ ] `just lint` and `just format` are clean (CI runs `ruff check` + `ruff format --check`)
- [ ] `just test-unit` is green (counts pasted above)
- [ ] Title follows [Conventional Commits](https://www.conventionalcommits.org/) (e.g. `fix(gateway): …`)
- [ ] Docs updated if I changed behavior, an endpoint, or a `YUNSHU_*` flag
- [ ] Change fits the project scope (single-node, Apple Silicon; not multi-node / multi-tenant — see README non-goals)
