"""The Yunshu console process: web console, docs, a reverse proxy to the engine, and the history.

Deliberately light. Nothing in this package imports MLX, mlx-lm, mlx-vlm or any model code (a unit
test imports it and asserts that no ``mlx*`` module was loaded), so it starts instantly, idles at a
few tens of MiB, and cannot be hurt by an engine fault. It reads ``yunshu_engine.settings`` and
``yunshu_engine.paths`` (plain modules) and the API-key store, nothing else from the engine.
"""
