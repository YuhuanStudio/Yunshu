"""Static draft-tree topologies.

A topology is a prefix-closed set of *rank paths*: node ``(0, 1)`` is the
second-ranked child of the top-ranked child of the pending token. Drafters fill
the tokens on the GPU (rank ``r`` of a parent's candidates), so the tree's
shape is known on the host before any draft is computed: the verify graph
(``tree_verify``) can be built while the drafter is still running, and the
round needs one host sync (the target's tokens) instead of one per draft level.

Nodes are ordered most-valuable first (parents before children), so any prefix
of ``nodes`` is itself a valid, smaller topology (used for short budgets and for
cost-aware trimming).
"""

from __future__ import annotations

from . import tree_verify as tv


class Topology:
    def __init__(self, paths):
        paths = [tuple(int(r) for r in p) for p in paths]
        seen = set()
        for p in paths:
            if len(p) > 1 and p[:-1] not in seen:
                raise ValueError(f"topology not prefix-closed / ordered: {p}")
            seen.add(p)
        self.paths = paths
        index = {p: i for i, p in enumerate(paths)}
        self.size = len(paths)
        self.parents = [index[p[:-1]] if len(p) > 1 else -1 for p in paths]
        self.ranks = [p[-1] for p in paths]
        self.depths = [len(p) for p in paths]
        self.max_depth = max(self.depths, default=0)
        # window rows: the pending token, then the nodes in this order
        self.window_parents = [-1] + [0 if q < 0 else q + 1 for q in self.parents]
        self._shape = None
        self._levels = None
        self._prefixes: dict = {}

    def shape(self) -> tv.TreeShape:
        if self._shape is None:
            self._shape = tv.TreeShape(self.window_parents)
        return self._shape

    def levels(self):
        """Per depth: (node indices in this topology's order, each node's
        position within the previous level's list, each node's rank)."""
        if self._levels is None:
            out = []
            prev_pos: dict[int, int] = {}
            for d in range(1, self.max_depth + 1):
                nodes = [i for i in range(self.size) if self.depths[i] == d]
                parent_pos = [prev_pos[self.parents[i]] if d > 1 else 0 for i in nodes]
                out.append((nodes, parent_pos, [self.ranks[i] for i in nodes]))
                prev_pos = {i: k for k, i in enumerate(nodes)}
            self._levels = out
        return self._levels

    def prefix(self, n: int) -> Topology:
        """The first ``n`` nodes as a topology of their own."""
        n = max(0, min(int(n), self.size))
        if n == self.size:
            return self
        top = self._prefixes.get(n)
        if top is None:
            top = self._prefixes[n] = Topology(self.paths[:n])
        return top

    def permutation(self):
        """Index of each node (in this order) within the level-concatenated list."""
        order = [i for nodes, _, _ in self.levels() for i in nodes]
        inv = [0] * self.size
        for at, node in enumerate(order):
            inv[node] = at
        return inv


__all__ = ["Topology"]
