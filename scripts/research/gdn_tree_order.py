"""Own idea: last-use traversal reduces tree GDN state liveness."""


def order_plan(parents):
    children = [[] for _ in parents]
    for row, parent in enumerate(parents[1:], 1):
        children[parent].append(row)
    need = [0] * len(parents)
    for row in reversed(range(len(parents))):
        kids = sorted(children[row], key=lambda c: (need[c], c))
        children[row] = kids
        need[row] = max(
            [
                1 if kids else 0,
                *[need[c] + (i + 1 < len(kids)) for i, c in enumerate(kids)],
            ]
        )
    order = []

    def visit(row):
        order.append(row)
        for child in children[row]:
            visit(child)

    visit(0)
    index = {old: new for new, old in enumerate(order)}
    return order, [-1, *[index[parents[old]] for old in order[1:]]], need[0]


_KERNEL = None


def live_bound(width):
    # Minimum nodes for rank r>=1: 3*2**(r-1)-1. A leaf needs no
    # persistent slot; a parent with one leaf needs one. Equal-ranked
    # children force another slot only while the parent remains live.
    return max(1, ((width + 1) // 3).bit_length())


def reorder(tokens, parents):
    """Integer-only topological reorder; each ancestor path is unchanged."""
    import mlx.core as mx

    global _KERNEL
    width = int(parents.shape[0])
    if width == 1:
        return tokens, parents, mx.array([0], dtype=mx.int32)
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="yunshu_gdn_last_use_order",
            input_names=["tokens", "parents"],
            output_names=["new_tokens", "new_parents", "permutation"],
            source="""
            if (thread_position_in_grid.x != 0) return;
            int need[W], stack[W], mapping[W];
            bool pushed[W];
            for (int r=W-1; r>=0; --r) {
                mapping[r]=-1; pushed[r]=false;
                int last=-1;
                for (int c=r+1; c<W; ++c) if (parents[c]==r) {
                    if (last<0 || need[c]>=need[last]) last=c;
                }
                int rank=last<0 ? 0 : max(1,need[last]);
                for (int c=r+1; c<W; ++c)
                    if (parents[c]==r && c!=last) rank=max(rank,1+need[c]);
                need[r]=rank;
            }
            int count=1; stack[0]=0;
            for (int row=0; row<W; ++row) {
                int old=stack[--count]; mapping[old]=row;
                permutation[row]=old;
                new_parents[row]=old==0 ? -1 : mapping[parents[old]];
                if (row>0) new_tokens[row-1]=tokens[old-1];
                // Push largest first: the smallest subtree is visited first,
                // and the largest last can reuse its parent's freed slot.
                while (true) {
                    int best=-1;
                    for (int c=old+1; c<W; ++c) if (parents[c]==old && !pushed[c])
                        if (best<0 || need[c]>=need[best]) best=c;
                    if (best<0) break;
                    pushed[best]=true; stack[count++]=best;
                }
            }
            """,
        )
    return _KERNEL(
        inputs=[tokens, parents],
        template=[("W", width)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(width - 1,), (width,), (width,)],
        output_dtypes=[mx.int32] * 3,
    )


def remap_landed(landed_nodes, permutation):
    """Keep budget probabilities in original proposal rank after row reordering."""
    return [int(permutation[node + 1]) - 1 for node in landed_nodes]
