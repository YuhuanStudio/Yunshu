# Upstream (inspired): ashhart/TensorFold (MIT) src/tensorfold/drafters/dflash_tree.py @ 34bae79a
# Row last-use ordering and state liveness are Yunshu's.
"""Bounded integer plans for a lossless, single-request DFlash tree."""

import mlx.core as mx

"""Bounded device best-first proposal search: expand only nodes actually selected."""
_SEARCH_KERNEL = None


def search_tree(lat, nodes):
    global _SEARCH_KERNEL
    if _SEARCH_KERNEL is None:
        _SEARCH_KERNEL = mx.fast.metal_kernel(
            name="yunshu_best_first_tree",
            input_names=["cands", "unary", "hproj", "succ", "pred", "anchor"],
            output_names=["tokens", "parents"],
            source="""
            uint lane=thread_index_in_simdgroup;
            uint candidate=thread_index_in_threadgroup / 32;
            threadgroup float scores[K];
            threadgroup float heap_score[4*N+4];
            threadgroup int heap_parent[4*N+4],heap_depth[4*N+4],heap_cand[4*N+4];
            threadgroup int count,depth,parent,previous_cand,previous_depth;
            threadgroup float cumulative;
            if(thread_index_in_threadgroup==0){
                count=0;depth=0;parent=0;previous_cand=-1;previous_depth=-1;cumulative=0.0f;
                parents[0]=-1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int node=0;node<N;++node){
                if(depth<D){
                    float dot=0.0f;
                    for(int r=int(lane);r<R;r+=32){
                        float p=previous_depth<0 ? anchor[r] : pred[(previous_depth*K+previous_cand)*R+r];
                        dot+=p*hproj[depth*R+r]*succ[(depth*K+candidate)*R+r];
                    }
                    dot=simd_sum(dot);
                    if(lane==0)scores[candidate]=(unary[depth*K+candidate]+0.6f*dot)/1.5f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(thread_index_in_threadgroup==0){
                    if(depth<D){
                        float top=scores[0];
                        for(int k=1;k<K;++k)top=max(top,scores[k]);
                        float z=0.0f;
                        for(int k=0;k<K;++k)z+=exp(scores[k]-top);
                        float norm=top+log(z);
                        for(int child=0;child<4;++child){
                            int best=0;
                            for(int k=1;k<K;++k)if(scores[k]>scores[best])best=k;
                            heap_score[count]=cumulative+min(scores[best]-norm,-0.0001f);
                            heap_parent[count]=parent;heap_depth[count]=depth;heap_cand[count]=best;
                            ++count;scores[best]=-INFINITY;
                        }
                    }
                    int best=0;
                    for(int j=1;j<count;++j)if(heap_score[j]>heap_score[best])best=j;
                    int d=heap_depth[best],k=heap_cand[best];
                    tokens[node]=cands[d*K+k];parents[node+1]=heap_parent[best];
                    cumulative=heap_score[best];previous_cand=k;previous_depth=d;depth=d+1;parent=node+1;
                    --count;
                    heap_score[best]=heap_score[count];heap_parent[best]=heap_parent[count];
                    heap_depth[best]=heap_depth[count];heap_cand[best]=heap_cand[count];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """,
        )
    d, k = lat.cands.shape
    r = lat.hproj.shape[-1]
    return _SEARCH_KERNEL(
        inputs=[lat.cands, lat.unary, lat.hproj, lat.succ, lat.pred, lat.anchor],
        template=[("D", d), ("K", k), ("R", r), ("N", nodes)],
        grid=(32 * k, 1, 1),
        threadgroup=(32 * k, 1, 1),
        output_shapes=[(nodes,), (nodes + 1,)],
        output_dtypes=[mx.int32, mx.int32],
    )


"""Same ancestry tables as DynamicShape, built in one bounded integer dispatch."""
_PATH_KERNEL = None


class FastShape:
    dynamic = True
    is_chain = False

    def __init__(self, parents, max_depth, *, original_ranks=None, is_chain=False):
        self.original_ranks = original_ranks
        self.is_chain = is_chain
        self._slots = None
        global _PATH_KERNEL
        self.width = int(parents.shape[0])
        self.max_depth = int(max_depth)
        self._parents = parents.astype(mx.int32)
        self.depths = None
        if _PATH_KERNEL is None:
            _PATH_KERNEL = mx.fast.metal_kernel(
                name="yunshu_fast_tree_plan",
                input_names=["parents"],
                output_names=["depths", "paths", "conv"],
                source="""
                uint row=thread_position_in_grid.x;
                if(row>=W)return;
                int depth=0,cur=int(row);
                while(parents[cur]>=0 && depth<M-1){cur=parents[cur];++depth;}
                depths[row]=depth;
                for(int j=0;j<M;++j)paths[row*M+j]=int(row);
                cur=int(row);
                for(int j=depth;j>=0;--j){paths[row*M+j]=cur;cur=max(parents[cur],0);}
                for(int m=3;m>=1;--m){
                    int value=3+depth-m;
                    if(depth>=m){cur=int(row);for(int j=0;j<m;++j)cur=max(parents[cur],0);value=3+cur;}
                    conv[row*3+(3-m)]=value;
                }
                """,
            )
        self._depth, self._paths, self._conv = _PATH_KERNEL(
            inputs=[self._parents],
            template=[("W", self.width), ("M", self.max_depth + 1)],
            grid=(self.width, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (self.width,),
                (self.width, self.max_depth + 1),
                (self.width * 3,),
            ],
            output_dtypes=[mx.int32] * 3,
        )

    def state_slots(self):
        if self._slots is None:
            self._slots = state_slots(self._parents)
        return self._slots

    def gdn_forward(self, layer, state, q, k, v, a, b):
        from .dflash_fast import gdn_forward

        return gdn_forward(self, layer, state, q, k, v, a, b)

    def gdn_prework(self, mixed, conv_prev, layer):
        from .dflash_fast import prework_gather

        return prework_gather(mixed, conv_prev, self.conv_index(), layer)

    def parents_array(self):
        return self._parents

    def depth_array(self):
        return self._depth

    def path_table(self):
        return self._paths

    def conv_index(self):
        return self._conv


_ORDER_KERNEL = None


def live_bound(width):
    # Minimum nodes for rank r>=1: 3*2**(r-1)-1. A leaf needs no
    # persistent slot; a parent with one leaf needs one. Equal-ranked
    # children force another slot only while the parent remains live.
    return max(1, ((width + 1) // 3).bit_length())


def reorder(tokens, parents):
    """Integer-only topological reorder; each ancestor path is unchanged."""

    global _ORDER_KERNEL
    width = int(parents.shape[0])
    if width == 1:
        return tokens, parents, mx.array([0], dtype=mx.int32)
    if _ORDER_KERNEL is None:
        _ORDER_KERNEL = mx.fast.metal_kernel(
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
    return _ORDER_KERNEL(
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


_SLOT_KERNEL = None


def state_slots(parents):
    global _SLOT_KERNEL
    if _SLOT_KERNEL is None:
        _SLOT_KERNEL = mx.fast.metal_kernel(
            name="yunshu_tree_state_slots",
            input_names=["parents"],
            output_names=["reads", "writes"],
            source="""
            if(thread_position_in_grid.x!=0)return;
            int remaining[T], slots[T]; bool used[LIVE];
            for(int i=0;i<T;++i){remaining[i]=0;slots[i]=-1;}
            for(int i=0;i<LIVE;++i)used[i]=false;
            for(int r=1;r<T;++r)++remaining[parents[r]];
            for(int r=0;r<T;++r){
                int par=parents[r];
                reads[r]=par<0?-1:slots[par];
                if(par>=0 && --remaining[par]==0)used[slots[par]]=false;
                int slot=-1;
                if(remaining[r]>0){
                    for(int i=0;i<LIVE;++i)if(!used[i]){slot=i;used[i]=true;break;}
                }
                slots[r]=slot;writes[r]=slot;
            }
            """,
        )
    width = int(parents.shape[0])
    live = max(1, width // 2)
    return _SLOT_KERNEL(
        inputs=[parents],
        template=[("T", width), ("LIVE", live)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(width,)] * 2,
        output_dtypes=[mx.int32] * 2,
    )
