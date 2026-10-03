"""Research-only native singleton forward views that retain KV capacity.

Replicate qwen3_5's singleton extract/merge metadata while leaving custom,
quantized, padded, multi-row, and speculative state on upstream's path.
"""


def extract_rows(caches):
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

    rows = []
    for source in caches:
        if type(source) is BatchKVCache:
            if (
                source.left_padding.size != 1
                or int(source.left_padding.item()) != 0
                or source._right_padding is not None
                or (source.keys is None) != (source.values is None)
            ):
                return None
            row = KVCache()
            if source.keys is not None:
                if any(
                    a.ndim != 4 or a.shape[0] != 1 or a.shape[2] < source._idx
                    for a in (source.keys, source.values)
                ):
                    return None
                row.keys = source.keys.view(source.keys.dtype)
                row.values = source.values.view(source.values.dtype)
                row.offset = source._idx
        elif type(source) is ArraysCache:
            if getattr(source, "_speculation", None) is not None or any(
                a is not None and (not a.ndim or a.shape[0] != 1) for a in source.cache
            ):
                return None
            row = ArraysCache(len(source.cache))
            row.cache = [
                a.view(a.dtype) if a is not None else None for a in source.cache
            ]
            if source.lengths is not None:
                row.lengths = source.lengths[:1]
        else:
            return None
        rows.append(row)
    return rows


def merge_rows(rows):
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

    from yunshu_engine.apc_manager import _single_native_arrays_row

    merged = []
    for row in rows:
        if type(row) is KVCache:
            batch = BatchKVCache([0])
            if row.offset:
                batch.keys = row.keys.view(row.keys.dtype)
                batch.values = row.values.view(row.values.dtype)
                batch._idx = row.offset
                batch.offset += row.offset
            merged.append(batch)
        elif type(row) is ArraysCache:
            merged.append(_single_native_arrays_row(row))
        else:
            raise TypeError("singleton forward replaced a native cache contract")
    return merged


def wrap(original):
    def forward(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        position_ids=None,
        capture_layer_ids=None,
        hidden_sink=None,
    ):
        from mlx_vlm.models.cache import BatchKVCache

        batch = (inputs_embeds if inputs_embeds is not None else inputs).shape[0]
        rows = None
        if (
            batch == 1
            and hidden_sink is None
            and cache
            and type(cache[self.fa_idx]) is BatchKVCache
        ):
            rows = extract_rows(cache)
        if rows is None:
            return original(
                self,
                inputs,
                inputs_embeds=inputs_embeds,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                capture_layer_ids=capture_layer_ids,
                hidden_sink=hidden_sink,
            )
        # Upstream's singleton shortcut drops mask/capture arguments here too.
        result = original(
            self,
            inputs,
            inputs_embeds=inputs_embeds,
            cache=rows,
            position_ids=position_ids,
        )
        replacements = merge_rows(rows)
        cache[:] = replacements
        return result

    return forward
