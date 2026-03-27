"""
Tokenizer Translation Table — Reusable Alignment Module

Produces a many-to-many token alignment between any two tokenizers
for a given input text, using character-span overlap.
"""

from dataclasses import dataclass, field


@dataclass
class AlignmentLink:
    model_a_indices: list[int]
    model_b_indices: list[int]
    shared_text: str
    alignment_type: str  # "one_to_one", "one_to_many", "many_to_one", "many_to_many"


@dataclass
class ActivationPair:
    model_a_positions: list[int]
    model_b_positions: list[int]
    text: str
    bucket: int  # 1, 2, or 3


@dataclass
class TokenAlignment:
    input_text: str
    model_a_name: str
    model_b_name: str
    model_a_tokens: list[str]
    model_b_tokens: list[str]
    model_a_char_spans: list[tuple[int, int]]
    model_b_char_spans: list[tuple[int, int]]
    alignment: list[AlignmentLink] = field(default_factory=list)


def _compute_offsets_fallback(tokenizer, text: str) -> tuple[list[int], list[str], list[tuple[int, int]]]:
    """Fallback offset computation for slow tokenizers that lack return_offsets_mapping."""
    encoding = tokenizer(text, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    tok_strs = tokenizer.convert_ids_to_tokens(input_ids)

    spans = []
    search_start = 0
    for tid, tok_str in zip(input_ids, tok_strs):
        # Decode single token to get its text representation
        decoded = tokenizer.decode([tid])
        # Try to find this decoded text in the original string
        idx = text.find(decoded, search_start)
        if idx == -1:
            # Try stripping common prefixes
            cleaned = tok_str
            for prefix in ("Ġ", "▁", "##", "Ã", " "):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
            idx = text.find(cleaned, search_start)
        if idx == -1:
            # Last resort: assign from search_start with length of decoded
            idx = search_start
        end = idx + len(decoded) if idx >= search_start else search_start + max(len(decoded), 1)
        end = min(end, len(text))
        if idx > len(text):
            idx = len(text)
        spans.append((idx, end))
        search_start = end

    return input_ids, tok_strs, spans


def _get_char_spans(tokenizer, text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Tokenize text and return (token_strings, char_spans) with special tokens stripped."""
    try:
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding["offset_mapping"]
        input_ids = encoding["input_ids"]

        tokens = []
        spans = []
        for i, (token_id, (start, end)) in enumerate(zip(input_ids, offsets)):
            if start == end:
                continue
            tok_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            tokens.append(tok_str)
            spans.append((start, end))

        return tokens, spans
    except Exception:
        # Fallback for slow tokenizers
        input_ids, tok_strs, spans = _compute_offsets_fallback(tokenizer, text)
        tokens = []
        valid_spans = []
        for tok_str, (s, e) in zip(tok_strs, spans):
            if s == e:
                continue
            tokens.append(tok_str)
            valid_spans.append((s, e))
        return tokens, valid_spans


def _build_alignment_units(
    spans_a: list[tuple[int, int]],
    spans_b: list[tuple[int, int]],
    text: str,
) -> list[AlignmentLink]:
    """
    Group tokens into alignment units using the atomic-segment merging algorithm.

    1. Collect all span boundaries from both models.
    2. Define atomic character segments from consecutive boundary pairs.
    3. For each atomic segment, find which tokens from A and B cover it.
    4. Merge adjacent segments with identical token coverage.
    5. Classify each merged group.
    """
    # Collect all unique boundaries
    boundaries = set()
    for s, e in spans_a:
        boundaries.add(s)
        boundaries.add(e)
    for s, e in spans_b:
        boundaries.add(s)
        boundaries.add(e)
    boundaries = sorted(boundaries)

    if len(boundaries) < 2:
        return []

    # Build index: for each atomic segment, which tokens cover it
    atomic_segments = []
    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]

        a_indices = frozenset(
            idx for idx, (s, e) in enumerate(spans_a)
            if s <= seg_start and seg_end <= e
        )
        b_indices = frozenset(
            idx for idx, (s, e) in enumerate(spans_b)
            if s <= seg_start and seg_end <= e
        )

        atomic_segments.append((seg_start, seg_end, a_indices, b_indices))

    # Merge adjacent segments with same coverage
    merged = []
    if atomic_segments:
        cur_start, _, cur_a, cur_b = atomic_segments[0]
        cur_end = atomic_segments[0][1]
        for seg_start, seg_end, a_idx, b_idx in atomic_segments[1:]:
            if a_idx == cur_a and b_idx == cur_b:
                cur_end = seg_end
            else:
                merged.append((cur_start, cur_end, cur_a, cur_b))
                cur_start, cur_end, cur_a, cur_b = seg_start, seg_end, a_idx, b_idx
        merged.append((cur_start, cur_end, cur_a, cur_b))

    # Convert to AlignmentLinks
    links = []
    for seg_start, seg_end, a_indices, b_indices in merged:
        a_list = sorted(a_indices)
        b_list = sorted(b_indices)

        if not a_list and not b_list:
            continue

        na = len(a_list)
        nb = len(b_list)
        if na == 1 and nb == 1:
            atype = "one_to_one"
        elif na == 1 and nb > 1:
            atype = "one_to_many"
        elif na > 1 and nb == 1:
            atype = "many_to_one"
        else:
            atype = "many_to_many"

        links.append(AlignmentLink(
            model_a_indices=a_list,
            model_b_indices=b_list,
            shared_text=text[seg_start:seg_end],
            alignment_type=atype,
        ))

    return links


def align_tokens(text: str, tokenizer_a, tokenizer_b,
                 name_a: str = "model_a", name_b: str = "model_b") -> TokenAlignment:
    """Produce the full alignment between two tokenizers for a given text."""
    tokens_a, spans_a = _get_char_spans(tokenizer_a, text)
    tokens_b, spans_b = _get_char_spans(tokenizer_b, text)

    links = _build_alignment_units(spans_a, spans_b, text)

    return TokenAlignment(
        input_text=text,
        model_a_name=name_a,
        model_b_name=name_b,
        model_a_tokens=tokens_a,
        model_b_tokens=tokens_b,
        model_a_char_spans=spans_a,
        model_b_char_spans=spans_b,
        alignment=links,
    )


def _classify_link(link: AlignmentLink) -> int:
    """Classify an alignment link into bucket 1, 2, or 3."""
    na = len(link.model_a_indices)
    nb = len(link.model_b_indices)

    if na == 1 and nb == 1:
        return 1  # Exact match
    total = na + nb
    if total <= 3:
        # one_to_two or two_to_one
        return 2  # Minor split
    return 3  # Major divergence


def classify_alignment(alignment: TokenAlignment) -> dict:
    """Return bucket distribution: percentage of alignment units in each bucket."""
    counts = {1: 0, 2: 0, 3: 0}
    for link in alignment.alignment:
        bucket = _classify_link(link)
        counts[bucket] += 1

    total = sum(counts.values())
    if total == 0:
        return {"bucket_1_pct": 0.0, "bucket_2_pct": 0.0, "bucket_3_pct": 0.0,
                "bucket_1_count": 0, "bucket_2_count": 0, "bucket_3_count": 0, "total": 0}

    return {
        "bucket_1_pct": counts[1] / total,
        "bucket_2_pct": counts[2] / total,
        "bucket_3_pct": counts[3] / total,
        "bucket_1_count": counts[1],
        "bucket_2_count": counts[2],
        "bucket_3_count": counts[3],
        "total": total,
    }


def get_activation_pairs(alignment: TokenAlignment) -> list[ActivationPair]:
    """
    From an alignment, produce a list of (model_a_positions, model_b_positions) pairs
    that can be used to extract and compare activations.

    For bucket 1: single position from each model.
    For bucket 2-3: list of positions from each model (to be pooled/averaged by downstream code).
    """
    pairs = []
    for link in alignment.alignment:
        bucket = _classify_link(link)
        pairs.append(ActivationPair(
            model_a_positions=link.model_a_indices,
            model_b_positions=link.model_b_indices,
            text=link.shared_text,
            bucket=bucket,
        ))
    return pairs
