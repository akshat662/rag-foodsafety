"""Structure-aware chunking for FSSAI regulatory chapters.

Clauses (e.g. 2.1.4, 2.4.12, 2.9.3) are the primary unit of chunking, not
fixed-size character windows. Short clauses are merged forward into their
neighbor; long clauses are split into overlapping parts. Chapter, clause
number, heading, and page metadata travel with every resulting chunk.
"""

import re
from dataclasses import dataclass
from typing import Sequence

# Clause bodies below this size are merged into an adjacent clause.
MIN_CLAUSE_TOKENS = 200
# Clause bodies above this size are split into overlapping parts.
MAX_CLAUSE_TOKENS = 800
# Overlap between adjacent parts of a split clause.
SPLIT_OVERLAP_TOKENS = 100

# Matches a clause heading line such as "2.1.4 Standard for ..." or
# "75[2.9.15 BLACK, WHITE ...". Leading spaces/tabs are tolerated because
# PDF extraction sometimes indents the first clause under a chapter title
# (e.g. " 2.1.1 General Standards..."). The optional "\d+[" prefix is an
# amendment marker the PDF extraction leaves inline; the optional "." or ":"
# after the number covers both "2.4.8." and "2.9.1:" numbering styles.
_CLAUSE_LINE_PATTERN = re.compile(
    r"^[ \t]*(?:\d+\[)?(?P<num>\d+\.\d+\.\d+)[.:]?[ \t]*(?P<rest>.*)$",
    re.MULTILINE,
)

# Where a heading ends and body prose begins on the same line, e.g.
# "DRIED THYME. - (1) Dried thyme is ..." or "MAIDA: ...".
_HEADING_TERMINATOR_PATTERN = re.compile(r"\.\s*-\s|:")

# Token spans (whitespace-delimited) used to split long clauses without
# breaking a number, unit, percentage, or clause reference apart.
_TOKEN_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of chunked text plus the metadata ingest.py needs."""

    text: str
    chapter_id: str
    chapter_title: str
    clause_number: str
    heading: str
    page_start: int | None
    page_end: int | None
    part_index: int
    part_count: int
    token_count: int


@dataclass
class _RawClause:
    """One structurally-detected clause before merge/split adjustments."""

    number: str
    heading: str
    body: str  # exact original text span, including its own heading line
    start_offset: int  # char offset into the joined document text
    end_offset: int


@dataclass
class _ClauseGroup:
    """One or more adjacent raw clauses merged together to reach MIN_CLAUSE_TOKENS."""

    clauses: list[_RawClause]

    @property
    def number_label(self) -> str:
        if len(self.clauses) == 1:
            return self.clauses[0].number
        return f"{self.clauses[0].number}-{self.clauses[-1].number}"

    @property
    def heading(self) -> str:
        return self.clauses[0].heading

    @property
    def text(self) -> str:
        return "\n\n".join(c.body for c in self.clauses)

    @property
    def start_offset(self) -> int:
        return self.clauses[0].start_offset

    @property
    def end_offset(self) -> int:
        return self.clauses[-1].end_offset


def chunk_document(
    pages: Sequence[str],
    chapter_id: str,
    chapter_title: str,
) -> list[Chunk]:
    """Chunk one chapter's extracted text into structure-aware Chunks.

    `pages` is the chapter's text broken out per source page, in order (one
    entry per PyMuPDF page, as produced by scripts/extract_check.py). Page
    numbers are derived from this list's positions rather than by parsing
    page-footer text, which is unreliable across PDF exports. Pass a
    single-element list if page boundaries are unknown; page_start/page_end
    will then always be 1.

    `chapter_id` (e.g. "2.4") is used to reject clause-shaped substrings
    that are actually cross-references into other chapters (e.g.
    "3.1.1.(10)" appearing inside chapter 2.4's body text) or in-body
    amendment references that reuse an earlier clause's number (e.g.
    "2.4.6(24)" inside the running text of a later clause).

    Deterministic: the same `pages`/`chapter_id`/`chapter_title` always
    produce the same chunks. No LLM or embedding calls are made.
    """
    full_text, page_offsets = _join_pages(pages)
    raw_clauses = _find_clauses(full_text, chapter_id)

    if not raw_clauses:
        # No structural markers found: fall back to the whole document as
        # one clause so merge/split logic still applies deterministically.
        raw_clauses = [
            _RawClause(
                number=chapter_id,
                heading=chapter_title,
                body=full_text,
                start_offset=0,
                end_offset=len(full_text),
            )
        ]

    groups = _merge_short_clauses(raw_clauses)

    chunks: list[Chunk] = []
    for group in groups:
        chunks.extend(_split_long_group(group, chapter_id, chapter_title, page_offsets))
    return chunks


def _join_pages(pages: Sequence[str]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join pages with '\\n' and record each page's (page_number, start, end) offsets."""
    parts: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for i, page_text in enumerate(pages):
        start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        offsets.append((i + 1, start, cursor))
        if i < len(pages) - 1:
            parts.append("\n")
            cursor += 1
    return "".join(parts), offsets


def _find_clauses(full_text: str, chapter_id: str) -> list[_RawClause]:
    """Locate clause boundaries in document order, filtering out cross-chapter
    references and non-monotonic in-body amendment references.
    """
    chapter_parts = tuple(int(p) for p in chapter_id.split("."))
    accepted: list[tuple[re.Match, tuple[int, ...]]] = []
    last_tuple = (0,) * 3
    for match in _CLAUSE_LINE_PATTERN.finditer(full_text):
        num_tuple = tuple(int(p) for p in match.group("num").split("."))
        if num_tuple[: len(chapter_parts)] != chapter_parts:
            continue  # belongs to a different chapter (cross-reference)
        if num_tuple <= last_tuple:
            continue  # not a new clause: repeated/backward reference
        accepted.append((match, num_tuple))
        last_tuple = num_tuple

    raw_clauses = []
    for i, (match, _num_tuple) in enumerate(accepted):
        start = match.start()
        end = accepted[i + 1][0].start() if i + 1 < len(accepted) else len(full_text)
        body = full_text[start:end]
        heading = _extract_heading(match.group("rest"), body)
        raw_clauses.append(
            _RawClause(
                number=match.group("num"),
                heading=heading,
                body=body,
                start_offset=start,
                end_offset=end,
            )
        )
    return raw_clauses


def _extract_heading(rest_of_line: str, body: str) -> str:
    """Heading text for a clause: the same-line remainder after the clause
    number, or — if that's blank, as when the number sits alone on its own
    line — the next non-blank line of the clause body.
    """
    candidate = rest_of_line.strip()
    if not candidate:
        for line in body.splitlines()[1:]:
            stripped = line.strip()
            if stripped:
                candidate = stripped
                break
    return _trim_heading(candidate)


def _trim_heading(candidate: str) -> str:
    match = _HEADING_TERMINATOR_PATTERN.search(candidate)
    heading = candidate[: match.start()] if match else candidate
    return heading.strip().rstrip(",.:-").strip()


def _approx_token_count(text: str) -> int:
    """Deterministic, dependency-free token proxy for merge/split thresholds."""
    return len(text.split())


def _merge_short_clauses(raw_clauses: list[_RawClause]) -> list[_ClauseGroup]:
    """Fold clauses under MIN_CLAUSE_TOKENS forward into their neighbor(s).

    A trailing under-sized remainder (no further clause to absorb into) is
    folded backward into the previous finalized group instead.
    """
    groups: list[_ClauseGroup] = []
    current: list[_RawClause] = []
    current_tokens = 0

    for clause in raw_clauses:
        current.append(clause)
        current_tokens += _approx_token_count(clause.body)
        if current_tokens >= MIN_CLAUSE_TOKENS:
            groups.append(_ClauseGroup(current))
            current = []
            current_tokens = 0

    if current:
        if groups:
            groups[-1] = _ClauseGroup(groups[-1].clauses + current)
        else:
            groups.append(_ClauseGroup(current))

    return groups


def _split_long_group(
    group: _ClauseGroup,
    chapter_id: str,
    chapter_title: str,
    page_offsets: list[tuple[int, int, int]],
) -> list[Chunk]:
    """Emit one Chunk for `group`, or several overlapping parts if it exceeds
    MAX_CLAUSE_TOKENS. Splits happen on whitespace-delimited token boundaries
    only, so numbers, units, percentages, and clause references are never
    broken apart.
    """
    page_start, page_end = _pages_for_range(group.start_offset, group.end_offset, page_offsets)
    header = f"{group.number_label} {group.heading}".strip()

    tokens = list(_TOKEN_PATTERN.finditer(group.text))
    total_tokens = len(tokens)

    if total_tokens <= MAX_CLAUSE_TOKENS:
        return [
            Chunk(
                text=f"{header}\n\n{group.text.strip()}",
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                clause_number=group.number_label,
                heading=group.heading,
                page_start=page_start,
                page_end=page_end,
                part_index=1,
                part_count=1,
                token_count=total_tokens,
            )
        ]

    stride = MAX_CLAUSE_TOKENS - SPLIT_OVERLAP_TOKENS
    windows: list[str] = []
    start = 0
    while start < total_tokens:
        end = min(start + MAX_CLAUSE_TOKENS, total_tokens)
        span_start = tokens[start].start()
        span_end = tokens[end - 1].end()
        windows.append(group.text[span_start:span_end])
        if end == total_tokens:
            break
        start += stride

    chunks = []
    for i, window_text in enumerate(windows, start=1):
        part_header = f"{header} (part {i} of {len(windows)})"
        chunks.append(
            Chunk(
                text=f"{part_header}\n\n{window_text.strip()}",
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                clause_number=group.number_label,
                heading=group.heading,
                page_start=page_start,
                page_end=page_end,
                part_index=i,
                part_count=len(windows),
                token_count=len(window_text.split()),
            )
        )
    return chunks


def _pages_for_range(
    start_offset: int, end_offset: int, page_offsets: list[tuple[int, int, int]]
) -> tuple[int | None, int | None]:
    """Page numbers spanned by [start_offset, end_offset) in the joined document text.

    Split parts of a clause report their whole parent clause's page range
    rather than a per-word page lookup — a safe superset, kept simple.
    """
    overlapping = [
        page_number
        for page_number, page_start, page_end in page_offsets
        if page_start < end_offset and page_end > start_offset
    ]
    if not overlapping:
        return None, None
    return min(overlapping), max(overlapping)
