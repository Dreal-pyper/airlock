"""Issue #100: the hand-drawn tree in `project_structure.md` must name every module in the package.

`project_structure.md` section 1 draws the `src/airlock` tree by hand, section 3 derives the import
direction from those rows, and `AGENTS.md` sends a contributor here before adding a module. A reader
who plans `from airlock.detectors.native_binary import native_binary` is working from that map, not
from the filesystem, so a row that went missing when a detector landed is a wrong answer to "where
does this belong". Nothing compared the prose with `src/` before this file.

Both directions are gated, because each hides a different defect: a module on disk with no row (the
map lost a file), and a row with no module on disk (the map kept a name after a deletion). Every
failure reports the file path and the line number of the tree it is judging, so the fix is a
one-line edit rather than a hunt.

Two parser defects would make a gate like this quietly useless. Both are designed out and proven
with synthetic blocks in `TestPathReconstruction`:

  * a name pattern that requires a trailing `/` reads directory rows and skips module rows, so every
    top-level module of the package looks absent from the map. `rules.py`, `strkey.py` and friends
    are exactly that shape, and the real block lists them as bare names.
  * rebuilding a path by splitting names without truncating the ancestor stack at the row's own
    indentation lets a directory row and a same-stem module row collide: a tree carrying both
    `line_cap.py` and a `line_cap/` directory yields one wrong path for the other. The package has
    no such pair today, which is why that case is driven from synthetic text: the shape is legal
    markdown in this document, and the reconstruction has to survive it.

Rows whose name is a template (`<name>.py`, `builtin/*.toml`) plan a file rather than name one, and
are skipped: counting them would report a phantom "kept after deletion" gap for every module the
roadmap has not written yet. Generated caches are skipped in the other direction for the same
reason, and `fenced_block` raises instead of returning an empty block, because an empty map
compares clean against an empty directory.

Absences the document itself explains are not failures. `project_structure.md` closes section 1 with
a "Removed on purpose" paragraph naming `jsmask.py`, `wasmparse.py`, `detectors/deploy_gate.py`,
`schema/` and `templates/`. Issue #100 section 5 asks that a documented removal read as an absence
and that the test name the paragraph, so the exemption set is parsed from that line rather than
typed as a literal here; a row the paragraph does not account for still fails.

`TestTheMapAgainstThePackage` is strict and unexempted, and it fails on this checkout:
`src/airlock/report/__init__.py` has never had a row (the `report/` block opens at `console.py`),
and `src/airlock/detectors/native_binary.py` landed with task 9.23 without one. That failing run is
the "before the document edit" evidence the issue asks for; the rows go in the same pull request,
and nothing here papers over the gap. Synthetic text carries the mutation teeth instead, because a
test that deleted a real module to prove it bites would be a test that broke the package.

Standard library and pytest only, parsing text with `re` and `pathlib` and no Markdown dependency.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_DOC = REPO_ROOT / "project_structure.md"

#: The label the failure text uses for the document under test.
DOC_LABEL = "project_structure.md"

#: The package the map is judged against: a repository-relative prefix, and the directory itself.
PKG_REL = "src/airlock"
PKG_ROOT = REPO_ROOT / "src" / "airlock"

#: Section 1's heading: the anchor the fenced block is taken from, so a document restructure that
#: moves the tree fails loudly instead of parsing nothing.
TREE_HEADING = "## 1. Annotated tree"

#: The sentence that opens the deliberate-removal list closing section 1.
REMOVALS_MARKER = "Removed on purpose"

#: Spaces per tree level in this document. Every row of the real block is a multiple of it.
INDENT = 2

#: Generated caches hold no module a reader can import.
IGNORED_DIRS = frozenset({"__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"})

_FENCE = re.compile(r"^\s*```")
_ROW = re.compile(r"^(?P<indent> *)(?P<body>\S.*?)\s*$")
_BACKTICK = re.compile(r"`([^`\n]+)`")


# ---------------------------------------------------------------------------
# Reading the block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One tree row: the path it points at, how deep it sits, and the line it is written on."""

    path: str
    line: int
    depth: int
    is_dir: bool
    comment: str


def line_of(text: str, needle: str) -> int:
    """The one-based line holding `needle`, which must appear exactly once.

    The synthetic cases use this to state an expected line number as a property of the sample text
    rather than as a constant copied from it, so reflowing a sample cannot quietly make an
    assertion mean something else.
    """
    hits = [index for index, raw in enumerate(text.splitlines(), start=1) if needle in raw]
    assert len(hits) == 1, f"{needle!r} appears {len(hits)} times in the sample, need exactly 1"
    return hits[0]


def fenced_block(text: str, heading: str = TREE_HEADING) -> list[tuple[int, str]]:
    """The `(line, raw)` pairs inside the first fence after `heading`.

    Raises rather than returning an empty list: a caller that parsed nothing from a moved heading
    would compare two empty sets and report that the map is perfect.
    """
    lines = text.splitlines()
    anchors = [index for index, raw in enumerate(lines) if raw.strip() == heading]
    if not anchors:
        raise ValueError(f"{heading!r} is not a heading in the document, so the tree has no anchor")
    start = next(
        (index for index in range(anchors[0] + 1, len(lines)) if _FENCE.match(lines[index])), None
    )
    if start is None:
        raise ValueError(f"no fenced block follows {heading!r}")
    stop = next(
        (index for index in range(start + 1, len(lines)) if _FENCE.match(lines[index])), None
    )
    if stop is None:
        raise ValueError(f"the block opened at line {start + 1} is never closed")
    return [(index + 1, lines[index]) for index in range(start + 1, stop)]


def split_name(body: str) -> tuple[str, str]:
    """A row's name and its trailing annotation. No name in this document contains `#`."""
    name, _, comment = body.partition("#")
    return name.strip(), comment.strip()


def is_template(name: str) -> bool:
    """Whether a row plans a file instead of naming one (`<name>.py`, `builtin/*.toml`)."""
    return any(marker in name for marker in ("*", "<", ">"))


def parse_rows(block: Sequence[tuple[int, str]]) -> list[Row]:
    """Rebuild a repository-relative path for every row in the block from its indentation.

    The ancestor stack is truncated at the row's own depth before its name is pushed, which is what
    keeps a module row from inheriting a directory row it merely follows, and is what lets a
    directory and a same-stem module coexist. The first row, when it sits at depth 0 and ends in
    `/`, labels the repository rather than a directory inside it, so its name is left out of every
    path below: that is the difference between `src/airlock/rules.py`, which a reader can paste,
    and `airlock/src/airlock/rules.py`, which they cannot.
    """
    first = _ROW.match(block[0][1]) if block else None
    drop_label = first is not None and first.group("indent") == "" and "/" in first.group("body")
    stack: list[str] = []
    rows: list[Row] = []
    for position, (line, raw) in enumerate(block):
        match = _ROW.match(raw)
        if match is None:
            continue
        depth = len(match.group("indent")) // INDENT
        name, comment = split_name(match.group("body"))
        stack = stack[:depth]
        stack.append(name)
        parts = stack if position == 0 or not drop_label else stack[1:]
        components = [part.rstrip("/") for part in parts]
        rows.append(
            Row(
                path="/".join(components) + ("/" if name.endswith("/") else ""),
                line=line,
                depth=depth,
                is_dir=name.endswith("/"),
                comment=comment,
            )
        )
    return rows


def _under_package(path: str) -> bool:
    """Whether a reconstructed path belongs to the package rather than to `site/` or `scripts/`."""
    return path == PKG_REL or path.startswith(PKG_REL + "/")


def documented_modules(rows: Sequence[Row]) -> dict[str, int]:
    """The concrete `.py` paths the map claims inside the package, as path to tree line."""
    return {
        row.path: row.line
        for row in rows
        if not row.is_dir
        and row.path.endswith(".py")
        and _under_package(row.path)
        and not is_template(PurePosixPath(row.path).name)
    }


def documented_directories(rows: Sequence[Row]) -> dict[str, int]:
    """The directories the map claims inside the package, the package row itself included."""
    return {
        row.path.rstrip("/"): row.line
        for row in rows
        if row.is_dir
        and _under_package(row.path.rstrip("/"))
        and not is_template(PurePosixPath(row.path).name)
    }


def template_rows(rows: Sequence[Row]) -> tuple[Row, ...]:
    """Rows skipped as plans rather than names, kept so a test can prove they were seen at all."""
    return tuple(row for row in rows if is_template(PurePosixPath(row.path).name))


def removals(text: str, doc: str = DOC_LABEL) -> dict[str, int]:
    """The deliberate-removal list closing section 1, as package-relative path to its line.

    Only tokens naming a Python module or a directory are read, which keeps the paragraph's prose
    citations (`ARCHITECTURE.md`, "an `init` command") out of the exemption set.
    """
    for index, raw in enumerate(text.splitlines(), start=1):
        if REMOVALS_MARKER not in raw:
            continue
        found: dict[str, int] = {}
        for token in _BACKTICK.findall(raw):
            if token.endswith(".py") or token.endswith("/"):
                found[f"{PKG_REL}/{token.rstrip('/')}"] = index
        return found
    raise ValueError(f"{doc} has no {REMOVALS_MARKER!r} paragraph, so removals cannot be explained")


# ---------------------------------------------------------------------------
# Reading the repository
# ---------------------------------------------------------------------------


def actual_modules(root: Path = PKG_ROOT, prefix: str = PKG_REL) -> tuple[str, ...]:
    """Every importable module under the package, sorted, as repository-relative paths."""
    return tuple(
        sorted(
            f"{prefix}/{path.relative_to(root).as_posix()}"
            for path in root.rglob("*.py")
            if not IGNORED_DIRS.intersection(path.relative_to(root).parts)
        )
    )


def actual_directories(root: Path = PKG_ROOT, prefix: str = PKG_REL) -> tuple[str, ...]:
    """Every directory under the package, the package itself included, caches excluded."""
    found = {prefix}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_dir() or IGNORED_DIRS.intersection(relative.parts):
            continue
        found.add(f"{prefix}/{relative.as_posix()}")
    return tuple(sorted(found))


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lost:
    """A module on disk the map does not name, plus the place in the map its row belongs."""

    path: str
    block_line: int | None
    last_sibling_line: int | None

    @property
    def parent(self) -> str:
        return str(PurePosixPath(self.path).parent)


@dataclass(frozen=True)
class Stale:
    """A row the map names for a module that is no longer on disk."""

    path: str
    line: int


@dataclass(frozen=True)
class Drift:
    """Both directions of disagreement, plus the absences the document already explains."""

    lost: tuple[Lost, ...]
    stale: tuple[Stale, ...]
    explained: tuple[tuple[str, int], ...]

    @property
    def gaps(self) -> int:
        return len(self.lost) + len(self.stale)

    @property
    def clean(self) -> bool:
        return self.gaps == 0


def anchor_for(
    path: str,
    files: Mapping[str, int],
    directories: Mapping[str, int],
) -> tuple[int | None, int | None]:
    """Where a missing row goes: the header line of its directory block, and that block's last row.

    Both halves are optional because each answers a different question. A directory with no header
    row at all is a worse failure than a header with a missing sibling, so the case is reported
    rather than smoothed over with an invented number.
    """
    parent = str(PurePosixPath(path).parent)
    block_line = directories.get(parent)
    same_block = [
        line for other, line in files.items() if str(PurePosixPath(other).parent) == parent
    ]
    return block_line, (max(same_block) if same_block else None)


def explained_by(path: str, removals_map: Mapping[str, int]) -> int | None:
    """The line of the paragraph that accounts for this absence, or None when nothing does."""
    for token, line in removals_map.items():
        if path == token or path.startswith(token + "/"):
            return line
    return None


def compare(
    documented: Mapping[str, int],
    actual: Iterable[str],
    removals_map: Mapping[str, int],
    directories: Mapping[str, int] | None = None,
) -> Drift:
    """Judge one map against one list of paths in both directions.

    `directories` supplies the block anchors for the lost direction; it is optional so the
    directory-to-directory comparison can leave it out rather than pass its own map twice.
    """
    directory_rows: Mapping[str, int] = directories if directories is not None else {}
    on_disk = set(actual)
    lost: list[Lost] = []
    explained: list[tuple[str, int]] = []
    for path in sorted(on_disk - set(documented)):
        reason = explained_by(path, removals_map)
        if reason is not None:
            explained.append((path, reason))
            continue
        block_line, last_line = anchor_for(path, documented, directory_rows)
        lost.append(Lost(path, block_line, last_line))
    stale: list[Stale] = []
    for path, line in sorted(documented.items(), key=lambda item: (item[1], item[0])):
        if path in on_disk:
            continue
        reason = explained_by(path, removals_map)
        if reason is not None:
            explained.append((path, reason))
            continue
        stale.append(Stale(path, line))
    return Drift(tuple(lost), tuple(stale), tuple(explained))


def compare_package(text: str, root: Path, prefix: str = PKG_REL, doc: str = DOC_LABEL) -> Drift:
    """The whole gate for one document: block, rows, module set, removals, comparison."""
    rows = parse_rows(fenced_block(text))
    return compare(
        documented_modules(rows),
        actual_modules(root, prefix),
        removals(text, doc),
        directories=documented_directories(rows),
    )


def describe(drift: Drift, doc: str = DOC_LABEL, noun: str = "module") -> str:
    """The failure text: the file, the line to edit, and which of the two fixes applies."""
    if drift.clean and not drift.explained:
        return f"every {noun} the map names is on disk, and every {noun} on disk is in the map"
    plural = noun if drift.gaps < 2 else noun + "s"
    lines = [f"{doc} and the package on disk disagree: {drift.gaps} {plural} unaccounted for."]
    for lost_row in drift.lost:
        lines.append(f"  on disk but not in the map: {lost_row.path}")
        if lost_row.block_line is None:
            lines.append(
                f"      {doc} has no row for its directory {lost_row.parent}/ either, so the whole"
                " block is missing"
            )
        elif lost_row.last_sibling_line is None:
            lines.append(
                f"      add a row under {doc}:{lost_row.block_line}, which opens that directory"
            )
        else:
            lines.append(
                f"      the block runs {doc}:{lost_row.block_line}-{lost_row.last_sibling_line},"
                " so the row belongs inside it"
            )
    for stale_row in drift.stale:
        lines.append(f"  in the map but not on disk: {stale_row.path} at {doc}:{stale_row.line}")
    for path, line in drift.explained:
        lines.append(f"  explained absence: {path} is recorded as removed at {doc}:{line}")
    if drift.gaps:
        lines.append(
            f"Either add the row to the block in {doc} section 1, or record the removal in the"
            f" {REMOVALS_MARKER.lower()} paragraph there. A reader who follows the map to plan an"
            " import works from this document, not from the tree."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic documents
# ---------------------------------------------------------------------------

SAMPLE_REMOVALS = (
    "Removed on purpose (do not recreate without a roadmap decision): `jsmask.py`, "
    "`wasmparse.py`, `detectors/deploy_gate.py`, `schema/`. See `ARCHITECTURE.md` 16.4."
)

#: The shape of the real block, small enough to read at a glance and mutated by the cases below.
SAMPLE_ROWS = (
    "  src/airlock/",
    "    __init__.py        # package docstring and the version the report carries",
    "    rules.py           # rule loading, the pattern engine and the line cap",
    "    strkey.py          # seed decode, a bare name with no slash in it",
    "    detectors/",
    "      registry.py      # the plain list of detector functions",
    "      native_binary.py # tracked ELF and Mach-O binaries",
    "    report/",
    "      __init__.py      # reporters package",
    "      console.py       # (2.9)",
)

SAMPLE_MODULES = (
    f"{PKG_REL}/__init__.py",
    f"{PKG_REL}/rules.py",
    f"{PKG_REL}/strkey.py",
    f"{PKG_REL}/detectors/registry.py",
    f"{PKG_REL}/detectors/native_binary.py",
    f"{PKG_REL}/report/__init__.py",
    f"{PKG_REL}/report/console.py",
)


def sample_document(rows: Sequence[str] = SAMPLE_ROWS, tail: str = SAMPLE_REMOVALS) -> str:
    """A document in the shape of `project_structure.md`: heading, fenced tree, removals line."""
    return f"{TREE_HEADING}\n\n```\nairlock/\n" + "\n".join(rows) + f"\n```\n\n{tail}\n"


def without_row(text: str, needle: str) -> str:
    """`text` with the single row containing `needle` deleted, which is how an edit loses one."""
    lines = text.splitlines(keepends=True)
    kept = [raw for raw in lines if needle not in raw]
    assert len(kept) == len(lines) - 1, f"{needle!r} did not match exactly one row"
    return "".join(kept)


def drift_of(text: str, actual: Iterable[str]) -> Drift:
    """Compare one synthetic map against a synthetic list of paths on disk."""
    rows = parse_rows(fenced_block(text))
    return compare(
        documented_modules(rows),
        actual,
        removals(text),
        directories=documented_directories(rows),
    )


class TestTheBlockIsRead:
    """A gate that parses nothing is a gate that passes, so the reader is held honest first."""

    def test_the_tree_block_is_located_by_its_heading(self) -> None:
        block = fenced_block(STRUCTURE_DOC.read_text(encoding="utf-8"))
        assert block, f"{DOC_LABEL} yielded an empty block"
        assert any("src/airlock/" in raw for _, raw in block), (
            f"the block at {DOC_LABEL}:{block[0][0]}-{block[-1][0]} does not draw the package"
        )

    def test_a_tree_without_its_heading_raises_instead_of_passing(self) -> None:
        """The failure mode this file is most likely to have is reading the wrong block."""
        with pytest.raises(ValueError, match="no anchor"):
            fenced_block("# project_structure.md\n\n## 2. Protected paths\n\n```\nairlock/\n```\n")
        with pytest.raises(ValueError, match="never closed"):
            fenced_block(f"{TREE_HEADING}\n\n```\nairlock/\n")

    def test_the_parser_reads_every_indented_row(self) -> None:
        block = fenced_block(STRUCTURE_DOC.read_text(encoding="utf-8"))
        indented = [1 for _, raw in block if _ROW.match(raw)]
        parsed = parse_rows(block)
        assert len(parsed) == len(indented), (
            f"{DOC_LABEL} indents {len(indented)} rows and the parser returned {len(parsed)}, so a"
            " shape the parser does not understand is being dropped silently"
        )

    def test_the_root_label_is_not_a_path_component(self) -> None:
        rows = parse_rows(fenced_block(STRUCTURE_DOC.read_text(encoding="utf-8")))
        assert rows[0].path == "airlock/", "section 1 no longer opens with the repository label"
        assert all(not row.path.startswith("airlock/") for row in rows[1:]), (
            "a documented path carries the repository label, so it is not the import path it looks"
            " like"
        )

    def test_the_removals_paragraph_is_read_from_the_document(self) -> None:
        text = STRUCTURE_DOC.read_text(encoding="utf-8")
        paragraph = line_of(text, REMOVALS_MARKER)
        assert removals(text) == {
            f"{PKG_REL}/jsmask.py": paragraph,
            f"{PKG_REL}/wasmparse.py": paragraph,
            f"{PKG_REL}/detectors/deploy_gate.py": paragraph,
            f"{PKG_REL}/schema": paragraph,
            f"{PKG_REL}/templates": paragraph,
        }, "the exemption set must come from the paragraph, not from a list typed in this file"


class TestPathReconstruction:
    """Criterion 1: paths come from indentation and names, not from a guess about their shape."""

    def test_a_module_row_needs_no_trailing_slash_to_become_a_path(self) -> None:
        text = sample_document()
        modules = documented_modules(parse_rows(fenced_block(text)))
        assert modules[f"{PKG_REL}/rules.py"] == line_of(text, "rules.py")
        assert modules[f"{PKG_REL}/strkey.py"] == line_of(text, "strkey.py")
        assert modules[f"{PKG_REL}/detectors/native_binary.py"] == line_of(text, "native_binary.py")

    def test_the_sample_map_rebuilds_exactly_the_paths_it_claims(self) -> None:
        rows = parse_rows(fenced_block(sample_document()))
        assert sorted(documented_modules(rows)) == sorted(SAMPLE_MODULES)
        assert sorted(documented_directories(rows)) == sorted(
            {PKG_REL, f"{PKG_REL}/detectors", f"{PKG_REL}/report"}
        )

    def test_a_directory_row_does_not_shadow_a_module_of_the_same_stem(self) -> None:
        """`line_cap.py` next to a `line_cap/` directory: both must survive as themselves."""
        text = sample_document(
            (
                "  src/airlock/",
                "    line_cap.py      # the cap, as a module",
                "    line_cap/",
                "      schema.py      # what the cap reads",
                "      walk.py        # what the cap skips",
                "      policy.py      # who sets the cap",
                "    model.py         # the module that follows the directory",
            )
        )
        modules = documented_modules(parse_rows(fenced_block(text)))
        assert sorted(modules) == sorted(
            {
                f"{PKG_REL}/line_cap.py",
                f"{PKG_REL}/line_cap/schema.py",
                f"{PKG_REL}/line_cap/walk.py",
                f"{PKG_REL}/line_cap/policy.py",
                f"{PKG_REL}/model.py",
            }
        ), "a stack truncated at the wrong depth merges the two and hides a real file"

    def test_a_row_name_may_itself_hold_a_slash(self) -> None:
        text = sample_document(
            (
                "  src/airlock/",
                "    report/",
                "      family/base.py   # a shared reporter, named on one row",
            )
        )
        modules = documented_modules(parse_rows(fenced_block(text)))
        assert modules == {f"{PKG_REL}/report/family/base.py": line_of(text, "family/base.py")}

    def test_a_template_row_plans_a_file_and_names_none(self) -> None:
        rows = (
            "  src/airlock/",
            "    detectors/",
            "      registry.py        # the plain list of detector functions",
            "      <name>.py          # one module per family, not written yet",
            "    data/",
            "      builtin/*.toml     # data rules, one file per family",
        )
        parsed = parse_rows(fenced_block(sample_document(rows)))
        assert sorted(documented_modules(parsed)) == [f"{PKG_REL}/detectors/registry.py"]
        assert sorted(documented_directories(parsed)) == [
            PKG_REL,
            f"{PKG_REL}/data",
            f"{PKG_REL}/detectors",
        ]
        assert sorted(row.path for row in template_rows(parsed)) == [
            f"{PKG_REL}/data/builtin/*.toml",
            f"{PKG_REL}/detectors/<name>.py",
        ], "the rows skipped as plans must be the templates, and only the templates"


class TestTheCompare:
    """Criterion 2: an omission and a leftover both fail, and each names the line to fix."""

    def test_a_map_that_matches_the_disk_has_no_drift(self) -> None:
        drift = drift_of(sample_document(), SAMPLE_MODULES)
        assert drift.clean, describe(drift, doc="the sample tree")

    def test_a_file_the_map_drops_fails_with_the_block_its_row_belongs_to(self) -> None:
        text = without_row(sample_document(), "native_binary.py")
        drift = drift_of(text, SAMPLE_MODULES)
        assert len(drift.lost) == 1, describe(drift, doc="the sample tree")
        entry = drift.lost[0]
        assert entry.path == f"{PKG_REL}/detectors/native_binary.py"
        assert entry.block_line == line_of(text, "    detectors/")
        assert entry.last_sibling_line == line_of(text, "registry.py")
        assert drift.stale == ()

    def test_a_name_the_map_keeps_after_deletion_fails_with_its_own_line(self) -> None:
        text = sample_document()
        drift = drift_of(text, tuple(module for module in SAMPLE_MODULES if "strkey" not in module))
        assert len(drift.stale) == 1, describe(drift, doc="the sample tree")
        entry = drift.stale[0]
        assert entry.path == f"{PKG_REL}/strkey.py"
        assert entry.line == line_of(text, "strkey.py")
        assert drift.lost == ()

    def test_the_failure_text_names_the_file_and_the_tree_line(self) -> None:
        text = without_row(sample_document(), "console.py")
        report_line = line_of(text, "    report/")
        init_line = line_of(text, "__init__.py      # reporters package")
        drift = drift_of(text, SAMPLE_MODULES)
        report = describe(drift, doc=DOC_LABEL)
        assert f"{PKG_REL}/report/console.py" in report
        assert f"{DOC_LABEL}:{report_line}-{init_line}" in report, report
        text_of_stale = describe(drift_of(text, ()), doc=DOC_LABEL)
        assert f"{PKG_REL}/report/console.py" not in text_of_stale

    def test_a_removal_the_paragraph_records_is_an_absence_not_a_failure(self) -> None:
        rows = (
            "  src/airlock/",
            "    jsmask.py            # masked before matching, which is not what we do",
            "    wasmparse.py         # folded into the manifest detector",
            "    deployer.py          # gone, and the paragraph says nothing about it",
        )
        text = sample_document(rows)
        drift = drift_of(text, (f"{PKG_REL}/schema/loader.py",))
        paragraph = line_of(text, REMOVALS_MARKER)
        assert sorted(entry.path for entry in drift.stale) == [f"{PKG_REL}/deployer.py"]
        assert drift.lost == (), "a file under a removed directory must not read as a lost row"
        assert sorted(drift.explained) == sorted(
            (
                (f"{PKG_REL}/jsmask.py", paragraph),
                (f"{PKG_REL}/wasmparse.py", paragraph),
                (f"{PKG_REL}/schema/loader.py", paragraph),
            )
        )
        assert f"recorded as removed at {DOC_LABEL}:{paragraph}" in describe(drift, doc=DOC_LABEL)

    def test_a_lost_row_with_no_directory_block_says_the_block_is_missing(self) -> None:
        text = sample_document(("  src/airlock/", "    model.py             # the data classes"))
        drift = drift_of(text, (f"{PKG_REL}/model.py", f"{PKG_REL}/sub/mystery.py"))
        assert len(drift.lost) == 1
        assert drift.lost[0].path == f"{PKG_REL}/sub/mystery.py"
        assert drift.lost[0].block_line is None
        assert "no row for its directory" in describe(drift, doc=DOC_LABEL)


class TestTheMapAgainstThePackage:
    """Criteria 1 and 3 against the real files: strict, and red until the map is fixed."""

    def test_no_module_on_disk_is_missing_from_the_map(self) -> None:
        drift = compare_package(STRUCTURE_DOC.read_text(encoding="utf-8"), PKG_ROOT)
        assert drift.lost == (), describe(drift)

    def test_no_row_names_a_module_that_is_no_longer_on_disk(self) -> None:
        drift = compare_package(STRUCTURE_DOC.read_text(encoding="utf-8"), PKG_ROOT)
        assert drift.stale == (), describe(drift)

    def test_the_documented_paths_are_the_paths_a_reader_can_reach(self) -> None:
        """Every mapped module is a real file, so an import planned from the map resolves."""
        modules = documented_modules(
            parse_rows(fenced_block(STRUCTURE_DOC.read_text(encoding="utf-8")))
        )
        absent = sorted(path for path in modules if not (REPO_ROOT / path).is_file())
        assert absent == [], f"{len(absent)} mapped paths are not files: {absent}"

    def test_no_package_directory_is_missing_from_the_map(self) -> None:
        text = STRUCTURE_DOC.read_text(encoding="utf-8")
        directories = documented_directories(parse_rows(fenced_block(text)))
        drift = compare(directories, actual_directories(), removals(text))
        assert drift.lost == (), describe(drift, noun="directory")

    def test_the_block_and_the_disk_are_both_read(self) -> None:
        """A guard against a green run built on an empty comparison of two empty sets."""
        text = STRUCTURE_DOC.read_text(encoding="utf-8")
        modules = documented_modules(parse_rows(fenced_block(text)))
        assert len(modules) >= 20, f"the parser found only {len(modules)} modules in the map"
        on_disk = actual_modules()
        assert len(on_disk) >= 20, f"only {len(on_disk)} modules under {PKG_ROOT}"
