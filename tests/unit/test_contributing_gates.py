"""Issue #99: the contributor checklist and the CI gates must name the same commands.

`CONTRIBUTING.md` is where a human is told what to run, `AGENTS.md` 5 is where an automated
contributor is told the same thing, and `SUPPORT.md` is the "commands that decide whether a pull
request is reviewable" page. Three documents, three lists, and the workflows that actually judge
the pull request: when they disagree, a change that passes every documented command still fails CI
on a command no document named, which is exactly what issue #99 reported.

PyYAML is not a dependency of this repository and adding one to read three workflow files would be
the wrong trade (`AGENTS.md` 4), so this is a line scanner over the text in the shape of
`tests/unit/test_workflow_matrix.py`. It reads inline `run:` scalars, block scalars (`run: |`) and
quoted ones, carries the enclosing `name:` so a failure can quote the step, and reads `uses:` as an
action rather than as a command. A scanner that silently reads nothing is a gate that never fires,
so `test_no_run_line_is_left_unread` counts the `run:` lines in the raw text against the steps it
parsed, and every document test asserts it read commands before it judges them.

Commands are compared after `normalise()`, which is deliberately narrow: it drops the interpreter
that only CI's job needs (`python -m airlock` and `airlock` are the same check, and so are
`python scripts/x.py` and `sh scripts/x.sh` versus the bare path), drops an argument CI
interpolates from `${{ }}` or `${}` because a document cannot name a branch that does not exist
yet, and drops a trailing `.` that means "this directory". Flags stay significant on both sides, so
`ruff check --fix` is not `ruff check .`, and the CI-to-document direction only ever accepts a
command the document actually wrote down, or one it wrote down with an extra positional argument.

The gate is exact in both directions, which is what makes it worth running: a command CI runs that
the document does not name fails, and a command the document names that no gated step runs fails
too. There is no ledger of accepted gaps here, because the documents are the ones this repository
asks a contributor to trust.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

#: The three workflows that judge a pull request or publish what a pull request produced.
#: `pr-mode.yml` and `scorecard.yml` are deliberately absent: the first runs a pytest subset of a
#: command already in the lists, and the second has no `run:` scalar at all.
GATED_WORKFLOWS = ("ci.yml", "roadmap-check.yml", "pages.yml")

#: The command the self-scan step runs, in the spelling `ci.yml` uses. Its own test pins the
#: claim that this is the same command as `python -m airlock scan --mode full` in `AGENTS.md` 5.
SELF_SCAN_CI = "airlock scan --mode full ."
SELF_SCAN_MODULE = "python -m airlock scan --mode full"

#: An interpreter or a shell that CI uses to address a program, and that a document may or may not
#: write. `python -m pytest` and `pytest` are the same check; `python3` and `sh` are the same habit.
_INTERPRETERS = ("python", "python3", "python3.11", "sh", "bash", "zsh")

#: Programs that are commands in their own right, with no interpreter in front of them.
_HEADS = ("ruff", "mypy", "pytest", "pip", "airlock", "npm", "npx", "uv")

#: Where this repository's own gate scripts live. A path outside it is documentation, not a command:
#: the protected paths listed under `CONTRIBUTING.md` Boundaries include `src/airlock/policy.py`,
#: which is a file to edit and never a command to run.
_SCRIPT_DIR = "scripts/"
_SCRIPT_SUFFIXES = (".py", ".sh")

#: Programs for which a trailing `.` is the thing being installed rather than the directory being
#: checked, so it survives normalisation.
_INSTALLERS = ("pip", "npm", "npx", "uv")

_RUN = re.compile(r"^run:\s*(?P<value>.*)$")
_USES = re.compile(r"^uses:\s*(?P<value>[^\s#]+)")
_NAME = re.compile(r"^name:\s*(?P<value>.+?)\s*$")
_BLOCK = re.compile(r"^[|>][-+]?[0-9]*[-+]?$")
_FENCE = re.compile(r"^\s*```")
_COMMENT = re.compile(r"\s+#\s.*$")
_SPAN = re.compile(r"`([^`\n]+)`")


@dataclass(frozen=True)
class Step:
    """One workflow step that either runs a command or references an action."""

    workflow: str
    line: int
    name: str
    command: str = ""
    action: str = ""

    @property
    def norm(self) -> str:
        return normalise(self.command)

    @property
    def where(self) -> str:
        return f"{self.workflow}:{self.line}"


@dataclass(frozen=True)
class Exemption:
    """A CI step a contributor is not asked to reproduce, with the one reason that is true."""

    pattern: str
    reason: str


#: The only exemptions the gate accepts, one reason each. `pattern` is matched as leading tokens of
#: a normalised `run:` command, or as the leading name of a `uses:` action.
EXEMPTIONS: tuple[Exemption, ...] = (
    Exemption(
        "pip install",
        "it makes the documented commands available rather than checking anything, so a "
        "contributor cannot run it before having run them",
    ),
    Exemption(
        "scripts/build_site.py",
        "the Pages build regenerates `site/` inside the runner to publish it; the generator a "
        "contributor runs is the one `CONTRIBUTING.md` 7 already names",
    ),
    Exemption(
        "actions/deploy-pages",
        "a deploy action is not a command: it publishes an artefact the runner already built and "
        "has no local equivalent",
    ),
)


@dataclass(frozen=True)
class Documented:
    """One command a document names, with the line and the raw span it came from."""

    document: str
    line: int
    span: str
    command: str


@dataclass(frozen=True)
class Document:
    """A contributor-facing document and the reader that extracts its commands."""

    path: str
    extract: str


#: `spans` reads backticked inline code, which is how `CONTRIBUTING.md` writes commands. `fenced`
#: reads whole lines inside a ``` block, which is how `AGENTS.md` 5 and `SUPPORT.md` write them.
#: Both readers strip a trailing `# comment`, and both ignore anything that does not start like an
#: invocation, so prose paths and file names are never mistaken for commands.
DOCUMENTS: tuple[Document, ...] = (
    Document("CONTRIBUTING.md", "spans"),
    Document("AGENTS.md", "fenced"),
    Document("SUPPORT.md", "fenced"),
)


@dataclass(frozen=True)
class Undocumented:
    """A command CI runs that the document under comparison never names."""

    command: str
    document: str
    workflow: str
    line: int
    step: str


@dataclass(frozen=True)
class Unrun:
    """A command a document names that no gated workflow step runs."""

    command: str
    document: str
    line: int
    span: str


@dataclass(frozen=True)
class Gate:
    """The two directions of disagreement between a document and the workflows."""

    missing: tuple[Undocumented, ...] = ()
    unrun: tuple[Unrun, ...] = ()
    exempt: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.missing and not self.unrun


def is_invocation(text: str) -> bool:
    """Whether a span of documentation is an instruction to run something.

    A leading interpreter, a leading program name, or a path under `scripts/` ending in a suffix
    this repository actually gates on. Anything else, including `src/airlock/policy.py`, is prose.
    """
    tokens = text.split()
    if not tokens:
        return False
    first = tokens[0]
    if first in _INTERPRETERS or first in _HEADS:
        return True
    return first.startswith(_SCRIPT_DIR) and first.endswith(_SCRIPT_SUFFIXES)


def _is_script_path(token: str) -> bool:
    return token.startswith(_SCRIPT_DIR) or token.endswith(_SCRIPT_SUFFIXES)


def normalise(command: str) -> str:
    """The command as a check rather than as a shell line.

    Four removals, each justified by what a document can honestly be expected to write:
    an interpreter that only addresses the program (`python -m airlock`, `python scripts/x.py`,
    `sh scripts/x.sh`); an argument CI interpolates (`"origin/${BASE_REF}"`); a trailing `.` that
    means the current directory, except for an installer, where `.` is the package being installed;
    and the quotes around either.
    """
    tokens = command.split()
    if not tokens:
        return ""
    if tokens[0] in _INTERPRETERS:
        if len(tokens) >= 3 and tokens[1] == "-m":
            tokens = tokens[2:]
        elif len(tokens) >= 2 and _is_script_path(tokens[1]):
            tokens = tokens[1:]
    tokens = [token for token in tokens if "${" not in token and "{{" not in token]
    if len(tokens) > 1 and tokens[-1] in (".", "./") and tokens[0] not in _INSTALLERS:
        tokens = tokens[:-1]
    return " ".join(token.strip("\"'") for token in tokens)


def strip_comment(text: str) -> str:
    """A trailing `# ...` on a command line, which markdown fences and `run:` scalars both use."""
    return _COMMENT.sub("", text).strip()


def _unquote(text: str) -> str:
    return text.strip().strip("\"'") if text.strip()[:1] in {'"', "'"} else text.strip()


def _block_scalar(lines: Sequence[str], index: int) -> tuple[list[str], int]:
    """The body of a `run: |` block, and the first line index after it."""
    base = len(lines[index]) - len(lines[index].lstrip())
    collected: list[str] = []
    cursor = index + 1
    while cursor < len(lines):
        line = lines[cursor]
        if not line.strip():
            cursor += 1
            continue
        if len(line) - len(line.lstrip()) <= base:
            break
        collected.append(line.strip())
        cursor += 1
    return collected, cursor


def parse_steps(text: str, workflow: str) -> list[Step]:
    """Every `run:` and `uses:` step in one workflow, with the line the key sits on.

    A block scalar is one command reported at its `run:` line, because that is the line a reader
    greps for and the line the step is declared on.
    """
    lines = text.splitlines()
    steps: list[Step] = []
    pending = ""
    index = 0
    while index < len(lines):
        body = lines[index].strip()
        if body.startswith("-"):
            pending = ""
            body = body[1:].strip()
        named = _NAME.match(body)
        if named is not None:
            pending = _unquote(named.group("value"))
        run = _RUN.match(body)
        if run is not None:
            value = run.group("value").strip()
            line = index + 1
            if _BLOCK.match(value):
                collected, index = _block_scalar(lines, index)
                command = " ".join(strip_comment(item) for item in collected if item)
            else:
                command = strip_comment(_unquote(value))
                index += 1
            steps.append(Step(workflow, line, pending, command))
            continue
        uses = _USES.match(body)
        if uses is not None:
            steps.append(Step(workflow, index + 1, pending, "", _unquote(uses.group("value"))))
        index += 1
    return steps


def run_line_count(text: str) -> int:
    """How many lines of a workflow declare a `run:` scalar, however the value is written."""
    return sum(1 for line in text.splitlines() if line.strip().lstrip("- ").startswith("run:"))


def workflow_steps(path: Path) -> list[Step]:
    text = path.read_text(encoding="utf-8")
    return parse_steps(text, display(path))


def display(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def document_commands(document: Document) -> list[Documented]:
    """The commands one contributor-facing document names, in the order it names them."""
    text = (REPO_ROOT / document.path).read_text(encoding="utf-8")
    found: list[Documented] = []
    inside = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            inside = not inside
            continue
        if document.extract == "fenced":
            if not inside:
                continue
            candidates = [strip_comment(line.strip())]
        else:
            candidates = [span for span in _SPAN.findall(line)]
        for span in candidates:
            if is_invocation(span):
                found.append(Documented(document.path, number, span, normalise(span)))
    return found


def exemption_for(command: str, exemptions: Sequence[Exemption]) -> Exemption | None:
    """The exemption whose pattern leads this normalised command, if one does."""
    tokens = command.split()
    for exemption in exemptions:
        pattern = exemption.pattern.split()
        if tokens[: len(pattern)] == pattern:
            return exemption
    return None


def _exemption_for_step(step: Step, exemptions: Sequence[Exemption]) -> Exemption | None:
    if step.action:
        name = step.action.split("@")[0]
        return next(
            (e for e in exemptions if name == e.pattern or name.startswith(e.pattern + "/")), None
        )
    return exemption_for(step.norm, exemptions)


def run_by_some_step(command: str, norms: Sequence[str]) -> bool:
    """Whether a documented command is the check a step runs, allowanced for detail.

    Equality, or either side leading the other token by token: a document may write
    `python scripts/build_site.py` where CI writes `--check`, or name a base ref CI takes from an
    environment variable. A different flag set is not a prefix and still fails.
    """
    for norm in norms:
        if norm == command:
            return True
        if norm.startswith(command + " ") or command.startswith(norm + " "):
            return True
    return False


def documented_covers(command: str, named: set[str]) -> bool:
    """Whether a document names this CI command closely enough to have told a contributor to run it.

    Equality, or the command with extra positional arguments: `SUPPORT.md` writes
    `scripts/check_roadmap_touched.sh origin/main` for a step that takes its base ref from an
    environment variable. A flag is not an argument a reader may invent, so `mypy src --strict`
    does not cover `mypy src` and the strict direction keeps its teeth.
    """
    if command in named:
        return True
    for name in named:
        if name.startswith(command + " ") and not any(
            token.startswith("-") for token in name[len(command) :].split()
        ):
            return True
    return False


def compare(
    steps: Sequence[Step],
    document: str,
    documented: Sequence[Documented],
    exemptions: Sequence[Exemption],
) -> Gate:
    """Both directions of disagreement between a set of workflow steps and one document."""
    run_steps = [step for step in steps if step.command]
    norms = [step.norm for step in run_steps]
    named = {item.command for item in documented}
    missing: list[Undocumented] = []
    exempt: list[str] = []
    for step in run_steps:
        if documented_covers(step.norm, named):
            continue
        allowance = exemption_for(step.norm, exemptions)
        if allowance is not None:
            exempt.append(step.norm)
            continue
        missing.append(
            Undocumented(step.norm, document, step.workflow, step.line, step.name or "unnamed step")
        )
    unrun: list[Unrun] = []
    for item in documented:
        if exemption_for(item.command, exemptions) is not None:
            continue
        if not run_by_some_step(item.command, norms):
            unrun.append(Unrun(item.command, item.document, item.line, item.span))
    return Gate(tuple(missing), tuple(unrun), tuple(exempt))


def describe_missing(entries: Sequence[Undocumented]) -> str:
    """The failure text for an undocumented CI command: the command, the file, the line, the fix."""
    if not entries:
        return "every command the gated workflows run is named by the document"
    lines = [
        f"{entries[0].document} does not name {len(entries)} command"
        f"{'s' if len(entries) > 1 else ''} that a gated workflow runs:",
    ]
    for entry in entries:
        lines.append(f"  - `{entry.command}`")
        lines.append(f"      at {entry.workflow}:{entry.line}, step `{entry.step}`")
    lines.append(
        f"A change that passes every command in {entries[0].document} still fails that step, so "
        "add the command to the document. If the step is not a gate a contributor should run, "
        "that argument belongs in the pull request, not in an exemption here."
    )
    return "\n".join(lines)


def describe_unrun(entries: Sequence[Unrun]) -> str:
    """The failure text for a documented command no step runs."""
    if not entries:
        return "every command the document names is run by a gated workflow step"
    lines = [f"no gated workflow runs {len(entries)} command a document names:"]
    for entry in entries:
        lines.append(f"  - `{entry.command}`")
        lines.append(f"      documented as `{entry.span}` at {entry.document}:{entry.line}")
    lines.append(
        "The document asks a contributor to run something CI does not, so either the step is "
        f"missing from {'/'.join(GATED_WORKFLOWS)} or the document is wrong."
    )
    return "\n".join(lines)


@pytest.fixture(scope="module")
def gated_steps() -> list[Step]:
    steps: list[Step] = []
    for name in GATED_WORKFLOWS:
        path = WORKFLOW_DIR / name
        assert path.is_file(), f"{path} is absent, so the gate would compare an empty step set"
        steps.extend(workflow_steps(path))
    assert steps, "the gated workflows yielded no steps at all, which no document can satisfy"
    return steps


class TestTheScannerReadsEveryRunScalar:
    """Criterion 1: the commands come from the workflows, not from a list someone typed here."""

    def test_no_run_line_is_left_unread(self, gated_steps: list[Step]) -> None:
        """Every `run:` keyword in the three files became a step, or the scanner missed a shape."""
        for name in GATED_WORKFLOWS:
            path = WORKFLOW_DIR / name
            text = path.read_text(encoding="utf-8")
            declared = run_line_count(text)
            parsed = [
                step for step in gated_steps if step.workflow == display(path) and step.command
            ]
            assert parsed, f"{display(path)} declares {declared} run steps and none were read"
            assert len(parsed) == declared, (
                f"{display(path)} has {declared} `run:` lines but the scanner read "
                f"{len(parsed)} commands, so a step went unjudged"
            )

    def test_every_reported_line_is_the_run_keyword(self, gated_steps: list[Step]) -> None:
        """The line in a failure message has to be the line a reader can act on."""
        for name in GATED_WORKFLOWS:
            path = WORKFLOW_DIR / name
            lines = path.read_text(encoding="utf-8").splitlines()
            for step in (s for s in gated_steps if s.workflow == display(path) and s.command):
                line = lines[step.line - 1]
                assert line.strip().lstrip("- ").startswith("run:"), (
                    f"{step.where} was read as a command but its line declares no `run:`: {line!r}"
                )

    def test_a_block_scalar_is_read_as_one_command(self) -> None:
        text = (
            "jobs:\n  c:\n    steps:\n      - name: Install\n        run: |\n"
            "          python -m pip install --quiet . pytest ruff mypy\n"
        )
        steps = parse_steps(text, ".github/workflows/x.yml")
        assert [step.command for step in steps] == [
            "python -m pip install --quiet . pytest ruff mypy"
        ]
        assert steps[0].line == 5, "a block scalar is reported where `run:` sits, not in its body"

    def test_a_quoted_scalar_and_an_inline_comment_are_both_read(self) -> None:
        text = (
            'steps:\n  - run: "python -m pytest"\n  - name: Types\n'
            "    run: mypy src  # the package only\n"
        )
        steps = parse_steps(text, ".github/workflows/x.yml")
        assert [(step.name, step.norm) for step in steps] == [
            ("", "pytest"),
            ("Types", "mypy src"),
        ]

    def test_a_key_that_is_not_run_or_uses_is_not_a_step(self) -> None:
        text = "on:\n  push:\n    branches: [main]\nwith:\n  fetch-depth: 0\n  name: not a step\n"
        assert parse_steps(text, ".github/workflows/x.yml") == []

    def test_the_gated_workflows_run_the_commands_the_docs_are_about(
        self, gated_steps: list[Step]
    ) -> None:
        """A pin against a rewrite: the scanner has to see all eight checks, not a subset."""
        norms = {step.norm for step in gated_steps}
        for command in (
            "pytest",
            "ruff check",
            "ruff format --check",
            "mypy src",
            "airlock scan --mode full",
            "scripts/roadmap_progress.py --check",
            "scripts/sync_readme.py --check",
            "scripts/check_roadmap_touched.sh",
        ):
            assert command in norms, (
                f"`{command}` is not among the parsed commands: {sorted(norms)}"
            )


class TestSpellingsAreNormalised:
    """`airlock scan --mode full .` and `python -m airlock scan --mode full` are one check."""

    def test_the_console_script_and_the_module_are_the_same_command(self) -> None:
        assert normalise(SELF_SCAN_CI) == normalise(SELF_SCAN_MODULE) == "airlock scan --mode full"

    def test_the_interpreter_in_front_of_a_gate_script_is_dropped(self) -> None:
        assert normalise("python scripts/roadmap_progress.py --check") == normalise(
            "scripts/roadmap_progress.py --check"
        )
        assert (
            normalise("sh scripts/check_roadmap_touched.sh") == "scripts/check_roadmap_touched.sh"
        )

    def test_an_argument_ci_interpolates_is_not_part_of_the_command(self) -> None:
        assert (
            normalise('sh scripts/check_roadmap_touched.sh "origin/${BASE_REF}"')
            == "scripts/check_roadmap_touched.sh"
        )

    def test_the_directory_argument_that_means_here_is_dropped(self) -> None:
        assert normalise("ruff check .") == normalise("ruff check") == "ruff check"
        assert normalise("ruff format --check .") == "ruff format --check"

    def test_a_different_flag_is_a_different_command(self) -> None:
        assert normalise("ruff check --fix .") != normalise("ruff check .")

    def test_a_prose_path_is_not_a_command(self) -> None:
        assert not is_invocation("src/airlock/policy.py")
        assert not is_invocation("scripts/")
        assert not is_invocation("feat/<task-id>-<slug>")
        assert is_invocation("scripts/check_roadmap_touched.sh")


class TestTheComparison:
    """The pure logic, against inputs written here, so both directions are provably live."""

    SYNTHETIC = ".github/workflows/synthetic.yml"

    def steps(self, *commands: str) -> list[Step]:
        body = "".join(
            f"      - name: step {number}\n        run: {command}\n"
            for number, command in enumerate(commands, start=1)
        )
        return parse_steps(f"jobs:\n  checks:\n    steps:\n{body}", self.SYNTHETIC)

    def documented(self, *commands: str) -> list[Documented]:
        return [
            Documented("CONTRIBUTING.md", number, command, normalise(command))
            for number, command in enumerate(commands, start=1)
        ]

    def test_a_step_missing_from_the_document_fails_with_file_and_line(self) -> None:
        gate = compare(
            self.steps("python -m pytest", "mypy src"),
            "CONTRIBUTING.md",
            self.documented("python -m pytest"),
            EXEMPTIONS,
        )
        assert [entry.command for entry in gate.missing] == ["mypy src"]
        assert gate.missing[0].workflow == self.SYNTHETIC
        assert gate.missing[0].line == 7, "the second step, so the line points at its `run:`"
        text = describe_missing(gate.missing)
        for fragment in ("mypy src", self.SYNTHETIC, ":7", "step 2", "CONTRIBUTING.md"):
            assert fragment in text, f"the failure text lost {fragment!r}: {text}"

    def test_a_documented_command_that_matches_no_step_fails_too(self) -> None:
        gate = compare(
            self.steps("python -m pytest"),
            "CONTRIBUTING.md",
            self.documented("python -m pytest", "docker buildx build ."),
            EXEMPTIONS,
        )
        assert [entry.command for entry in gate.unrun] == ["docker buildx build"]
        assert gate.unrun[0].line == 2
        text = describe_unrun(gate.unrun)
        assert "docker buildx build" in text and "CONTRIBUTING.md:2" in text

    def test_an_exemption_stops_requiring_the_step(self) -> None:
        gate = compare(
            self.steps("python -m pytest", "python -m pip install --quiet . pytest"),
            "CONTRIBUTING.md",
            self.documented("python -m pytest"),
            EXEMPTIONS,
        )
        assert gate.clean and gate.exempt == ("pip install --quiet . pytest",)

    def test_an_exemption_also_forgives_the_document_naming_it_differently(self) -> None:
        """`SUPPORT.md` writes the install without `--quiet`; that is not a gate failure."""
        gate = compare(
            self.steps("python -m pip install --quiet . pytest ruff mypy"),
            "SUPPORT.md",
            self.documented("python -m pip install . pytest ruff mypy"),
            EXEMPTIONS,
        )
        assert gate.clean

    def test_a_deploy_action_is_exempt_because_it_is_not_a_command(self) -> None:
        steps = parse_steps(
            "jobs:\n  deploy:\n    steps:\n      - name: Deploy\n        id: deployment\n"
            "        uses: actions/deploy-pages@"
            "0000000000000000000000000000000000000000 # v5.0.1\n",
            ".github/workflows/pages.yml",
        )
        assert _exemption_for_step(steps[0], EXEMPTIONS) is not None

    def test_the_gate_is_exact_when_document_and_ci_agree(self) -> None:
        commands = ("python -m pytest", "ruff check .", "mypy src")
        gate = compare(self.steps(*commands), "CONTRIBUTING.md", self.documented(*commands), ())
        assert gate.clean and not gate.exempt

    def test_a_document_that_names_nothing_leaves_every_step_undocumented(self) -> None:
        """The shape of the bug this task closes, read straight off the comparison."""
        gate = compare(
            self.steps("python -m pytest", "mypy src"), "CONTRIBUTING.md", (), EXEMPTIONS
        )
        assert len(gate.missing) == 2

    def test_a_new_step_is_not_silently_allowed_by_an_old_document(self) -> None:
        old = self.documented("python -m pytest", "ruff check .", "mypy src")
        gate = compare(
            self.steps("python -m pytest", "ruff check .", "mypy src", "taplo fmt"),
            "CONTRIBUTING.md",
            old,
            EXEMPTIONS,
        )
        assert [entry.command for entry in gate.missing] == ["taplo fmt"]

    def test_a_documented_flag_ci_does_not_run_fails_in_both_directions(self) -> None:
        """`mypy --strict src` is not `mypy src`, in either direction, prefix leniency aside."""
        gate = compare(
            self.steps("mypy src"), "CONTRIBUTING.md", self.documented("mypy --strict src"), ()
        )
        assert [entry.command for entry in gate.missing] == ["mypy src"]
        assert [entry.command for entry in gate.unrun] == ["mypy --strict src"]


class TestChecklistAgainstCi:
    """The gate itself, over the real documents and the real workflows."""

    @pytest.mark.parametrize("document", DOCUMENTS, ids=lambda item: item.path)
    def test_every_ci_command_is_documented(
        self, document: Document, gated_steps: list[Step]
    ) -> None:
        gate = compare(gated_steps, document.path, document_commands(document), EXEMPTIONS)
        assert not gate.missing, describe_missing(gate.missing)

    @pytest.mark.parametrize("document", DOCUMENTS, ids=lambda item: item.path)
    def test_no_documented_command_is_one_ci_never_runs(
        self, document: Document, gated_steps: list[Step]
    ) -> None:
        gate = compare(gated_steps, document.path, document_commands(document), EXEMPTIONS)
        assert not gate.unrun, describe_unrun(gate.unrun)

    @pytest.mark.parametrize("document", DOCUMENTS, ids=lambda item: item.path)
    def test_each_document_names_commands_worth_reading(self, document: Document) -> None:
        """A reader that finds nothing could never fail, so finding something is asserted."""
        found = document_commands(document)
        assert len(found) >= 4, (
            f"{document.path} yielded {len(found)} commands ({[item.command for item in found]}); "
            "the extractor and the document have diverged, and the gate is reading nothing"
        )

    def test_every_exemption_is_exercised_by_a_real_step(self, gated_steps: list[Step]) -> None:
        """An exemption nobody uses is a hole, so the three reasons must each match a step."""
        for exemption in EXEMPTIONS:
            matched = [
                step for step in gated_steps if _exemption_for_step(step, EXEMPTIONS) is exemption
            ]
            assert matched, (
                f"exemption `{exemption.pattern}` matches no step in {'/'.join(GATED_WORKFLOWS)}: "
                f"it is carried for a reason that no longer applies ({exemption.reason})"
            )

    def test_the_install_steps_are_exempt_in_every_job_that_needs_them(
        self, gated_steps: list[Step]
    ) -> None:
        """`AGENTS.md` 5 asks for the install before the checks, and three jobs do it."""
        install = [step for step in gated_steps if step.norm.startswith("pip install")]
        assert len(install) >= 3, f"expected an install in each job, saw {install}"
        assert all(_exemption_for_step(step, EXEMPTIONS) for step in install), (
            "an install step is not exempt, so a contributor is being asked to run it as a check: "
            f"{[step.where for step in install]}"
        )

    def test_the_three_documents_all_name_the_four_core_checks(self) -> None:
        """The claim issue #99 rests on: the four commands CI runs are in every list."""
        for document in DOCUMENTS:
            named = {item.command for item in document_commands(document)}
            for command in ("pytest", "ruff check", "ruff format --check", "mypy src"):
                assert command in named, f"{document.path} does not name `{command}`"

    def test_the_self_scan_runs_once_and_both_spellings_are_known(
        self, gated_steps: list[Step]
    ) -> None:
        """`ci.yml` runs the installed console script; `AGENTS.md` writes the module form."""
        scans = [step for step in gated_steps if step.norm == normalise(SELF_SCAN_CI)]
        assert [step.workflow for step in scans] == [".github/workflows/ci.yml"], scans
        assert normalise(SELF_SCAN_MODULE) in {
            item.command for item in document_commands(Document("AGENTS.md", "fenced"))
        }
