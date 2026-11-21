"""The install lines in `.github/workflows/` and the dev group in `pyproject.toml` (task 9.74).

Two files agree about one thing: which versions of `pytest`, `ruff` and `mypy` a check is run with.
`pyproject.toml` writes the floors in `[dependency-groups]`, and the workflows write them again, in
freehand, inside a `run:` block. Only the first is read by a human who is changing a floor, so the
second drifts. Nothing in the suite noticed, because nothing read both.

The failure mode is not only drift. A specifier pasted into a `run:` block without quotes is not a
specifier at all: the shell sees `pytest>=8` as the word `pytest` with standard output redirected to
a file named `=8`, so the floor disappears from the command line and a stray file appears in the
workspace. Both tests here read the raw YAML text rather than a parsed value precisely so that the
quoting is part of what is checked.

There is no YAML parser available (the project installs no runtime dependency and PyYAML is not in
the dev group), and a real parser would hand back the unquoted value, which is the one thing that
has to stay visible. So the workflow side is tokenised with the shell's own quoting rules.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: One `pip install` invocation per line, captured with the line it sits on so a failure can name
#: the place rather than the file. These files write the command as a block scalar, except
#: `roadmap-check.yml`, which writes `run: python -m pip install ...` inline, so both are matched.
_PIP_INSTALL = re.compile(r"^(?:run:\s*)?python(?:3)? -m pip install\b(?P<args>.*)$")

#: Shell tokens: a double-quoted run, a single-quoted run, or a bare word up to the next space.
_TOKEN = re.compile(r'"[^"]*"|\'[^\']*\'|[^\s]+')

#: A bare word that still carries a redirection character, which is what an unquoted specifier
#: looks like to the shell before it splits the word from the operator.
_BARE_OPERATOR = re.compile(r"[<>]")

#: `pip install` takes the package name plus an optional specifier, or a local path.
_REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?P<specifier>.*)$")


@dataclass(frozen=True)
class Spec:
    """One requirement exactly as the workflow writes it, plus the spelling that made it safe.

    `quoted` is what the shell saw, not what pip did: a quoted token and a bare one both unquote to
    `pytest>=8`, and the difference between them is whether the floor arrives at pip at all.
    """

    raw: str
    value: str
    quoted: bool
    name: str
    specifier: str

    @property
    def is_local_path(self) -> bool:
        return self.value in (".", "./")


@dataclass(frozen=True)
class InstallLine:
    """A `pip install` command in a workflow, with its requirements already split out."""

    file: Path
    line: int
    args: str
    flags: tuple[str, ...]
    specs: tuple[Spec, ...]

    @property
    def label(self) -> str:
        """Where this command is, in the form a reviewer can open.

        Synthetic lines built inside the tests have no place in the repository, and a failure that
        printed a path under `.github/` for one of them would send a reviewer to a file that is
        fine, so those name the step they came from instead.
        """
        if self.file.is_absolute():
            return f"{self.file.relative_to(REPO_ROOT)}:{self.line}"
        return f"{self.file} (case) :{self.line}"


@dataclass(frozen=True)
class Problem:
    """Why a workflow line and the dev group do not agree, phrased with both locations.

    The test's usefulness is a reviewer being able to act on it without opening two files, so every
    problem names the tool, what CI says, what `pyproject.toml` says, and the line number.
    """

    tool: str
    ci: str
    declared: str
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.tool}: {self.detail}\n"
            f"    CI installs           {self.ci}\n"
            f"    pyproject.toml declares {self.declared}"
        )


def dev_group(text: str) -> dict[str, str]:
    """The declared floors as `{name: specifier}`, from a `pyproject.toml` text.

    Read from the file rather than written here: a copy of the floors in the test would be a third
    place for them to live, and the point of the task is that there are two.
    """
    config = tomllib.loads(text)
    group = config["dependency-groups"]["dev"]
    parsed: dict[str, str] = {}
    for entry in group:
        match = _REQUIREMENT.match(entry)
        assert match is not None, f"unparseable requirement in the dev group: {entry!r}"
        parsed[match["name"].lower()] = match["specifier"]
    return parsed


def tokenize(args: str) -> tuple[list[str], list[Spec]]:
    """Split a `pip install` argument tail into flags and requirements, keeping the shell quoting.

    Returns `(flags, specs)`. A flag that arrives as one quoted token containing a space is
    collected as a flag here rather than rejected, because the rejection is its own test: an
    option that eats its neighbour is exactly how a package silently stops being installed.
    """
    flags: list[str] = []
    specs: list[Spec] = []
    for raw in _TOKEN.findall(args):
        quoted = raw[0] in "\"'"
        value = raw[1:-1] if quoted else raw
        if value.startswith("-"):
            flags.append(value)
            continue
        match = _REQUIREMENT.match(value)
        if match is None:
            continue
        specs.append(
            Spec(
                raw=raw,
                value=value,
                quoted=quoted,
                name=match["name"].lower(),
                specifier=match["specifier"],
            )
        )
    return flags, specs


def install_lines(workflow_text: str, file: Path | None = None) -> list[InstallLine]:
    """Every `pip install` in a workflow, with the line number it appears on.

    Each `run:` value in these files is a block scalar, so the command is a whole line and a line
    scan finds every one of them. A flow-style `run: python -m pip install ...` would be missed,
    and `test_every_install_line_in_the_repo_is_found_by_the_line_scan` keeps that true.

    `file` is absolute for a file on disk and relative for a case built inside a test, which is how
    `label` tells the two apart.
    """
    origin = file if file is not None else Path("case")
    found: list[InstallLine] = []
    for number, text in enumerate(workflow_text.split("\n"), start=1):
        match = _PIP_INSTALL.match(text.strip())
        if match is None:
            continue
        flags, specs = tokenize(match["args"])
        found.append(
            InstallLine(
                file=origin, line=number, args=match["args"], flags=flags, specs=tuple(specs)
            )
        )
    return found


def workflow_installs() -> list[InstallLine]:
    """Every install line in every workflow, with its real file attached."""
    found: list[InstallLine] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        found.extend(install_lines(path.read_text(encoding="utf-8"), file=path))
    return found


def compare(installs: list[InstallLine], declared: dict[str, str]) -> list[Problem]:
    """What CI would install against what the project declares, as a list of disagreements.

    A tool that CI installs and the dev group does not name is reported, and so is the reverse
    inside the job that is meant to run the whole group. Subsets are allowed elsewhere on purpose:
    the roadmap gate job runs `pytest` and nothing else, and making it install mypy to satisfy a
    test would be the test deciding CI's jobs for it.
    """
    problems: list[Problem] = []
    for install in installs:
        for spec in install.specs:
            if spec.is_local_path:
                continue
            if _BARE_OPERATOR.search(spec.raw) is not None and not spec.quoted:
                problems.append(
                    Problem(
                        tool=spec.name,
                        ci=f"{install.label} {spec.raw}",
                        declared=f"{spec.name}{declared.get(spec.name, '?')}",
                        detail=(
                            "the specifier is not quoted, so the shell reads `>` as a redirect "
                            "and pip never sees the floor"
                        ),
                    )
                )
                continue
            floor = declared.get(spec.name)
            if floor is None:
                problems.append(
                    Problem(
                        tool=spec.name,
                        ci=f"{install.label} {spec.raw}",
                        declared="(absent)",
                        detail="CI installs a tool the dev group does not declare",
                    )
                )
            elif spec.specifier != floor:
                problems.append(
                    Problem(
                        tool=spec.name,
                        ci=f"{install.label} {spec.raw}",
                        declared=f"{spec.name}{floor}",
                        detail="the floor in CI is not the floor in the dev group",
                    )
                )
            elif not spec.quoted:
                problems.append(
                    Problem(
                        tool=spec.name,
                        ci=f"{install.label} {spec.raw}",
                        declared=f"{spec.name}{floor}",
                        detail="a specifier with an operator has to be quoted",
                    )
                )
    return problems


@pytest.fixture(scope="module")
def declared_floors() -> dict[str, str]:
    return dev_group(PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def installs() -> list[InstallLine]:
    return workflow_installs()


class TestTheDeclaredFloors:
    def test_the_dev_group_is_the_thing_being_read(self, declared_floors: dict[str, str]) -> None:
        assert sorted(declared_floors) == ["mypy", "pytest", "ruff"]

    def test_every_floor_names_a_lower_bound_not_an_exact_pin(
        self, declared_floors: dict[str, str]
    ) -> None:
        """A `==` would make Dependabot's job this test's job, and the group is a floor set."""
        for name, floor in declared_floors.items():
            assert floor.startswith(">="), f"{name} declares {floor!r}, which is not a floor"

    def test_the_group_is_what_the_issue_records(self, declared_floors: dict[str, str]) -> None:
        assert declared_floors == {"pytest": ">=8", "ruff": ">=0.6", "mypy": ">=1.11"}


class TestQuotingInPlace:
    def test_no_install_specifier_reaches_the_shell_bare(self, installs: list[InstallLine]) -> None:
        bare = [
            f"{entry.label}: {spec.raw}"
            for entry in installs
            for spec in entry.specs
            if not spec.quoted and _BARE_OPERATOR.search(spec.value)
        ]
        assert not bare, "unquoted specifiers redirect instead of pinning: " + "; ".join(bare)

    def test_an_unquoted_specifier_is_caught_and_named(
        self, declared_floors: dict[str, str]
    ) -> None:
        """The word here is the one that would create a file called `=8` in the workspace."""
        problems = compare(
            install_lines('python -m pip install --quiet . pytest>=8 "ruff>=0.6" "mypy>=1.11"'),
            declared_floors,
        )
        assert [problem.tool for problem in problems] == ["pytest"]
        assert "redirect" in str(problems[0])

    def test_a_quoted_specifier_is_clean(self, declared_floors: dict[str, str]) -> None:
        assert (
            compare(
                install_lines(
                    'python -m pip install --quiet . "pytest>=8" "ruff>=0.6" "mypy>=1.11"'
                ),
                declared_floors,
            )
            == []
        )

    def test_a_float_without_an_operator_needs_no_quotes(
        self, declared_floors: dict[str, str]
    ) -> None:
        """`pytest` is not a redirect hazard; it is a floor that is missing, which is the other
        half of the same check and is reported as `the floor in CI is not the floor`."""
        problems = compare(install_lines("python -m pip install --quiet . pytest"), declared_floors)
        assert [p.tool for p in problems] == ["pytest"]
        assert "floor" in str(problems[0])


class TestDriftBetweenTheTwoFiles:
    def test_the_workflows_and_the_dev_group_agree_today(
        self, installs: list[InstallLine], declared_floors: dict[str, str]
    ) -> None:
        problems = compare(installs, declared_floors)
        assert not problems, "\n".join(str(problem) for problem in problems)

    def test_a_floor_moved_in_ci_alone_is_reported_with_both_places(
        self, declared_floors: dict[str, str]
    ) -> None:
        problems = compare(
            install_lines('python -m pip install --quiet . "pytest>=9" "ruff>=0.6" "mypy>=1.11"'),
            declared_floors,
        )
        assert len(problems) == 1
        assert problems[0].tool == "pytest"
        assert ">=9" in problems[0].ci
        assert ">=8" in problems[0].declared

    def test_a_floor_moved_in_the_group_alone_is_the_same_problem(
        self, declared_floors: dict[str, str]
    ) -> None:
        """The test reads both sides, so moving either one trips it; this moves the group."""
        moved = dict(declared_floors, ruff=">=0.12")
        problems = compare(
            install_lines('python -m pip install --quiet . "pytest>=8" "ruff>=0.6" "mypy>=1.11"'),
            moved,
        )
        assert [p.tool for p in problems] == ["ruff"]

    def test_a_tool_in_ci_but_not_in_the_group_is_reported(
        self, declared_floors: dict[str, str]
    ) -> None:
        problems = compare(
            install_lines('python -m pip install --quiet . "bandit>=1.7" "pytest>=8"'),
            declared_floors,
        )
        assert [p.tool for p in problems] == ["bandit"]
        assert "(absent)" in str(problems[0])

    def test_a_new_tool_in_the_group_does_not_invent_itself_into_ci(
        self, declared_floors: dict[str, str]
    ) -> None:
        """Adding to the group is a deliberate act; this test does not force CI to install it, and
        says so rather than failing a job that only needs pytest."""
        added = dict(declared_floors, semgrep=">=1.90")
        assert compare(install_lines('python -m pip install --quiet . "pytest>=8"'), added) == []


class TestTheConsoleScriptStaysInstalled:
    def test_every_install_line_installs_the_package_itself(
        self, installs: list[InstallLine]
    ) -> None:
        """`AGENTS.md` 5: the self-scan needs the console script, so no edit here may drop `.`."""
        missing = [
            entry.label for entry in installs if not any(s.is_local_path for s in entry.specs)
        ]
        assert not missing, "install lines without the local package: " + ", ".join(missing)

    def test_the_local_path_is_a_separate_argument(self, installs: list[InstallLine]) -> None:
        """A quoted token holding `--quiet .` is one argument to the shell, and pip's `--quiet` is
        a counter that takes no value; the path has to be its own bare token."""
        glued = [
            f"{entry.label}: {entry.flags}"
            for entry in installs
            for flag in entry.flags
            if len(flag.split()) > 1
        ]
        assert not glued, "flag glued to another word: " + "; ".join(glued)


class TestTheParserMatchesTheFiles:
    def test_every_install_line_in_the_repo_is_found_by_the_line_scan(
        self, installs: list[InstallLine]
    ) -> None:
        """A `run:` written flow-style would be invisible to the scan, so count the same way the
        shell would: grep the tree for the command and compare."""
        grep = [
            (path, number)
            for path in sorted(WORKFLOWS.glob("*.yml"))
            for number, text in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1)
            if "pip install" in text
        ]
        scanned = [(entry.file, entry.line) for entry in installs]
        assert scanned == grep, f"line scan found {scanned}, the tree has {grep}"

    def test_the_quoted_line_is_the_line_ci_runs(self, installs: list[InstallLine]) -> None:
        """Not a snapshot of the version floors, which the other tests read from the group: this
        one says the checks job installs the whole group in one command, quoted."""
        ci = WORKFLOWS / "ci.yml"
        checks = [entry for entry in installs if entry.file == ci]
        install_steps = [entry for entry in checks if len(entry.specs) > 1]
        assert len(install_steps) == 1, [entry.label for entry in checks]
        specs = install_steps[0].specs
        assert [(s.value, s.quoted) for s in specs] == [
            (".", False),
            ("pytest>=8", True),
            ("ruff>=0.6", True),
            ("mypy>=1.11", True),
        ]

    def test_the_self_scan_job_installs_only_the_package(self, installs: list[InstallLine]) -> None:
        ci = WORKFLOWS / "ci.yml"
        solo = [
            entry
            for entry in installs
            if entry.file == ci and [s.value for s in entry.specs] == ["."]
        ]
        assert len(solo) == 1

    def test_quoting_is_preserved_from_the_file_not_from_a_parser(
        self, installs: list[InstallLine]
    ) -> None:
        """`raw` still has the quote characters; a parsed YAML scalar keeps them too, but this test
        exists so that the next refactor to a YAML library notices this one depends on them."""
        quoted_raws = [spec.raw for entry in installs for spec in entry.specs if spec.quoted]
        assert quoted_raws, "no quoted specifier anywhere in the workflows"
        assert all(raw[0] in "\"'" and raw[-1] == raw[0] for raw in quoted_raws)
