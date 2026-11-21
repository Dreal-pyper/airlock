"""The commands in the README's quick start are run, not reread (task 9.72).

The page is the first thing a new user executes, and until now nothing in the suite executed it.
`tests/unit/test_cli.py` calls `main()` with its own argv, and the manifest paths go through the
helpers in `src/airlock/cli.py`, so both agree with the implementation while the page says something
else. A renamed flag in `src/airlock/cli.py` would leave the documented line broken and the
suite green, which is the exact gap this task names. `TestFlagsAgreeWithTheParser` catches that
rename from the flag list, and the subprocess tests catch it from the exit code.

So the command text is taken from the file: the fenced `bash` blocks under `## Quick start` are
extracted, shell continuations are joined, and each surviving line is run through a shell with `cwd`
in a temporary git repository. Copying the commands into this module would test the copy.

Two classes of line are filtered rather than run, and both are asserted on: the install snippet
(`git clone`, `cd`, `python3 -m venv`, `pip install`) needs the network and would install over the
interpreter running the suite, and `--event pull_request --base <sha>` is prose with a placeholder,
not a command. Nothing else is excluded, and `test_what_was_filtered_is_filtered_by_rule` says which
line went which way.

The exit codes come from the table at the end of the same section, so the assertion is "the page
promises this number for this situation" and not a constant in here. `shell=True` is used because a
globe and a quoted `--build` argument are part of what the page tells a reader to type; splitting
the line in Python would test the splitter.

Like the other subprocess suites here this needs the interpreter's own `bin` first on `PATH`: the
`airlock` and `python3`, and under the venv both are the 3.11 that `requires-python` declares.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.fixtures import assemble

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

#: The section a new user is pointed at, and the table that closes it. The fence boundaries are
#: found from the heading rather than hard-coded line numbers, because the page reflows.
_QUICK_START = re.compile(r"^## Quick start$", re.MULTILINE)
_NEXT_H3_SECTION = re.compile(r"^## ", re.MULTILINE)
_BASH_FENCE = re.compile(r"^```bash\n(?P<body>.*?)^```$", re.MULTILINE | re.DOTALL)

#: Lines the reader runs somewhere else. Cloning and installing are the setup the page assumes has
#: already happened by the time the scan lines appear.
_FILTERED_PREFIXES = ("git clone", "cd ", "python3 -m venv", "pip install", "python -m pip install")

#: A `<...>` is a placeholder the reader replaces, so the line cannot be executed as written.
_PLACEHOLDER = re.compile(r"<[^>\n]+>")

_EXIT_TABLE_ROW = re.compile(r"^\|\s*(?P<code>\d)\s*\|\s*(?P<meaning>[^|]+)\|", re.MULTILINE)

#: Flag-looking tokens in a documented command, used to ask the real parser about each one.
_FLAG = re.compile(r"--[a-z][a-z-]*")

#: Shell tokens: a double-quoted run, a single-quoted run, or a bare word up to the next space.
_SHELL_TOKEN = re.compile(r'"[^"]*"|\'[^\']*\'|[^\s]+')

#: The words the scanner itself prints as its verdict, in the order the summary line uses them.
_HIGH_COUNT = re.compile(r"airlock: (?P<high>\d+) HIGH")

CLEAN_CONFIG = "export default { plugins: [] }\n"
# Built from fragments at call time (`AGENTS.md` 4): this repository is scanned by its own rules as
# role `source`, so a committed literal of the payload line would be a HIGH finding against it.
PAYLOAD = f"{assemble('ev', 'al')}({assemble('Buf', 'fer.from')}(process.env.P, 'base64'))\n"

BASE_POLICY = """version = 1

[policy]
fail_on = "high"
line_cap = 500
"""


@dataclass(frozen=True)
class Command:
    """One executable line from the page, with the fence it came from.

    `line` is the README line number of the first physical line, so a failure points at the page
    instead of at a number in here.
    """

    text: str
    line: int

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Run:
    """A documented command and what actually happened when it was typed."""

    command: Command
    code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    def describe(self, expected: int, row: str) -> str:
        return (
            f"README:{self.command.line} says `{self.command.text}` => exit {expected} ({row}), "
            f"it exited {self.code}\n{self.output.strip()}"
        )


def quick_start(readme: str) -> str:
    """The text of the `## Quick start` section, up to the next `##` heading."""
    start = _QUICK_START.search(readme)
    assert start is not None, "README.md has no `## Quick start` section to run"
    rest = readme[start.end() :]
    end = _NEXT_H3_SECTION.search(rest)
    return rest[: end.start()] if end else rest


def join_continuations(block: str) -> list[tuple[str, int]]:
    """Merge backslash-newline continuations into one command, keeping the first line's number.

    The page writes the `manifest add` command over three physical lines. Without the join the
    second and third would be run as separate commands, which is not what a reader types.
    """
    joined: list[tuple[str, int]] = []
    number = 0
    pending: list[str] = []
    for offset, raw in enumerate(block.split("\n")):
        stripped = raw.strip()
        if not stripped:
            continue
        if not pending:
            number = offset + 1
        if stripped.endswith("\\"):
            pending.append(stripped[:-1].strip())
            continue
        pending.append(stripped)
        joined.append((" ".join(pending), number))
        pending = []
    if pending:
        joined.append((" ".join(pending), number))
    return joined


def documented_commands(readme: str) -> list[Command]:
    """Every executable line in the quick start's bash fences, continuations joined."""
    section = quick_start(readme)
    base = readme[: readme.index(section)].count("\n")
    commands: list[Command] = []
    for fence in _BASH_FENCE.finditer(section):
        first_line = base + section[: fence.start()].count("\n") + 2
        for text, offset in join_continuations(fence.group("body")):
            if any(text.startswith(prefix) for prefix in _FILTERED_PREFIXES):
                continue
            commands.append(Command(text=text, line=first_line + offset - 1))
    return commands


def filtered_commands(readme: str) -> list[str]:
    """The fence lines this suite deliberately does not run, for the record."""
    section = quick_start(readme)
    kept: list[str] = []
    for fence in _BASH_FENCE.finditer(section):
        for text, _ in join_continuations(fence.group("body")):
            if any(text.startswith(prefix) for prefix in _FILTERED_PREFIXES):
                kept.append(text)
    return kept


def exit_code_table(readme: str) -> dict[int, str]:
    """The `### Exit codes` rows, as `{code: meaning}`."""
    rows = {
        int(match["code"]): match["meaning"].strip() for match in _EXIT_TABLE_ROW.finditer(readme)
    }
    assert rows, "README.md has no exit code table to hold the tool to"
    return rows


def select(commands: list[Command], needle: str) -> Command:
    """The one documented command a test means: exact text first, then a unique substring.

    `airlock scan --mode full` is a substring of the `PYTHONPATH=src python3 -m airlock` line, so
    exact matching comes first; a command renamed on the page matches neither way and this reports
    the whole list rather than running the wrong line.
    """
    found = [command for command in commands if command.text == needle] or [
        command for command in commands if needle in command.text
    ]
    if len(found) != 1:
        listed = "\n".join(f"  {command.line}: {command}" for command in commands)
        raise AssertionError(
            f"expected one quick start command for {needle!r}, found {len(found)}.\n{listed}"
        )
    return found[0]


def flags_outside_quotes(text: str) -> set[str]:
    """The long flags the shell would pass to the tool, not the words inside an argument.

    `--build "cargo build --release --target ..."` documents a flag whose *value* is another
    program's command line; scanning the raw line would ask this repository's parser about cargo.
    """
    flags: set[str] = set()
    for raw in _SHELL_TOKEN.findall(text):
        quoted = raw[0] in "\"'"
        value = raw[1:-1] if quoted else raw
        if not quoted and value.startswith("--"):
            flags.update(_FLAG.findall(value))
    return flags


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=airlock-test",
            "-c",
            "user.email=airlock-test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def write(root: Path, rel: str, data: str, *, stage: bool = True) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    if stage:
        git(root, "add", "-A")


def run_command(command: Command, repo: Path) -> Run:
    """Type the documented line into a shell whose working directory is `repo`.

    The environment is the suite's own with the interpreter's `bin` first, because the page assumes
    an activated venv and `pip install .`: `airlock` is the console script and `python3` is the
    interpreter the reader chose. Nothing in the command text is rewritten.
    """
    env = dict(os.environ)
    bin_dir = str(Path(sys.executable).parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        command.text,
        shell=True,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return Run(command=command, code=result.returncode, stdout=result.stdout, stderr=result.stderr)


def check_exit(run: Run, expected: int, row: str) -> None:
    """Assert the documented number, and say which command, which number and which row if not."""
    if run.code != expected:
        raise AssertionError(run.describe(expected, row))


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture
def commands(readme: str) -> list[Command]:
    return documented_commands(readme)


@pytest.fixture
def table(readme: str) -> dict[int, str]:
    return exit_code_table(readme)


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """A committed repository with a policy and one clean build config, as the page assumes.

    `src` is a symlink to the package so the module-run line finds it relative to the working
    directory exactly as a reader's checkout would, and the collector does not follow it, so the
    scan sees three files rather than the scanner's own tree.
    """
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "--initial-branch=main")
    write(root, "airlock.toml", BASE_POLICY, stage=False)
    write(root, "wasm-manifest.toml", "version = 1\n", stage=False)
    write(root, "postcss.config.mjs", CLEAN_CONFIG, stage=False)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "documented state")
    (root / "src").symlink_to(REPO_ROOT / "src")
    return root


@pytest.fixture
def wasm_tree(site: Path) -> Path:
    """The same repository with one artifact at the path the page's globe matches."""
    write(site, "target/wasm32-unknown-unknown/counter.wasm", "", stage=False)
    (site / "target/wasm32-unknown-unknown/counter.wasm").write_bytes(
        b"\x00asm\x01\x00\x00\x00\x00\x04\x01a\x00\x00\x00\x00"
    )
    git(site, "add", "-A")
    git(site, "commit", "-q", "-m", "artifact")
    return site


def help_flags(verb: tuple[str, ...]) -> set[str]:
    """Every long flag argparse advertises for one command, read out of its own `--help`.

    This asks the parser rather than walking it: argparse reaches the flags through private
    attributes, and a test that reads `parser._actions` breaks when the CLI is right.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "airlock", *verb, "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(_FLAG.findall(result.stdout.replace("\n", " ")))


class TestExtraction:
    def test_the_page_yields_the_commands_a_reader_types(self, commands: list[Command]) -> None:
        assert [command.text for command in commands] == [
            "airlock scan --mode full",
            "airlock manifest add target/*/*.wasm --origin built --source contracts/counter "
            '--build "cargo build --release --target wasm32-unknown-unknown"',
            "airlock manifest check",
            "PYTHONPATH=src python3 -m airlock scan --mode full",
        ]

    def test_what_was_filtered_is_filtered_by_rule(self, readme: str) -> None:
        """Nothing disappears quietly: the setup snippet is the only thing not run."""
        assert filtered_commands(readme) == [
            "git clone https://github.com/posture-sec/airlock.git",
            "cd airlock",
            "python3 -m venv .venv && . .venv/bin/activate",
            "pip install .",
        ]

    def test_a_continuation_is_one_command_not_three(self, commands: list[Command]) -> None:
        adds = [command for command in commands if "manifest add" in command.text]
        assert len(adds) == 1
        assert "--build" in adds[0].text
        assert not adds[0].text.endswith("\\")

    def test_no_documented_command_is_left_unrun_by_accident(self, commands: list[Command]) -> None:
        """A placeholder line is prose, and the fences hold no other excuse for not running."""
        assert not [command for command in commands if _PLACEHOLDER.search(command.text)]

    def test_the_exit_table_is_read_from_the_page(self, table: dict[int, str]) -> None:
        assert sorted(table) == [0, 1, 2, 3]
        assert "threshold" in table[1]
        assert "unresolvable base" in table[2]


class TestFlagsAgreeWithTheParser:
    def test_every_flag_in_the_quick_start_exists(self, commands: list[Command]) -> None:
        """A rename in `build_parser()` is the failure this task was written for, caught here
        before the subprocess step, so the report names the flag rather than an argparse message."""
        known: set[str] = set()
        for verb in (("scan",), ("manifest", "add"), ("manifest", "check")):
            known |= help_flags(verb)
        used = {flag for command in commands for flag in flags_outside_quotes(command.text)}
        missing = sorted(used - known)
        assert not missing, f"the page documents flags the CLI does not have: {missing}"

    def test_a_flag_inside_an_argument_is_not_counted_as_a_flag(self) -> None:
        """The cargo line inside `--build` is a value, and this is the guard that keeps the check
        above from becoming a scan for the string `--`."""
        assert flags_outside_quotes('airlock manifest add x --build "cargo build --release"') == {
            "--build"
        }

    def test_the_mode_value_named_on_the_page_is_a_mode(self, commands: list[Command]) -> None:
        from airlock.model import Mode

        documented = {
            match.group(1)
            for command in commands
            for match in re.finditer(r"--mode ([a-z_]+)", command.text)
        }
        assert documented == {"full"}
        assert documented <= {member.value for member in Mode}


class TestDocumentedBehaviour:
    def test_scan_mode_full_exits_zero_and_reports_no_high(
        self, commands: list[Command], site: Path, table: dict[int, str]
    ) -> None:
        run = run_command(select(commands, "airlock scan --mode full"), site)
        check_exit(run, 0, table[0])
        match = _HIGH_COUNT.search(run.output)
        assert match is not None, run.output
        assert int(match["high"]) == 0

    def test_manifest_add_writes_the_manifest_then_check_passes(
        self, commands: list[Command], wasm_tree: Path, table: dict[int, str]
    ) -> None:
        """The page's order is the page's promise: record, then verify the tree you recorded."""
        add = run_command(select(commands, "manifest add"), wasm_tree)
        check_exit(add, 0, table[0])
        manifest = wasm_tree / "wasm-manifest.toml"
        assert "counter.wasm" in manifest.read_text(encoding="utf-8")
        assert 'origin = "built"' in manifest.read_text(encoding="utf-8")
        assert 'source = "contracts/counter"' in manifest.read_text(encoding="utf-8")

        check = run_command(select(commands, "manifest check"), wasm_tree)
        check_exit(check, 0, table[0])

    def test_the_module_run_line_works_as_written(
        self, commands: list[Command], site: Path, table: dict[int, str]
    ) -> None:
        """`PYTHONPATH=src` is relative on purpose: it resolves against the working directory, so
        the line only holds if the package is reachable from where the reader is standing."""
        run = run_command(select(commands, "PYTHONPATH=src"), site)
        check_exit(run, 0, table[0])
        assert "0 HIGH" in run.output

    def test_a_scan_with_a_finding_exits_the_number_the_table_gives(
        self, commands: list[Command], site: Path, table: dict[int, str]
    ) -> None:
        write(site, "postcss.config.mjs", PAYLOAD)
        git(site, "commit", "-q", "-m", "payload")
        run = run_command(select(commands, "airlock scan --mode full"), site)
        check_exit(run, 1, table[1])
        match = _HIGH_COUNT.search(run.output)
        assert match is not None, run.output
        assert int(match["high"]) >= 1, run.output

    def test_an_unresolvable_base_takes_the_configuration_row(
        self, site: Path, table: dict[int, str]
    ) -> None:
        """Not a quick start line, but the row directly above it promises it: `--base <sha>` is
        exit 2 when the base cannot be resolved, and that is the trust model the page rests on."""
        run = run_command(Command(text="airlock scan --mode pr --base " + "0" * 40, line=0), site)
        check_exit(run, 2, table[2])
        assert "reachable" in run.output.lower()

    def test_the_tool_never_exits_a_number_the_page_does_not_list(
        self, commands: list[Command], site: Path, table: dict[int, str]
    ) -> None:
        for needle in ("airlock scan --mode full", "manifest check", "PYTHONPATH=src"):
            found = select(commands, needle)
            assert run_command(found, site).code in table, found.text


class TestTeeth:
    def test_a_renamed_flag_fails_here_rather_than_staying_green(
        self, commands: list[Command], site: Path, table: dict[int, str]
    ) -> None:
        """The regression this task exists for, applied to the extracted text instead of to the
        page: `--mode` becomes a flag the CLI does not have, and the documented exit code no
        longer matches."""
        renamed = Command(
            text=select(commands, "airlock scan --mode full").text.replace("--mode", "--model"),
            line=0,
        )
        run = run_command(renamed, site)
        assert run.code != 0
        with pytest.raises(AssertionError) as raised:
            check_exit(run, 0, table[0])
        message = str(raised.value)
        assert "exit 0" in message and f"exited {run.code}" in message
        assert "unrecognized" in run.output.lower() or "invalid" in run.output.lower()

    def test_a_wrong_documented_code_is_reported_with_the_row(self, site: Path) -> None:
        """The page's table is the contract. If the page said a clean scan exits 1, that is the
        thing that must be reported, with the command, the row and the number actually seen."""
        run = run_command(Command(text="airlock scan --mode full", line=193), site)
        assert run.code == 0
        with pytest.raises(AssertionError) as raised:
            check_exit(run, 1, "a finding at or above the threshold")
        message = str(raised.value)
        assert "README:193" in message
        assert "airlock scan --mode full" in message
        assert "=> exit 1 (a finding at or above the threshold)" in message
        assert "it exited 0" in message

    def test_a_scan_that_never_ran_is_not_read_as_a_pass(self, site: Path) -> None:
        """An assertion on the code alone would pass on a command that failed to start, so the
        verdict line is what proves a scan happened."""
        missing_verb = Command(text="airlock scans --mode full", line=0)
        run = run_command(missing_verb, site)
        assert run.code != 0
        assert _HIGH_COUNT.search(run.output) is None
