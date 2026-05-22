# SWE-bench Grader Audit — AOrchestra

**Date:** 2026-05-21
**Trigger:** GitHub issue [FoundationAgents/AOrchestra#6](https://github.com/FoundationAgents/AOrchestra/issues/6) — *"Question about SWE-bench grader"* (by @NutsCracker1, 2026-05-20)
**Audited file:** `benchmark/swebench/swebench_executor.py` (public repo `FoundationAgents/AOrchestra`, also verified identical on `didiforgithub/FoundationAgent-Dev@feature/merge_claude_subagents`)
**Reference:** [`swe-bench/SWE-bench`](https://github.com/swe-bench/SWE-bench) `main` branch — specifically `swebench/harness/log_parsers/python.py`, `swebench/harness/grading.py`, `swebench/harness/constants/__init__.py`, `swebench/harness/constants/python.py`

---

## 0. TL;DR

The issue reporter is correct: AOrchestra re-implements log parsing and grading by hand rather than calling `swebench.harness`. The reimplementation is **not behavior-equivalent** to the official harness. Three independent bugs interact such that, for any non-Django instance whose `pytest` invocation reaches the test-run phase and prints a trailing `"N passed"` summary, the grader will report `resolved=True` regardless of whether the gold `FAIL_TO_PASS` tests actually passed. Django parsing is closer to correct but still affected by a brittle global-OK fallback.

If the reported SWE-bench numbers were produced by this code path (not by re-running predictions through the official harness), they are likely **systematically inflated**.

---

## 1. Issue text (verbatim)

> Hi, and thanks for releasing AOrchestra!
>
> My question: Is there a reason the harness re-implements log parsing/grading by hand rather than using the official [SWE-bench](https://github.com/swe-bench/SWE-bench) swebench.harness (`get_eval_report` + the per-repo log parsers in `MAP_REPO_TO_PARSER`)? Was the custom grader a deliberate choice (e.g. to avoid a dependency or Docker-in-Docker), and were the reported numbers produced with this code path or with the official harness?
>
> Thanks!

---

## 2. What AOrchestra actually does

File: `benchmark/swebench/swebench_executor.py` (624 lines). The header comment reads `"based on official swebench harness implementation"` and individual functions cite the official source files they were ported from. Concretely:

| Concern | Official harness | AOrchestra |
|---|---|---|
| Per-repo log parser registry | `MAP_REPO_TO_PARSER_PY` in `swebench/harness/log_parsers/python.py` (18 Python repos, 6 distinct parsers) | Inline `if repo == "django/django"` else default to one pytest parser |
| Pytest parser | `parse_log_pytest` / `parse_log_pytest_options` / `parse_log_pytest_v2` / `parse_log_seaborn` / `parse_log_sympy` / `parse_log_matplotlib` | One function, `parse_log_pytest` (lines 152-169) |
| Django parser | `parse_log_django` (multi-line stateful, handles 3 brittle multi-line patterns, FAIL/ERROR prefixes, `XFAIL`, etc.) | One regex `^(test_\w+)\s+\(([^)]+)\)\s+\.\.\.\s+(ok\|FAIL\|ERROR\|skipped)` (lines 172-189) |
| Per-(repo, version) test command | `MAP_REPO_VERSION_TO_SPECS` in `swebench/harness/constants/python.py` (per-version `test_cmd`) | Flat dict `REPO_TEST_CMDS` keyed only by repo (lines 27-40) |
| Grading | `get_logs_eval` + `get_eval_tests_report` + `get_resolution_status` | `get_eval_tests_report` (lines 192-274) |
| `resolved` definition | `ResolvedStatus.FULL` (F2P==1 ∧ P2P==1); also tracks `PARTIAL` | `all_f2p_pass and all_p2p_pass` (line 552) |
| Pre-test-output error detection | Checks `APPLY_PATCH_FAIL` / `RESET_FAILED` / `TESTS_ERROR` / `TESTS_TIMEOUT` markers and short-circuits | None |
| Missing test marker fallback | Returns `found=False` (then `patch_successfully_applied=False`); only falls back to full-log parse when status_map is empty | Falls back to parsing the entire `test_output` whenever markers are absent (lines 206-209) |
| Docker | Reuses official `swebench/sweb.eval.x86_64.*` images via `docker run` | Same |

---

## 3. Verbatim side-by-side: the three load-bearing bugs

### Bug 3.1 — Pytest regex direction is reversed

**Official** (`swebench/harness/log_parsers/python.py` lines 7-26):

```python
def parse_log_pytest(log: str, test_spec: TestSpec) -> dict[str, str]:
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map
```

`TestStatus` enum values: `FAILED, PASSED, SKIPPED, ERROR, XFAIL` (`constants/__init__.py:52-58`). The official parser expects lines that **begin with a status token**, e.g.:

```
PASSED tests/foo.py::test_bar
FAILED tests/foo.py::test_baz - AssertionError: x != y
```

`test_case = line.split()` → `["PASSED", "tests/foo.py::test_bar"]` → `test_status_map["tests/foo.py::test_bar"] = "PASSED"`. This matches the actual pytest `-rA` short-summary format that the harness configures.

**AOrchestra** (`benchmark/swebench/swebench_executor.py` lines 152-169):

```python
def parse_log_pytest(log: str) -> Dict[str, str]:
    test_status = {}
    pattern = r"^(.*?)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)"
    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            test_name = match.group(1).strip()
            status = match.group(2)
            test_status[test_name] = status
    return test_status
```

This pattern expects the test name **first**, status **second** — the opposite of what pytest emits. Against a real pytest log such as `"PASSED tests/foo.py::test_bar"`, `^(.*?)\s+(PASSED|...)` matches `group(1)="PASSED"` (yes, the regex *will* match by treating `"PASSED"` as the test name and looking for any of the status words after it — but there is no second status word, so the match fails. Even if it matched, the assignment `test_status["PASSED"] = "<something>"` would be wrong.)

In practice, against `pytest -rA` output, `parse_log_pytest` returns an **empty dict** for the vast majority of lines. Any per-test verdicts that pytest prints inline like `"tests/foo.py::test_bar PASSED [12%]"` (an older format) would be parsed, but the official harness is configured to emit the short-summary format, where this does not occur. Note that AOrchestra also adds `--no-header -rA --tb=no -p no:cacheprovider` to the pytest command, which produces the `STATUS test_id` format — exactly the format this parser cannot read.

**Verdict:** silent total parse failure for nearly every non-Django instance.

### Bug 3.2 — "Global OK / N passed" fallback in `test_passed`

`benchmark/swebench/swebench_executor.py` lines 223-244:

```python
def test_passed(test_name: str) -> bool:
    if test_name in test_status:
        return test_status[test_name] == "PASSED"

    for key, status in test_status.items():
        if test_name in key or key in test_name:
            return status == "PASSED"
        test_method = test_name.split("::")[-1] if "::" in test_name else test_name.split(".")[-1]
        if test_method in key:
            return status == "PASSED"

    # Check for overall pass (e.g., "OK" at end for Django, or "X passed" for pytest)
    if "OK" in test_log.split("\n")[-10:]:
        return True
    if re.search(r"\d+ passed", test_log):
        return True

    return False
```

Two independent problems:

1. **`re.search(r"\d+ passed", test_log)`** — pytest's final summary line is e.g. `"===== 3 passed, 1 failed, 2 errors in 0.42s ====="`. The substring `"3 passed"` will match this regex even when the run is **a net failure**. As long as *any* test passes (or the summary line says `"0 passed, 0 failed, 0 skipped"`... actually `"0 passed"` matches `\d+ passed` too), this returns True.

2. **`"OK" in test_log.split("\n")[-10:]`** — this is a `list.__contains__` check, asking whether the literal string `"OK"` is *itself* one of the last 10 lines. That is rare in real output (Django's success summary is `"OK"` on its own line — but Django's "errors" form is `"FAILED (errors=N)"` which does *not* contain a bare `"OK"`). But if interpreted as `"OK" in <some line>` (which is what happens when the read on this line is `"OK" in test_log.split(...)`)... let me re-check.

   Re-reading: `test_log.split("\n")[-10:]` is a `list[str]`. `"OK" in <list>` is True iff one element of the list equals `"OK"` exactly. So this only fires when one of the last 10 lines is the bare string `"OK"`. That happens in Django successful runs. For pytest it almost never fires. **Correction to my earlier informal claim**: this is not as catastrophic as a substring search; it is "merely" an unjustified global pass for any run whose last ten lines happen to include a bare `"OK"`. Still wrong, but narrower than `"OK" in line` would have been.

**Net effect of 3.1 + 3.2 on a non-Django pytest run:** `test_status` is empty → the dict lookup fails → the substring matches in the for-loop never trigger → falls through to the fallbacks → `"\d+ passed"` regex hits → returns True. Every F2P case is marked `success`.

The official `test_passed` (`swebench/harness/grading.py` lines 27-28) has no such fallbacks:

```python
def test_passed(case: str, sm: dict[str, str]) -> bool:
    return case in sm and sm[case] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]
```

A test name not in `sm` is **not** a pass. (Note `XFAIL` counts as pass officially — AOrchestra omits this entirely.)

### Bug 3.3 — `PASS_TO_PASS` defaults to success on miss

`benchmark/swebench/swebench_executor.py` lines 267-272:

```python
# Classify PASS_TO_PASS tests
for test in pass_to_pass or []:
    if test_failed(test):
        results["PASS_TO_PASS"]["failure"].append(test)
    else:
        results["PASS_TO_PASS"]["success"].append(test)
```

`test_failed` (lines 246-258) requires the test name to be present in `test_status` or in a fuzzy match. When `test_status` is empty (which it is, per 3.1), `test_failed` returns False. So **every P2P test is marked success by default**.

The official harness instead uses `check_pass_and_fail` (`swebench/harness/grading.py` lines 123-128):

```python
def check_pass_and_fail(test_case, eval_status_map, success, failed):
    if test_passed(test_case, eval_status_map):
        success.append(test_case)
    elif test_failed(test_case, eval_status_map):
        failed.append(test_case)
```

A test that is neither demonstrably passed nor demonstrably failed is appended to **neither** list — and since `compute_pass_to_pass` divides successes by `success+failure`, missing tests are excluded. Note: this means a malicious patch that simply disables P2P tests scores P2P=1 officially too; AOrchestra inherits this property in its `default-success` form, but for entirely different reasons.

### Net behavior under combined bugs

For a typical non-Django SWE-bench instance:
1. Agent's patch is applied; eval script runs `pytest ... --tb=no -p no:cacheprovider`.
2. Pytest exits, prints `"===== K passed, L failed in T s ====="` at the end.
3. AOrchestra parses log → `test_status = {}` (Bug 3.1).
4. For each F2P test → not in dict → no fuzzy hit → fallback `\d+ passed` matches → marked success (Bug 3.2).
5. For each P2P test → `test_failed` returns False → marked success (Bug 3.3).
6. `all_f2p_pass and all_p2p_pass` = True → `resolved=True`, `reward=1.0`.

The only paths where this does *not* produce a false positive are:
- The pytest invocation fails before any test runs (import error, missing dependency, etc.), so no line containing `\d+ passed` appears.
- The eval script crashes before the test-output markers, but AOrchestra's marker-absent fallback (lines 206-209) re-parses the full log, so this protection is weakened.

---

## 4. Other substantive deviations (less load-bearing but worth noting)

### 4.1 Six parsers collapsed to two

Official `MAP_REPO_TO_PARSER_PY` (lines 270-289 of `python.py`):

| Parser | Repos |
|---|---|
| `parse_log_pytest` | `pallets/flask`, `pytest-dev/pytest`, `pydata/xarray`, `marshmallow-code/marshmallow`, `pylint-dev/astroid`, `pvlib/pvlib-python`, `pyvista/pyvista`, `sqlfluff/sqlfluff` |
| `parse_log_pytest_options` | `psf/requests`, `pylint-dev/pylint`, `pydicom/pydicom` |
| `parse_log_pytest_v2` | `astropy/astropy`, `scikit-learn/scikit-learn`, `sphinx-doc/sphinx` |
| `parse_log_django` | `django/django` |
| `parse_log_seaborn` | `mwaskom/seaborn` |
| `parse_log_sympy` | `sympy/sympy` |
| `parse_log_matplotlib` | `matplotlib/matplotlib` |

AOrchestra uses one pytest parser (broken per 3.1) and one Django parser. Even if Bug 3.1 were fixed, the following would still be wrong:

- **sympy** uses `bin/test` which prints `test_xxx ok` / `test_xxx F` / `test_xxx E`, *not* `PASSED`/`FAILED`. The official `parse_log_sympy` matches these forms; AOrchestra's pytest parser cannot.
- **seaborn** has an idiosyncratic `<test> PASSED <something>` format handled by `parse_log_seaborn`.
- **astropy / scikit-learn / sphinx** use `parse_log_pytest_v2`, which strips ANSI escape sequences (`re.sub(r"\[(\d+)m", "", line)`) and C0 control chars, and also handles the *older* pytest format where status appears at the **end** of the line. AOrchestra strips neither, so colorized output is unparseable.
- **requests / pylint / pydicom** use `parse_log_pytest_options`, which normalizes parametrized test names like `test_foo[arg]` (collapsing long path-like params). Without it, parametrized test IDs in gold `FAIL_TO_PASS` lists won't match parsed keys.
- **matplotlib** uses `parse_log_matplotlib`, which replaces `MouseButton.LEFT/RIGHT` with `1`/`3` to match gold test IDs.

### 4.2 Django parser is stricter than official

Official `parse_log_django` (`python.py:64-141`) handles:
- `... ok` / `... OK` / `...  OK` (note double-space) suffixes
- `FAIL:` and `ERROR:` line prefixes (from Django's failure-summary section)
- `... skipped` (any tail)
- Multi-line cases where the test header and the verdict are separated by output (uses `prev_test` state across lines)
- Three brittle regex patches for "Testing against Django installed in ..." / "Internal Server Error: /..." / "System check identified no issues..." interleaving

AOrchestra's single-line regex `^(test_\w+)\s+\(([^)]+)\)\s+\.\.\.\s+(ok|FAIL|ERROR|skipped)` misses:
- Tests with names not starting with `test_` (unusual but allowed)
- The `FAIL:` / `ERROR:` summary-section format
- Tests whose verdict is on a later line (multi-line case)
- `OK` (capitalized) tail, `XFAIL`

### 4.3 Per-(repo, version) test commands collapsed

Official `MAP_REPO_VERSION_TO_SPECS` (`constants/python.py`) assigns a `test_cmd` per (repo, version). Examples:

- `TEST_DJANGO = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"`; Django version `1.9` overrides to `TEST_DJANGO_NO_PARALLEL` (`SPECS_DJANGO["1.9"]["test_cmd"]`).
- `TEST_ASTROPY_PYTEST = "pytest -rA -vv -o console_output_style=classic --tb=no"`.
- `TEST_SPHINX = "tox --current-env -epy39 -v --"` — uses tox, not pytest directly.
- `TEST_SYMPY = "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose"`.
- `TEST_PYTEST = "pytest -rA"` (after the second assignment at module level overrides the first) — *no* `--no-header --tb=no -p no:cacheprovider`.

AOrchestra `REPO_TEST_CMDS` (`swebench_executor.py:27-40`):
- All non-Django, non-sympy entries use `"python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider"`.
- `django/django`: `"./tests/runtests.py --verbosity 2 {tests}"` — **missing `--settings=test_sqlite` and `--parallel 1`**. Without `--settings=test_sqlite`, Django uses its default settings, which require a configured database; depending on the image, this may run different tests or fail entirely. (The official `swebench/sweb.eval.x86_64.*` images are built expecting `test_sqlite`.)
- `sphinx-doc/sphinx`: uses pytest, **not** tox. The official image's sphinx env is configured for `tox --current-env -epy39`, so pytest-direct invocation may pick up the wrong configuration.
- `astropy/astropy`: missing `-vv` and `-o console_output_style=classic`, which `parse_log_pytest_v2` was built around.

Repos in the Verified subset but **not** in `REPO_TEST_CMDS` (will fall through to `DEFAULT_TEST_CMD = "python -m pytest ..."`):
- `marshmallow-code/marshmallow`, `pvlib/pvlib-python`, `pydicom/pydicom`, `pylint-dev/astroid`, `pyvista/pyvista`, `sqlfluff/sqlfluff`

### 4.4 Fuzzy name matching in `test_passed` / `test_failed`

`swebench_executor.py:230-236`:

```python
for key, status in test_status.items():
    if test_name in key or key in test_name:
        return status == "PASSED"
    test_method = test_name.split("::")[-1] if "::" in test_name else test_name.split(".")[-1]
    if test_method in key:
        return status == "PASSED"
```

Substring-based matching is unsafe: `test_foo` is a substring of `test_foo_bar`, `test_foo_baz`, etc. A test that genuinely should be looked up by exact ID can get its verdict from an unrelated test that happens to have an overlapping name. The official harness does **exact dict lookup only**.

### 4.5 `resolved` definition

- Official: `get_resolution_status` returns `FULL` (F2P=1 ∧ P2P=1), `PARTIAL` (0<F2P<1 ∧ P2P=1), or `NO`. Only `FULL` counts toward the resolved rate.
- AOrchestra: `resolved = all_f2p_pass and all_p2p_pass`, where `all_f2p_pass = (success_count == total)`.

These two definitions are **equivalent in conclusion** for binary resolved/not-resolved reporting (both require F2P=1 and P2P=1). AOrchestra loses the `PARTIAL` signal but does not over-count in this dimension on its own.

### 4.6 No detection of upstream test-harness failure

Official `get_logs_eval` (`grading.py:39-91`) first checks for `APPLY_PATCH_FAIL`, `RESET_FAILED`, `TESTS_ERROR`, `TESTS_TIMEOUT` markers and returns `found=False` if any are present, which then sets `patch_successfully_applied=False` in the report. AOrchestra has no such check — a timeout or apply-patch failure will still go through the parser, and (given Bugs 3.1/3.2/3.3) may still produce `resolved=True` if any `"N passed"` substring is present anywhere in the merged stdout/stderr.

### 4.7 Marker-absent fallback parses the full log

Official (`grading.py:74-89`): if `START_TEST_OUTPUT`/`END_TEST_OUTPUT` markers are not both present, returns `found=False` immediately. The full-log fallback only kicks in when markers were present but `status_map` came back empty.

AOrchestra (`swebench_executor.py:202-209`): when markers are missing, parses the **entire `test_output`** as the test log. This means setup/install noise, conda activation messages, and `git status` output get fed to the parser, increasing both false positives (from accidental `passed`/`OK` substrings) and false negatives (from setup-time `FAILED` lines).

---

## 5. Reproducibility — what to verify on dev infra

To convert "this grader is broken" from analytical claim into measured impact, run any of:

1. **Construct a synthetic log** that contains only a pytest summary line like `===== 5 passed, 3 failed in 0.4s =====` between the `>>>>> Start Test Output` / `>>>>> End Test Output` markers, and feed it to `get_eval_tests_report` with non-empty `FAIL_TO_PASS`. Expected official result: F2P all failure, `resolved=False`. AOrchestra result: F2P all success, P2P all success, `resolved=True`.

2. **Pick 10-20 instances** from a real AOrchestra run's saved `test_output.txt` logs, re-grade them by feeding the **same logs** to the official `swebench.harness.grading.get_eval_report`, and compare the binary `resolved` flag. The delta is the upper bound on grader-induced inflation in the headline number.

3. **Re-run predictions through the official harness**: take the `predictions.jsonl` (or equivalent) saved per instance and invoke `python -m swebench.harness.run_evaluation --predictions_path ... --run_id audit`. This is the gold-standard reproducibility check and is what would need to back any public number in the paper or README.

## 6. Recommendations

Ranked by effort/impact:

1. **Lowest-effort, highest-impact:** replace `parse_log_pytest`, `parse_log_django`, and `get_eval_tests_report` with calls to `swebench.harness.log_parsers.MAP_REPO_TO_PARSER` and `swebench.harness.grading.get_eval_report`. Adding `swebench` as a dependency is cheap — it's a pure-Python package and the docker images are already being used. This removes Bugs 3.1, 3.2, 3.3, and section 4.1/4.2 in one move.
2. **Replace `REPO_TEST_CMDS` with the official `MAP_REPO_VERSION_TO_SPECS` lookup**, keyed by `(repo, version)`. Section 4.3.
3. **Add the upstream-failure short-circuit** from `get_logs_eval` (4.6).
4. **Tighten the marker-absent fallback** to match official semantics (4.7).
5. **Publish a comparison table** in the README showing AOrchestra's reported number against an official-harness re-run on the same predictions, on at least the Verified subset.

If keeping the reimplementation is desired (e.g., to avoid the dependency), the minimum fix set is:
- Bug 3.1: swap the regex to `^(PASSED|FAILED|SKIPPED|ERROR|XFAIL)\s+(.+?)(?:\s*-\s*.*)?$` and extract name from group 2; or just port the official `line.split()` approach verbatim.
- Bug 3.2: delete the two fallback `if` statements at lines 238-242. A test not in `test_status` is *not* a pass.
- Bug 3.3: change the P2P branch so a test absent from `test_status` is neither success nor failure (matches official `check_pass_and_fail` semantics) — or, more conservatively for SWE-bench's "maintenance" interpretation, mark it as failure. The current default-to-success is the worst option.
- 4.4: drop the substring fuzzy match. Use exact dict lookup.

## 7. Answering the issue reporter

The reporter asks two things:

1. **Was the custom grader deliberate?** Code-archaeologically: yes — the file is a deliberate port (every function references its official source file in a comment), and the goal appears to have been avoiding the `swebench` PyPI dependency. The docker images are still the official `swebench/sweb.eval.x86_64.*` set, so it isn't about avoiding Docker-in-Docker. It's a dependency-elimination port.

2. **Were the reported numbers produced with this code path?** This audit cannot answer that — the README/paper does not state it. Given Bugs 3.1/3.2/3.3, this is the question the reply must take seriously. Either:
   - Numbers were produced by this code path → they need re-running through the official harness before being relied on.
   - Numbers were produced by re-running predictions through the official harness → README/paper should say so explicitly, and the in-tree grader should be either fixed or labeled as "for local CI only, not for reported metrics."

A reply to the issue should acknowledge the deliberate-port intent, confirm the divergences flagged in this document, and commit to either fixing the grader or clarifying which harness produced each reported number.

---

## Appendix A — Files inspected

| File | SHA / commit | Notes |
|---|---|---|
| `benchmark/swebench/swebench_executor.py` | local @ `23f0e8b` (HEAD of `FoundationAgents/AOrchestra` at audit time) | Identical content confirmed against user-supplied copy from `didiforgithub/FoundationAgent-Dev@feature/merge_claude_subagents` |
| `swebench/harness/log_parsers/python.py` | `swe-bench/SWE-bench@main` (fetched 2026-05-21) | `/tmp/swebench_python_parsers.py` |
| `swebench/harness/grading.py` | `swe-bench/SWE-bench@main` (fetched 2026-05-21) | `/tmp/swebench_grading.py` |
| `swebench/harness/constants/__init__.py` | `swe-bench/SWE-bench@main` (fetched 2026-05-21) | `TestStatus`, `ResolvedStatus`, `FAIL_ONLY_REPOS` |
| `swebench/harness/constants/python.py` | `swe-bench/SWE-bench@main` (fetched 2026-05-21) | `MAP_REPO_VERSION_TO_SPECS_PY`, `TEST_*` constants |

## Appendix B — Line references (AOrchestra)

- `REPO_TEST_CMDS` definition: `swebench_executor.py:27-40`
- `make_eval_script`: `swebench_executor.py:79-149`
- `parse_log_pytest`: `swebench_executor.py:152-169` ← **Bug 3.1**
- `parse_log_django`: `swebench_executor.py:172-189`
- `get_eval_tests_report`: `swebench_executor.py:192-274`
  - Marker-absent fallback: lines 202-209
  - `test_passed` with fallbacks: lines 223-244 ← **Bug 3.2**
  - Fuzzy substring matching: lines 230-236 ← section 4.4
  - P2P default-success: lines 267-272 ← **Bug 3.3**
- `resolved` computation: `swebench_executor.py:545-553`

---

## 8. Status: Fixed (2026-05-21)

The in-tree grader has been removed in favour of direct delegation to the official `swebench` package. The bugs in §3 and the deviations in §4.1, §4.2, §4.3, §4.4, §4.6 are resolved by this rewrite. §4.5 (`PARTIAL` vs `FULL`) and §4.7 (marker-absent fallback) are addressed below.

### 8.1 Changes landed

**`requirements.txt`** — added `swebench>=4.1.0` (already present in the `orchestra` conda env at audit time; no install needed for existing setups, but new bootstraps will pick it up via `pip install -r requirements.txt`).

**`benchmark/swebench/swebench_executor.py`** — 624 → 484 lines. Removed `parse_log_pytest`, `parse_log_django`, `get_eval_tests_report`, `make_eval_script`, `REPO_TEST_CMDS`, `DEFAULT_TEST_CMD`, `NON_TEST_EXTS`, `get_modified_files`, `get_test_directives`, and the local `START_TEST_OUTPUT` / `END_TEST_OUTPUT` constants. Added three module-level helpers that delegate to upstream:

| Helper | Delegates to | Purpose |
|---|---|---|
| `_instance_to_dict(instance)` | — | Convert `SWEBenchInstance` dataclass to the dict shape upstream expects |
| `_grade_test_output(test_output, instance)` | `MAP_REPO_TO_PARSER[repo]`, `swebench.harness.grading.test_passed` / `test_failed`, `EvalType.FAIL_ONLY` via `FAIL_ONLY_REPOS` | Parse the captured eval-script log and classify F2P / P2P verdicts |
| `_build_eval_script(instance, repo_directory)` | `swebench.harness.test_spec.python.make_eval_script_list_py`, `MAP_REPO_VERSION_TO_SPECS[repo][version]` | Build the bash eval script with the correct per-(repo, version) test command |

`run_tests()` was rewritten to use these helpers. The external return shape `(reward, results)` is preserved, including `results["fail_to_pass"]["passed" / "failed"]`, `results["pass_to_pass"]["passed" / "failed"]`, `results["reward"]`, and `results["summary"]`. The `summary` denominator for P2P now counts only tests with a recorded verdict (matches upstream `compute_pass_to_pass`); F2P denominator stays at the gold list size and a dropped test counts as failure (the patch did not demonstrate the bug was fixed).

**§4.5 — `resolved` definition.** Still computed locally as `all_f2p_pass and all_p2p_pass`, which matches upstream `ResolvedStatus.FULL` semantics for the binary resolved/not-resolved flag downstream consumers expect. The `PARTIAL` signal is not surfaced; if needed later, call `swebench.harness.grading.get_resolution_status` directly.

**§4.7 — marker-absent fallback.** Now matches upstream: if `START_TEST_OUTPUT` / `END_TEST_OUTPUT` markers are not both present in the log, the parser is fed an empty string (rather than the full log), which yields an empty `status_map`. Every gold F2P test then lands in failure, and gold P2P tests fall out of both lists (dropped per upstream `check_pass_and_fail`).

### 8.2 Bonus: forced-submit reward bug in the runner

`aorchestra/runners/swebench_runner.py:241` was treating `executor.run_tests()`'s tuple return value `(reward, results)` as a scalar:

```python
reward = await executor.run_tests()
total_reward = float(reward if isinstance(reward, (int, float)) else 0.0)
```

The tuple is never an `int`/`float`, so `total_reward` was always `0.0` whenever the orchestrator hit max attempts without an explicit submit and the forced-submit branch fired. Fixed by unpacking the tuple:

```python
reward, _ = await executor.run_tests()
total_reward = float(reward)
```

This is independent of the grader bugs but compounds the inflation/deflation discussion: pre-fix, any run that exhausted attempts silently scored 0 even if its tests passed.

### 8.3 Smoke-import in setup script

`scripts/setup_env.sh` step 5 now imports `swebench` along with the other required packages, so a future env bootstrap that fails to install it surfaces the breakage at setup time rather than at first eval.

### 8.4 Validation

All checks executed against `~/miniconda3/envs/orchestra/bin/python` (Python 3.13.13, `swebench` 4.1.0):

1. **Import smoke test**: `from benchmark.swebench.swebench_executor import SWEBenchExecutor, _grade_test_output, _build_eval_script, _instance_to_dict` succeeded.
2. **Upstream registry sanity**: `MAP_REPO_TO_PARSER` has 64 entries (covers Verified subset and beyond); `django/django` → `parse_log_django`, `astropy/astropy` → `parse_log_pytest_v2`, `sphinx-doc/sphinx` → `parse_log_pytest_v2`.
3. **Test-spec sanity**: `MAP_REPO_VERSION_TO_SPECS["django/django"]["4.0"]["test_cmd"]` is `./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1` (vs. the buggy old `./tests/runtests.py --verbosity 2`); `MAP_REPO_VERSION_TO_SPECS["sphinx-doc/sphinx"]["7.2"]["test_cmd"]` is `tox --current-env -epy39 -v --` (vs. the buggy old pytest direct invocation).
4. **Synthetic log grading via `_grade_test_output`**:

   | # | Scenario | Expected | Got |
   |---|---|---|---|
   | 1 | pytest log shows F2P actually `FAILED tests/foo.py::test_bar`, summary line `===== 3 passed, 1 failed =====` between markers | F2P failure, `resolved=False` (this was the canonical bug — old grader returned True) | ✅ failure, False |
   | 2 | pytest log shows F2P actually `PASSED tests/foo.py::test_bar` between markers | F2P success | ✅ success |
   | 3 | Markers entirely absent (harness crash) | All F2P → failure, P2P → dropped | ✅ |
   | 4 | Django log shows F2P actually `FAIL: test_bar (foo.tests.FooTest)` between markers | F2P failure | ✅ |

5. **Real eval-script generation**: `_build_eval_script` on a synthetic `django/django==4.0` instance produced the expected setup (`sed -i ... locale-gen`, `export LANG=en_US.UTF-8`, `python -m pip install -e .`) and the correct test command `./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 foo.tests`.

### 8.5 Remaining work / caveats

- **README / paper claim.** Section 7 question 2 (which harness produced the reported numbers) is unchanged by this fix — it's a documentation question. Whoever drafts the issue #6 reply should still address it.
- **Patch-apply short-circuit.** The fix relies on the eval script itself emitting `>>>>> Patch Apply Failed` etc. — `make_eval_script_list_py` includes those branches. Upstream's `get_logs_eval` then surfaces them via `found=False`. Our path treats marker-missing as "every F2P failed", which is the same conclusion (`resolved=False`) but loses the distinction in logs. If finer-grained reporting is wanted later, call `get_logs_eval` directly and surface its `(found, patch_successfully_applied)` flags.
- **Non-Python languages.** `_build_eval_script` only calls `make_eval_script_list_py`. SWE-bench has a JS branch (`make_eval_script_list_js`); if AOrchestra ever runs Verified-JS or the multi-language subset, swap to `make_eval_script_list` in `swebench/harness/test_spec/create_scripts.py` (the language-dispatching wrapper). Out of scope until that's actually wanted.
