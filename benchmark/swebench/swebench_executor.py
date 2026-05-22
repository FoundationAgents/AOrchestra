"""SWE-bench executor.

Grading and eval-script construction delegate to the official ``swebench``
package (``swebench.harness.{grading, log_parsers, constants, test_spec}``).
The in-tree grader that previously lived here was inconsistent with the
official harness (see swebench_grader_audit.md); calling upstream directly
ensures the per-(repo, version) test commands, the per-repo log parsers, and
the FAIL_TO_PASS / PASS_TO_PASS classification all match the published
benchmark.
"""
import asyncio
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
    EvalType,
    TestStatus,
)
from swebench.harness.grading import test_failed as _official_test_failed
from swebench.harness.grading import test_passed as _official_test_passed
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import make_eval_script_list_py

from base.engine.logs import logger
from benchmark.swebench.data_loader import SWEBenchInstance


def _instance_to_dict(instance: SWEBenchInstance) -> Dict[str, Any]:
    """Convert SWEBenchInstance dataclass to the dict shape upstream expects."""
    data = asdict(instance)
    data["FAIL_TO_PASS"] = list(instance.FAIL_TO_PASS or [])
    data["PASS_TO_PASS"] = list(instance.PASS_TO_PASS or [])
    return data


def _grade_test_output(
    test_output: str,
    instance: SWEBenchInstance,
) -> Dict[str, Dict[str, list]]:
    """Parse a captured eval-script log and classify F2P / P2P verdicts.

    Mirrors the official ``grading.get_logs_eval`` + ``get_eval_tests_report``
    pipeline but operates on an in-memory string (we already have the log) and
    skips the file-IO indirection.
    """
    repo = instance.repo
    parser = MAP_REPO_TO_PARSER.get(repo)
    if parser is None:
        raise KeyError(
            f"No upstream log parser registered for repo {repo!r}; "
            f"is the instance from a newer swebench split than the installed package?"
        )

    if START_TEST_OUTPUT in test_output and END_TEST_OUTPUT in test_output:
        test_content = test_output.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    else:
        # Markers missing → patch apply / reset / harness failure. Upstream treats
        # this as "patch did not apply"; we return empty status_map so every
        # FAIL_TO_PASS test is marked failure (and PASS_TO_PASS is excluded).
        test_content = ""

    status_map = parser(test_content, None) if test_content else {}

    eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL

    def _check(test_case: str, success: list, failed: list) -> None:
        if eval_type == EvalType.FAIL_ONLY:
            if (
                test_case in status_map
                and status_map[test_case] == TestStatus.FAILED.value
            ):
                failed.append(test_case)
            else:
                success.append(test_case)
        else:
            if _official_test_passed(test_case, status_map):
                success.append(test_case)
            elif _official_test_failed(test_case, status_map):
                failed.append(test_case)
            # Tests that are neither demonstrably passed nor failed are dropped,
            # matching upstream's ``check_pass_and_fail``.

    f2p_success: list = []
    f2p_failure: list = []
    for case in instance.FAIL_TO_PASS or []:
        _check(case, f2p_success, f2p_failure)

    p2p_success: list = []
    p2p_failure: list = []
    for case in instance.PASS_TO_PASS or []:
        _check(case, p2p_success, p2p_failure)

    return {
        "FAIL_TO_PASS": {"success": f2p_success, "failure": f2p_failure},
        "PASS_TO_PASS": {"success": p2p_success, "failure": p2p_failure},
    }


def _build_eval_script(instance: SWEBenchInstance, repo_directory: str) -> str:
    """Build the bash eval script via upstream ``make_eval_script_list_py``.

    Upstream's script activates conda, applies the test patch, runs the
    per-(repo, version) ``test_cmd`` from ``MAP_REPO_VERSION_TO_SPECS``,
    wraps the test output between the ``START_TEST_OUTPUT`` / ``END_TEST_OUTPUT``
    sentinels, then reverts the test files.
    """
    instance_dict = _instance_to_dict(instance)
    try:
        specs = MAP_REPO_VERSION_TO_SPECS[instance.repo][instance.version]
    except KeyError as e:
        raise KeyError(
            f"No specs for ({instance.repo!r}, version={instance.version!r}) in "
            f"upstream MAP_REPO_VERSION_TO_SPECS"
        ) from e

    env_name = "testbed"
    eval_commands = make_eval_script_list_py(
        instance_dict,
        specs,
        env_name,
        repo_directory,
        instance.base_commit,
        instance.test_patch or "",
    )
    return "\n".join(["#!/bin/bash", "set -uxo pipefail", *eval_commands]) + "\n"


class SWEBenchExecutor:
    """Executes SWE-bench tasks using Docker containers."""

    def __init__(
        self,
        instance: SWEBenchInstance,
        logs_dir: Path,
        timeout: int = 1800,
        env_init: Optional[Dict[str, str]] = None,
    ):
        self.instance = instance
        self.logs_dir = logs_dir
        self.timeout = timeout
        self.env_init = env_init or {}
        
        self.container_id: Optional[str] = None
        self._temp_dir: Optional[Path] = None
        self._repo_path: Optional[str] = None  # Linux path in container, use str not Path
        
        # Create logs directory
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def start_container(self):
        """Start Docker container for the SWE-bench instance."""
        # Create temporary directory for workspace
        self._temp_dir = Path(tempfile.mkdtemp(prefix=f"swebench_{self.instance.instance_id}_"))
        
        # Determine image name based on instance_id
        # SWE-bench official images use format: swebench/sweb.eval.x86_64.{owner}_1776_{owner}-{issue}
        # Example: astropy__astropy-12907 -> swebench/sweb.eval.x86_64.astropy_1776_astropy-12907
        # Parse instance_id: "astropy__astropy-12907" -> owner="astropy", issue="12907"
        parts = self.instance.instance_id.split("__")
        owner = parts[0]  # "astropy"
        repo_issue = parts[1] if len(parts) > 1 else self.instance.instance_id  # "astropy-12907"
        image_name = f"swebench/sweb.eval.x86_64.{owner}_1776_{repo_issue}"
        
        logger.info(f"Starting container for {self.instance.instance_id}")
        logger.info(f"Image: {image_name}")
        
        try:
            # Check if image exists locally
            check_result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "images", "-q", image_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if not check_result.stdout.strip():
                # Pull image
                logger.info(f"Pulling image: {image_name}")
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "pull", image_name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to pull image: {result.stderr}")
            
            # Remove any existing container with the same name (cleanup from previous runs)
            container_name = f"swebench_{self.instance.instance_id}"
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=30,
            )
            
            # Run container
            env_args = []
            for key, value in self.env_init.items():
                env_args.extend(["-e", f"{key}={value}"])
            
            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "-v", f"{self._temp_dir}:/workspace",
                *env_args,
                image_name,
                "tail", "-f", "/dev/null",  # Keep container running
            ]
            
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container: {result.stderr}")
            
            self.container_id = result.stdout.strip()
            logger.info(f"Container started: {self.container_id[:12]}")
            
            # Setup repository in container
            await self._setup_repo()
            
        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Failed to start container: {e}") from e

    async def _setup_repo(self):
        """Setup repository at base commit in container."""
        if not self.container_id:
            raise RuntimeError("Container not started")
        
        # SWE-bench official images always place the repository at /testbed
        self._repo_path = "/testbed"
        
        # Checkout base commit
        logger.info(f"Checking out base commit: {self.instance.base_commit}")
        output, exit_code = await self.execute_command(
            f"cd {self._repo_path} && git checkout -f {self.instance.base_commit}"
        )
        if exit_code != 0:
            logger.warning(f"Failed to checkout base commit: {output}")
        
        # Reset any local changes
        await self.execute_command(f"cd {self._repo_path} && git reset --hard HEAD")
        await self.execute_command(f"cd {self._repo_path} && git clean -fd")

    async def execute_command(
        self, 
        command: str, 
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Execute command in container.
        
        Uses stdin to pass command to avoid Windows command line length limit (~8191 chars).
        This allows executing commands with large content (e.g., base64 encoded files).
        """
        if not self.container_id:
            raise RuntimeError("Container not started")

        exec_timeout = timeout if timeout is not None else self.timeout
        
        try:
            # Use -i (interactive) to read command from stdin
            # This bypasses Windows command line length limits
            cmd = ["docker", "exec", "-i"]
            if workdir:
                cmd.extend(["-w", workdir])
            cmd.extend([self.container_id, "bash"])
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            # Pass command through stdin
            stdout, _ = await asyncio.wait_for(
                proc.communicate(input=command.encode('utf-8')),
                timeout=exec_timeout
            )

            output = stdout.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            return output, exit_code

        except asyncio.TimeoutError:
            return "Command timed out", -1
        except Exception as e:
            return f"Error executing command: {e}", -1

    async def apply_patch(self, patch_content: str) -> Tuple[bool, str]:
        """Apply a patch to the repository."""
        if not self.container_id:
            raise RuntimeError("Container not started")
        
        # Write patch to temp file in container
        patch_path = "/tmp/agent_patch.diff"
        
        # Escape patch content for shell
        escaped_patch = patch_content.replace("'", "'\\''")
        output, exit_code = await self.execute_command(
            f"echo '{escaped_patch}' > {patch_path}"
        )
        
        if exit_code != 0:
            return False, f"Failed to write patch: {output}"
        
        # Apply patch
        output, exit_code = await self.execute_command(
            f"cd {self._repo_path} && git apply --check {patch_path}"
        )
        
        if exit_code != 0:
            return False, f"Patch check failed: {output}"
        
        output, exit_code = await self.execute_command(
            f"cd {self._repo_path} && git apply {patch_path}"
        )
        
        if exit_code != 0:
            return False, f"Failed to apply patch: {output}"
        
        return True, "Patch applied successfully"

    async def run_tests(self) -> Tuple[float, Dict[str, Any]]:
        """Run tests and return reward and details.

        Eval script is built by ``swebench.harness.test_spec.python.make_eval_script_list_py``
        and grading is done by ``swebench.harness.grading`` (via ``_grade_test_output``),
        so the verdict matches the official harness.
        """
        if not self.container_id:
            raise RuntimeError("Container not started")

        # Build eval script via upstream
        eval_script = _build_eval_script(self.instance, self._repo_path)

        # Save eval script to log for debugging
        eval_script_log = self.logs_dir / "eval.sh"
        with eval_script_log.open("w", encoding="utf-8") as f:
            f.write(eval_script)

        # Write eval script to container and execute
        await self.execute_command(
            f"cat > /eval.sh << 'EOF_EVAL_SCRIPT'\n{eval_script}\nEOF_EVAL_SCRIPT"
        )
        await self.execute_command("chmod +x /eval.sh")

        # Run eval script with extended timeout for test execution
        test_output, exit_code = await self.execute_command(
            "/bin/bash /eval.sh",
            timeout=self.timeout,
        )

        # Save test output to log
        test_output_log = self.logs_dir / "test_output.txt"
        with test_output_log.open("w", encoding="utf-8") as f:
            f.write(test_output)

        # Grade using upstream parsers + grading semantics
        test_results = _grade_test_output(test_output, self.instance)

        # Build results dict (keep field names stable for downstream consumers)
        results = {
            "fail_to_pass": {
                "passed": test_results["FAIL_TO_PASS"]["success"],
                "failed": test_results["FAIL_TO_PASS"]["failure"],
            },
            "pass_to_pass": {
                "passed": test_results["PASS_TO_PASS"]["success"],
                "failed": test_results["PASS_TO_PASS"]["failure"],
            },
        }

        # Compute resolution (matches upstream ``get_resolution_status`` FULL):
        #   resolved iff every gold F2P test passed AND every gold P2P test passed.
        # When upstream's check_pass_and_fail drops a test (neither passed nor
        # failed in the parsed log), it falls out of both lists. We treat the
        # gold totals as the denominator: a dropped test counts as failure for
        # F2P (the patch did not demonstrate the bug was fixed) and is excluded
        # for P2P (matches upstream's compute_pass_to_pass).
        f2p_total = len(self.instance.FAIL_TO_PASS or [])
        f2p_success = len(results["fail_to_pass"]["passed"])
        p2p_classified = (
            len(results["pass_to_pass"]["passed"])
            + len(results["pass_to_pass"]["failed"])
        )
        p2p_success = len(results["pass_to_pass"]["passed"])

        all_f2p_pass = (f2p_success == f2p_total) if f2p_total > 0 else True
        all_p2p_pass = (p2p_success == p2p_classified) if p2p_classified > 0 else True
        resolved = all_f2p_pass and all_p2p_pass
        reward = 1.0 if resolved else 0.0

        results["reward"] = reward
        # ``pass_to_pass`` denominator follows upstream: only count tests we
        # actually saw a verdict for, not the gold-list size.
        results["summary"] = {
            "fail_to_pass": f"{f2p_success}/{f2p_total}",
            "pass_to_pass": f"{p2p_success}/{p2p_classified}",
        }

        # Save test results to log
        test_log = self.logs_dir / "test_results.log"
        with test_log.open("w", encoding="utf-8") as f:
            f.write(f"Instance: {self.instance.instance_id}\n")
            f.write(f"Resolved: {resolved}\n")
            f.write(f"Reward: {reward}\n")
            f.write(f"FAIL_TO_PASS: {f2p_success}/{f2p_total}\n")
            f.write(f"PASS_TO_PASS: {p2p_success}/{p2p_classified}\n")
            f.write(f"\nDetailed results:\n")
            f.write(f"F2P passed: {results['fail_to_pass']['passed']}\n")
            f.write(f"F2P failed: {results['fail_to_pass']['failed']}\n")
            f.write(f"P2P passed: {results['pass_to_pass']['passed']}\n")
            f.write(f"P2P failed: {results['pass_to_pass']['failed']}\n")

        return reward, results

    async def get_file_content(self, file_path: str) -> Tuple[str, int]:
        """Read file content from container."""
        return await self.execute_command(f"cat {file_path}")

    async def write_file(self, file_path: str, content: str) -> Tuple[bool, str]:
        """Write content to file in container."""
        # Escape content for shell
        escaped_content = content.replace("'", "'\\''")
        output, exit_code = await self.execute_command(
            f"cat > {file_path} << 'EOFMARKER'\n{content}\nEOFMARKER"
        )
        if exit_code != 0:
            return False, f"Failed to write file: {output}"
        return True, "File written successfully"

    async def list_files(self, directory: str = ".") -> Tuple[str, int]:
        """List files in directory."""
        return await self.execute_command(f"find {directory} -type f -name '*.py' | head -100")

    def get_container_id(self) -> Optional[str]:
        """Get the container ID."""
        return self.container_id

    async def cleanup(self):
        """Clean up container and temporary files."""
        if self.container_id:
            try:
                # Stop and remove container
                await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "rm", "-f", self.container_id],
                    capture_output=True,
                    timeout=30,
                )
                logger.info(f"Container removed: {self.container_id[:12]}")
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")
            finally:
                self.container_id = None
        
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temp dir: {e}")
            finally:
                self._temp_dir = None

