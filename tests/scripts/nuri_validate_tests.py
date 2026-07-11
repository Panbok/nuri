import importlib.util
import argparse
import contextlib
import io
import json
import statistics
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "nuri_validate.py"
SPEC = importlib.util.spec_from_file_location("nuri_validate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
nuri_validate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nuri_validate
SPEC.loader.exec_module(nuri_validate)

CTEST_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "_nuri_ctest.py"
CTEST_SPEC = importlib.util.spec_from_file_location("nuri_ctest", CTEST_SCRIPT)
assert CTEST_SPEC is not None and CTEST_SPEC.loader is not None
nuri_ctest = importlib.util.module_from_spec(CTEST_SPEC)
CTEST_SPEC.loader.exec_module(nuri_ctest)

STARTUP_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_tool_startup.py"
STARTUP_SPEC = importlib.util.spec_from_file_location("check_tool_startup", STARTUP_SCRIPT)
assert STARTUP_SPEC is not None and STARTUP_SPEC.loader is not None
check_tool_startup = importlib.util.module_from_spec(STARTUP_SPEC)
STARTUP_SPEC.loader.exec_module(check_tool_startup)


class NuriValidateTests(unittest.TestCase):
    def test_identifier_contract(self):
        self.assertTrue(nuri_validate._valid_identifier("renderer.case_1"))
        self.assertFalse(nuri_validate._valid_identifier("../outside"))
        self.assertFalse(nuri_validate._valid_identifier("con"))
        self.assertFalse(nuri_validate._valid_identifier("Upper.case"))

    def test_affected_plan_is_conservative(self):
        self.assertEqual(nuri_validate.affected_preset(["docs/tooling.md"]), "tool-core")
        self.assertEqual(nuri_validate.affected_preset(["lib/nuri/gfx/a.cpp"]), "renderer")
        self.assertEqual(nuri_validate.affected_preset(["tools/cli/main.cpp"]), "contract")

    def test_ctest_controls_are_forwarded(self):
        command = nuri_validate.step_command(
            nuri_validate.Step("tool-core", "bench-tests", "tool-core"),
            True,
            jobs=3,
            test_filter="Registry",
            repeat=2,
            extra_ctest_args=("--timeout", "5"),
        )
        self.assertIn("-j", command)
        self.assertIn("Registry", command)
        self.assertIn("until-fail:2", command)
        self.assertEqual(command[-2:], ["--timeout", "5"])

        commands = nuri_validate.step_commands(
            nuri_validate.Step("tool-core", "bench-tests", "tool-core"),
            False,
            test_filter="(Registry|Envelope)",
        )
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[1][-2:], ["-R", "(Registry|Envelope)"])
        self.assertEqual(commands[1][0], "ctest")

    def test_windows_ctest_dispatch_preserves_regex_argv(self):
        regex = "(Envelope|Baseline.*Verify)"
        command = nuri_ctest.ctest_command(
            ["release", "bench-tests", "-R", regex, "--timeout", "5"]
        )
        self.assertEqual(command[-4:], ["-R", regex, "--timeout", "5"])
        self.assertIn("release-bench-tests", command[2])
        environment = {
            "NURI_CTEST_MODE": "release",
            "NURI_CTEST_PROFILE": "bench-tests",
            "NURI_CTEST_ARGC": "2",
            "NURI_CTEST_ARG_0": "-R",
            "NURI_CTEST_ARG_1": regex,
        }
        command = nuri_ctest.ctest_command_from_environment(environment)
        self.assertEqual(command[-2:], ["-R", regex])

    def test_nested_duplicates_are_reported(self):
        document = {
            "resolution": [1, 1],
            "checkpoints": [{"id": "same"}, {"id": "same"}],
        }
        errors = nuri_validate._validate_nested_ids(document, Path("case.json"))
        self.assertTrue(any("duplicate" in error for error in errors))

    def test_doctor_recognizes_all_governed_approval_kinds(self):
        for kind in nuri_validate.GOVERNED_APPROVAL_KINDS:
            self.assertTrue(
                nuri_validate._is_governed_approval(
                    {
                        "schemaVersion": 1,
                        "kind": kind,
                        "planDigest": "sha256:" + "0" * 64,
                    }
                )
            )
        self.assertFalse(
            nuri_validate._is_governed_approval(
                {"schemaVersion": 1, "kind": "legacy", "planDigest": "fnv:1"}
            )
        )

    def test_case_location_matches_suite_and_id_namespace(self):
        with tempfile.TemporaryDirectory() as root_text:
            case_root = Path(root_text)
            valid = case_root / "stress" / "case.json"
            valid.parent.mkdir()
            self.assertEqual(
                nuri_validate._validate_case_location(
                    {"id": "stress.procedural.case", "suite": "stress"},
                    valid,
                    case_root,
                ),
                [],
            )
            errors = nuri_validate._validate_case_location(
                {"id": "correctness.procedural.case", "suite": "perf"},
                valid,
                case_root,
            )
            self.assertTrue(any("must match directory" in error for error in errors))
            self.assertTrue(any("id namespace" in error for error in errors))
            self.assertTrue(any("reserved" in error for error in errors))

    def test_safe_child_rejects_escape(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            self.assertEqual(nuri_validate._safe_child(root, root / "child"), (root / "child").resolve())
            with self.assertRaises(ValueError):
                nuri_validate._safe_child(root, root / ".." / "outside")

    def test_repository_contract_is_clean(self):
        self.assertEqual(nuri_validate.validate_repository(), [])

    def test_all_native_run_wrappers_have_guarded_no_build_fast_path(self):
        scripts = SCRIPT.parent
        for wrapper_name in ("run_benchmarks", "run_snapshots", "run_autotests"):
            for suffix in (".bat", ".sh"):
                path = scripts / f"{wrapper_name}{suffix}"
                with self.subTest(wrapper=path.name):
                    self.assertEqual(
                        nuri_validate._validate_run_wrapper_contract(
                            path.read_text(encoding="utf-8"), suffix, path
                        ),
                        [],
                    )

    def test_no_build_wrapper_contract_rejects_unguarded_build(self):
        errors = nuri_validate._validate_run_wrapper_contract(
            "--no-build\nno_build=1\n_nuri_build.sh\nbaseline\nif [[ ${no_build} -eq 0 ]]; then\n",
            ".sh",
            Path("scripts/run_broken.sh"),
        )
        self.assertTrue(any("not guarded" in error for error in errors))

    def test_startup_measurement_uses_successful_direct_process_samples(self):
        samples = check_tool_startup.measure_command(
            Path(sys.executable),
            ("-c", "print('metadata')"),
            warmups=1,
            iterations=3,
        )
        self.assertEqual(len(samples), 3)
        self.assertGreater(statistics.median(samples), 0.0)

    def test_dry_run_writes_common_v2_envelope(self):
        with tempfile.TemporaryDirectory() as root_text:
            old_root = nuri_validate.ARTIFACT_ROOT
            nuri_validate.ARTIFACT_ROOT = Path(root_text)
            try:
                args = argparse.Namespace(
                    preset="tool-core",
                    affected=[],
                    shard_index=0,
                    shard_count=1,
                    dry_run=True,
                    no_build=True,
                    timeout_seconds=1.0,
                    progress_jsonl=None,
                    keep_going=False,
                    junit=None,
                    jobs=2,
                    test_filter="ToolCore",
                    repeat=2,
                    ctest_args=[],
                )
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(nuri_validate.run_preset(args), 0)
                report_path = Path(stdout.getvalue().strip().splitlines()[-1])
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["kind"], "nuri.tool.result")
                self.assertEqual(report["tool"], "validate")
                self.assertEqual(report["status"], "investigative")
                self.assertEqual(report["selection"]["selected"], 3)
                self.assertEqual(report["selection"]["notRun"], 3)
            finally:
                nuri_validate.ARTIFACT_ROOT = old_root

    def test_bundle_contains_checksum_manifest(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            old_root = nuri_validate.ARTIFACT_ROOT
            nuri_validate.ARTIFACT_ROOT = root
            try:
                run_id = "20260710T120000.000Z-test"
                run = root / run_id
                run.mkdir()
                (run / "run.json").write_text(json.dumps({"runId": run_id}), encoding="utf-8")
                output = root / "bundle.tar.gz"
                with contextlib.redirect_stdout(io.StringIO()):
                    result = nuri_validate.bundle_run(argparse.Namespace(run_id=run_id, out=str(output)))
                self.assertEqual(result, 0)
                with tarfile.open(output, "r:gz") as archive:
                    manifest = json.load(archive.extractfile(f"{run_id}/bundle-manifest.json"))
                self.assertEqual(manifest["entries"][0]["path"], "run.json")
                self.assertTrue(manifest["entries"][0]["digest"].startswith("sha256:"))
                verification = nuri_validate.verify_bundle_file(output)
                self.assertEqual(verification["status"], "pass")
                self.assertEqual(verification["entries"], 1)
            finally:
                nuri_validate.ARTIFACT_ROOT = old_root

    def test_bundle_verifier_rejects_traversal_member(self):
        with tempfile.TemporaryDirectory() as root_text:
            output = Path(root_text) / "unsafe.tar"
            with tarfile.open(output, "w") as archive:
                payload = b"escape"
                info = tarfile.TarInfo("run/../outside.txt")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(ValueError, "unsafe bundle member"):
                nuri_validate.verify_bundle_file(output)


if __name__ == "__main__":
    unittest.main()
