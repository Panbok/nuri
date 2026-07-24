from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "nuri_build", ROOT / "scripts" / "nuri_build.py"
)
assert SPEC and SPEC.loader
nuri_build = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nuri_build
SPEC.loader.exec_module(nuri_build)


class NuriBuildPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = nuri_build.load_policy()

    def test_release_profiles_map_to_stable_semantic_variants(self) -> None:
        checked = nuri_build.legacy_request("release", "editor", (), self.policy)
        fast = nuri_build.legacy_request("release", "snapshot", (), self.policy)
        tracy = nuri_build.legacy_request(
            "release", "benchmark", ("cpu",), self.policy
        )

        self.assertEqual(checked.variant, "release-checked")
        self.assertEqual(checked.capability, "developer-full")
        self.assertEqual(fast.variant, "release-fast")
        self.assertEqual(fast.capability, "runtime-tools")
        self.assertEqual(tracy.variant, "release-tracy")
        self.assertEqual(tracy.target, "nuri-bench")

    def test_debug_off_is_a_distinct_identity(self) -> None:
        default = nuri_build.legacy_request("debug", "app", (), self.policy)
        off = nuri_build.legacy_request("debug", "app", ("off",), self.policy)
        self.assertEqual(default.variant, "debug-dev")
        self.assertEqual(off.variant, "debug-dev-no-tracy")

    def test_every_legacy_alias_resolves_to_a_declared_target(self) -> None:
        for name, record in self.policy["legacyProfiles"].items():
            for alias in [name, *record.get("aliases", [])]:
                request = nuri_build.legacy_request(
                    "release", alias, (), self.policy
                )
                targets = self.policy["capabilities"][request.capability]["targets"]
                self.assertIn(request.target, targets, alias)

    def test_conflicting_legacy_modifiers_fail(self) -> None:
        with self.assertRaises(nuri_build.NuriBuildError):
            nuri_build.legacy_request(
                "release", "app", ("cpu", "off"), self.policy
            )

    def test_compile_digest_excludes_source_revision_by_construction(self) -> None:
        variant = self.policy["variants"]["release-checked"]
        self.assertNotIn("sourceRevision", variant)
        self.assertNotIn("requestedTarget", variant)


class NuriBuildClangdTests(unittest.TestCase):
    def _context(
        self,
        tree: Path,
        *,
        configuration: str = "Debug",
        capability: str = "developer-full",
    ) -> mock.Mock:
        context = mock.Mock()
        context.tree = tree
        context.tree_digest = "sha256:" + "1" * 64
        context.dependency_digest = "sha256:" + "2" * 64
        context.request.configuration = configuration
        context.request.capability = capability
        context.request.target = "nuri_editor"
        return context

    def test_debug_developer_database_is_published_at_stable_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "_trees" / "debug"
            tree.mkdir(parents=True)
            database = b'[{"directory":"hashed-tree","file":"source.cpp"}]\n'
            (tree / "compile_commands.json").write_bytes(database)

            with mock.patch.object(nuri_build, "BUILD_ROOT", root):
                nuri_build._publish_clangd_database(self._context(tree))

            self.assertEqual(
                (root / "compile_commands.json").read_bytes(),
                database,
            )
            self.assertEqual(
                list(root.glob(".compile_commands.json.*.tmp")),
                [],
            )

    def test_non_debug_or_partial_database_does_not_replace_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "_trees" / "candidate"
            tree.mkdir(parents=True)
            (tree / "compile_commands.json").write_text(
                "candidate", encoding="utf-8"
            )
            destination = root / "compile_commands.json"
            destination.write_text("debug-developer", encoding="utf-8")

            with mock.patch.object(nuri_build, "BUILD_ROOT", root):
                for configuration, capability in (
                    ("Release", "developer-full"),
                    ("Debug", "runtime-tools"),
                ):
                    with self.subTest(
                        configuration=configuration, capability=capability
                    ):
                        nuri_build._publish_clangd_database(
                            self._context(
                                tree,
                                configuration=configuration,
                                capability=capability,
                            )
                        )

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "debug-developer",
            )

    def test_successful_configure_publishes_clangd_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = self._context(Path(directory) / "tree")
            lock = mock.MagicMock()

            with (
                mock.patch.object(nuri_build, "reserve_tree"),
                mock.patch.object(
                    nuri_build,
                    "_nvrhi_input",
                    return_value=("sha256:" + "3" * 64, {}),
                ),
                mock.patch.object(
                    nuri_build,
                    "_needs_configure",
                    return_value=(False, "ready identity matches"),
                ),
                mock.patch.object(nuri_build, "update_registry"),
                mock.patch.object(nuri_build, "FileLock", return_value=lock),
                mock.patch.object(
                    nuri_build, "prepare_nvrhi", return_value=Path(directory)
                ),
                mock.patch.object(nuri_build, "_dependency_owner"),
                mock.patch.object(nuri_build, "_build_receipts", return_value={}),
                mock.patch.object(nuri_build, "_publish_manifest", return_value={}),
                mock.patch.object(
                    nuri_build, "_publish_clangd_database"
                ) as publish,
            ):
                nuri_build.build(context, configure_only=True)

            publish.assert_called_once_with(context)


class NuriBuildHealthTests(unittest.TestCase):
    def test_configured_tree_without_build_receipts_is_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory)
            identity = {
                "treeIdentityDigest": "sha256:" + "1" * 64,
                "treeIdentity": {
                    "tools": {
                        "ninja": {
                            "path": sys.executable,
                            "version": "test-version",
                        }
                    }
                },
            }
            (tree / nuri_build.IDENTITY_FILE).write_text(
                json.dumps(identity), encoding="utf-8"
            )
            (tree / "CMakeCache.txt").write_text(
                "CMAKE_GENERATOR:INTERNAL=Ninja\n", encoding="utf-8"
            )
            (tree / "build.ninja").write_text("rule noop", encoding="utf-8")
            (tree / nuri_build.ARTIFACT_FILE).write_text("{}", encoding="utf-8")

            with mock.patch.object(
                nuri_build, "command_output", return_value="test-version"
            ):
                result = nuri_build.health_for_tree(tree)

            self.assertEqual(result["state"], "healthy")
            self.assertIn("configured, not built", result["reasons"][0])

    def test_health_detects_truncated_ninja_deps_without_mutating_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory)
            identity = {
                "treeIdentityDigest": "sha256:" + "1" * 64,
            }
            (tree / nuri_build.IDENTITY_FILE).write_text(
                json.dumps(identity), encoding="utf-8"
            )
            (tree / "CMakeCache.txt").write_text("cache", encoding="utf-8")
            (tree / "build.ninja").write_text("rule noop", encoding="utf-8")
            (tree / ".ninja_log").write_text(
                "# ninja log v5\n1\t2\t3\toutput\tdeadbeef\n",
                encoding="utf-8",
            )
            deps = tree / ".ninja_deps"
            deps.write_bytes(b"# ninjadeps\n")
            before = deps.read_bytes()

            result = nuri_build.health_for_tree(tree)

            self.assertEqual(result["state"], "damaged")
            self.assertEqual(deps.read_bytes(), before)
            self.assertTrue(result["sideEffectFree"])

    def test_safe_managed_child_rejects_the_managed_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(nuri_build.NuriBuildError):
                nuri_build.safe_managed_child(root, root)


if __name__ == "__main__":
    unittest.main()
