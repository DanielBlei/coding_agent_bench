"""Tests for the --ae/--ak/--ek Harbor passthrough flags and frontend parity."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

os.environ.setdefault(
    "JOB_STORE_PATH", str(Path(tempfile.gettempdir()) / "cab-test-passthrough.db")
)

from coding_agent_bench.api import CreateJobRequest, build_cli_command
from coding_agent_bench.builder import HarborCommandBuilder, SupportedAgent
from coding_agent_bench.cli import app
from coding_agent_bench.utils import parse_envs, validate_key_value_entries


def _flag_pairs(command: list[str], flag: str) -> list[list[str]]:
    """Collect [flag, value] pairs for a repeatable flag, in order."""
    return [
        command[index : index + 2]
        for index, value in enumerate(command)
        if value == flag
    ]


class BuilderPassthroughTests(unittest.TestCase):
    def test_user_agent_env_follows_config_env_and_wins_collisions(self):
        command = HarborCommandBuilder()._build_command(
            agent="claude-code",
            dataset="example/dataset",
            model="example-model",
            environment="docker",
            agent_env={"ANTHROPIC_BASE_URL": "http://config:8000", "EXTRA": "1"},
            extra_agent_env=[
                "ANTHROPIC_BASE_URL=${ENMAAS_API_ORIGIN}",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000",
            ],
        )

        pairs = _flag_pairs(command, "--ae")
        self.assertEqual(
            pairs,
            [
                ["--ae", "ANTHROPIC_BASE_URL=http://config:8000"],
                ["--ae", "EXTRA=1"],
                # User entries come last so Harbor's last-wins parsing
                # lets them override the config value.
                ["--ae", "ANTHROPIC_BASE_URL=${ENMAAS_API_ORIGIN}"],
                ["--ae", "CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000"],
            ],
        )

    def test_user_agent_kwargs_follow_config_kwargs_and_version_pin(self):
        command = HarborCommandBuilder()._build_command(
            agent="claude-code",
            dataset="example/dataset",
            model="example-model",
            environment="docker",
            agent_kwargs={"reasoning_effort": "low"},
            extra_agent_kwargs=["reasoning_effort=high"],
        )

        pairs = _flag_pairs(command, "--ak")
        # The version pin is emitted first, config kwargs next, user entries last.
        version_values = [pair[1] for pair in pairs if pair[1].startswith("version=")]
        self.assertTrue(version_values)  # claude-code pins a version
        self.assertEqual(pairs[-1], ["--ak", "reasoning_effort=high"])

    def test_user_env_kwargs_follow_generated_persistent_env(self):
        with patch.dict(os.environ, {"HARBOR_PARENT": "parent-pod"}):
            command = HarborCommandBuilder()._build_command(
                agent="oracle",
                dataset="example/dataset",
                model="example-model",
                environment="openshift",
                extra_env_kwargs=["namespace=sre-leaderboard-test", "nft_redir_init=True"],
            )

        pairs = _flag_pairs(command, "--ek")
        self.assertTrue(pairs[0][1].startswith("persistent_env="))
        self.assertIn("parent-pod", pairs[0][1])
        self.assertEqual(
            pairs[1:],
            [
                ["--ek", "namespace=sre-leaderboard-test"],
                ["--ek", "nft_redir_init=True"],
            ],
        )

    def test_passthrough_values_stay_verbatim(self):
        command = HarborCommandBuilder()._build_command(
            agent="oracle",
            dataset="example/dataset",
            model="example-model",
            environment="docker",
            extra_agent_env=["ANTHROPIC_API_KEY=${ENMAAS_KEY}"],
            extra_agent_kwargs=["persistent_env={\"a\": 1, \"b\": 2}"],
        )

        self.assertIn(
            ["--ae", "ANTHROPIC_API_KEY=${ENMAAS_KEY}"], _flag_pairs(command, "--ae")
        )
        self.assertIn(
            ["--ak", 'persistent_env={"a": 1, "b": 2}'], _flag_pairs(command, "--ak")
        )

    def test_no_passthrough_flags_without_entries(self):
        command = HarborCommandBuilder()._build_command(
            agent="oracle",
            dataset="example/dataset",
            model="example-model",
            environment="docker",
        )

        self.assertNotIn("--ae", command)
        self.assertNotIn("--ak", command)
        self.assertNotIn("--ek", command)

    def test_build_forwards_passthrough_lists(self):
        with patch.object(
            HarborCommandBuilder,
            "_build_command",
            return_value=["harbor", "run"],
        ) as build_command:
            HarborCommandBuilder().build(
                agent="oracle",
                dataset="example/dataset",
                model_name="example-model",
                server_url="http://model.example",
                environment="docker",
                extra_agent_env=["FOO=bar"],
                extra_agent_kwargs=["thinking=high"],
                extra_env_kwargs=["namespace=ns"],
            )

        self.assertEqual(build_command.call_args.kwargs["extra_agent_env"], ["FOO=bar"])
        self.assertEqual(
            build_command.call_args.kwargs["extra_agent_kwargs"], ["thinking=high"]
        )
        self.assertEqual(
            build_command.call_args.kwargs["extra_env_kwargs"], ["namespace=ns"]
        )


class CliPassthroughTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.required_args = [
            "--agent",
            "oracle",
            "--dataset",
            "example/dataset",
            "--model-name",
            "example-model",
            "--server-url",
            "http://model.example",
        ]

    def test_cli_forwards_repeatable_passthrough_flags(self):
        with patch.object(
            HarborCommandBuilder,
            "build",
            return_value=(["harbor", "run"], Path("jobs/test")),
        ) as build:
            result = self.runner.invoke(
                app,
                [
                    "run",
                    *self.required_args,
                    "--ae", "FOO=bar",
                    "--agent-env", "BAZ=qux",
                    "--ak", "thinking=high",
                    "--agent-kwarg", "version=9.9.9",
                    "--ek", "namespace=ns",
                    "--env-kwarg", "nft_redir_init=True",
                    "--dry-run",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            build.call_args.kwargs["extra_agent_env"], ["FOO=bar", "BAZ=qux"]
        )
        self.assertEqual(
            build.call_args.kwargs["extra_agent_kwargs"],
            ["thinking=high", "version=9.9.9"],
        )
        self.assertEqual(
            build.call_args.kwargs["extra_env_kwargs"],
            ["namespace=ns", "nft_redir_init=True"],
        )

    def test_cli_rejects_entry_without_separator(self):
        result = self.runner.invoke(
            app,
            ["run", *self.required_args, "--ae", "NOT_AN_ENTRY", "--dry-run"],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid --ae entry", result.output)

    def test_cli_rejects_invalid_env_var_name(self):
        result = self.runner.invoke(
            app,
            ["run", *self.required_args, "--ae", "1BAD-KEY=value", "--dry-run"],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid --ae entry", result.output)

    def test_cli_accepts_empty_agent_env_value(self):
        with patch.object(
            HarborCommandBuilder,
            "build",
            return_value=(["harbor", "run"], Path("jobs/test")),
        ) as build:
            result = self.runner.invoke(
                app,
                ["run", *self.required_args, "--ae", "ANTHROPIC_API_KEY=", "--dry-run"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            build.call_args.kwargs["extra_agent_env"], ["ANTHROPIC_API_KEY="]
        )

    def test_cli_envs_flag_is_repeatable_and_merges_comma_pairs(self):
        with patch.object(
            HarborCommandBuilder,
            "build",
            return_value=(["harbor", "run"], Path("jobs/test")),
        ):
            result = self.runner.invoke(
                app,
                [
                    "run",
                    *self.required_args,
                    "--envs", "FOO=bar,BAZ=qux",
                    "--envs", "QUX=1",
                    "--dry-run",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("export FOO=bar", result.output)
        self.assertIn("export BAZ=qux", result.output)
        self.assertIn("export QUX=1", result.output)

    def test_cli_thinking_flag_is_removed(self):
        result = self.runner.invoke(
            app,
            ["run", *self.required_args, "--thinking", "high", "--dry-run"],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--thinking", result.output)


class QueuePassthroughTests(unittest.TestCase):
    def _request(self, **overrides) -> CreateJobRequest:
        defaults = dict(
            job_name="passthrough-test",
            agent=SupportedAgent.oracle,
            dataset="example/dataset",
            model_name="example-model",
            server_url="http://model.example",
        )
        defaults.update(overrides)
        return CreateJobRequest(**defaults)

    def test_queue_command_serializes_each_passthrough_entry(self):
        request = self._request(
            agent_env=["ANTHROPIC_BASE_URL=https://gateway.example", "FOO=bar"],
            agent_kwargs=["reasoning_effort=high"],
            env_kwargs=["namespace=sre-leaderboard-test"],
            harbor_envs=["DATA_ROOT=/data"],
            agent_timeout_multiplier=3,
            allow_agent_host=["api.enmaas.devshift.net", "downloads.claude.ai"],
            max_retries=1,
            retry_include=["UnknownApiError"],
        )

        command = build_cli_command(request)

        self.assertEqual(
            _flag_pairs(command, "--ae"),
            [
                ["--ae", "ANTHROPIC_BASE_URL=https://gateway.example"],
                ["--ae", "FOO=bar"],
            ],
        )
        self.assertEqual(
            _flag_pairs(command, "--ak"), [["--ak", "reasoning_effort=high"]]
        )
        self.assertEqual(
            _flag_pairs(command, "--ek"),
            [["--ek", "namespace=sre-leaderboard-test"]],
        )
        self.assertEqual(_flag_pairs(command, "--envs"), [["--envs", "DATA_ROOT=/data"]])
        self.assertEqual(
            _flag_pairs(command, "--agent-timeout-multiplier"),
            [["--agent-timeout-multiplier", "3.0"]],
        )
        self.assertEqual(
            _flag_pairs(command, "--allow-agent-host"),
            [
                ["--allow-agent-host", "api.enmaas.devshift.net"],
                ["--allow-agent-host", "downloads.claude.ai"],
            ],
        )
        self.assertEqual(_flag_pairs(command, "--max-retries"), [["--max-retries", "1"]])
        self.assertEqual(
            _flag_pairs(command, "--retry-include"),
            [["--retry-include", "UnknownApiError"]],
        )

    def test_queue_command_omits_unset_passthrough_flags(self):
        command = build_cli_command(self._request())

        for flag in ("--ae", "--ak", "--ek", "--envs", "--agent-timeout-multiplier",
                     "--allow-agent-host", "--max-retries", "--retry-include"):
            self.assertNotIn(flag, command)

    def test_queue_rejects_env_entry_without_separator(self):
        with pytest.raises(ValidationError) as excinfo:
            self._request(agent_env=["ANTHROPIC_BASE_URL"])

        self.assertIn("agent_env", str(excinfo.value))
        self.assertIn("key=value", str(excinfo.value))

    def test_queue_rejects_kwarg_entry_without_separator(self):
        with pytest.raises(ValidationError) as excinfo:
            self._request(env_kwargs=["namespace"])

        self.assertIn("env_kwargs", str(excinfo.value))
        self.assertIn("key=value", str(excinfo.value))

    def test_queue_rejects_invalid_env_var_name(self):
        with pytest.raises(ValidationError) as excinfo:
            self._request(harbor_envs=["1BAD-KEY=value"])

        self.assertIn("harbor_envs", str(excinfo.value))
        self.assertIn("[A-Za-z_]", str(excinfo.value))

    def test_queue_accepts_empty_env_value(self):
        request = self._request(agent_env=["ANTHROPIC_API_KEY="])

        command = build_cli_command(request)

        self.assertEqual(
            _flag_pairs(command, "--ae"), [["--ae", "ANTHROPIC_API_KEY="]]
        )


class ParseEnvsTests(unittest.TestCase):
    def test_parse_envs_accepts_single_comma_separated_string(self):
        self.assertEqual(
            parse_envs("FOO=bar, BAZ=qux"), {"FOO": "bar", "BAZ": "qux"}
        )

    def test_parse_envs_merges_repeatable_entries_in_order(self):
        self.assertEqual(
            parse_envs(["FOO=bar,BAZ=qux", "QUX=1", "FOO=override"]),
            {"FOO": "override", "BAZ": "qux", "QUX": "1"},
        )

    def test_parse_envs_ignores_empty_input(self):
        self.assertEqual(parse_envs(None), {})
        self.assertEqual(parse_envs([]), {})
        self.assertEqual(parse_envs(""), {})

    def test_parse_envs_rejects_entry_without_separator(self):
        with pytest.raises(ValueError, match="expected key=value"):
            parse_envs(["FOO"])


class ValidateKeyValueEntriesTests(unittest.TestCase):
    def test_accepts_well_formed_entries(self):
        validate_key_value_entries(["FOO=bar", "BAZ="], "--ae", env_keys=True)
        validate_key_value_entries(["a.b=c,d", "x=1"], "--ek")

    def test_rejects_missing_separator(self):
        with pytest.raises(ValueError, match="Invalid --ae entry.*FOO"):
            validate_key_value_entries(["FOO"], "--ae")

    def test_rejects_empty_key(self):
        with pytest.raises(ValueError, match="empty key"):
            validate_key_value_entries(["=value"], "--ak")

    def test_rejects_invalid_env_key_only_in_env_mode(self):
        # kwarg keys may contain dots and dashes; env keys may not.
        validate_key_value_entries(["a.b-c=v"], "--ek")
        with pytest.raises(ValueError, match="variable name"):
            validate_key_value_entries(["a.b-c=v"], "--ae", env_keys=True)
