"""Tests for Nebius resource/model concurrency and launch configuration."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from coding_agent_bench import api
from coding_agent_bench.nebius_utils import (
    NebiusInstanceManager,
    get_model_max_concurrency,
)


@pytest.fixture(scope="module")
def client():
    """Use one client because the API owns module-level async state."""
    with TestClient(api.app) as test_client:
        yield test_client


def test_get_model_max_concurrency_returns_configured_value():
    assert get_model_max_concurrency(
        gpu_config="b200",
        model_name="Qwen/Qwen3.8-27B",
    ) == 12


def test_get_model_max_concurrency_uses_inherited_regional_config():
    assert get_model_max_concurrency(
        gpu_config="b200a",
        model_name="Qwen/Qwen3.8-27B",
    ) == 12
    assert get_model_max_concurrency(
        gpu_config="b200x8a",
        model_name="RedHatAI/DeepSeek-V4-Flash",
    ) == 14


@pytest.mark.parametrize(
    ("gpu_config", "model_name"),
    [
        ("not-a-config", "Qwen/Qwen3.8-27B"),
        ("h200", "Qwen/Qwen3.8-27B"),
        ("b200", "not-a-model"),
    ],
)
def test_get_model_max_concurrency_rejects_unknown_or_unsupported_pairs(
    gpu_config, model_name,
):
    with pytest.raises(ValueError):
        get_model_max_concurrency(gpu_config=gpu_config, model_name=model_name)


def test_model_max_concurrency_endpoint_returns_integer(client):
    response = client.get(
        "/api/model-max-concurrency",
        params={"model_name": "Qwen/Qwen3.8-27B", "gpu_config": "b200"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "Qwen/Qwen3.8-27B",
        "gpu_config": "b200",
        "max_concurrency": 12,
    }


@pytest.mark.parametrize(
    ("model_name", "gpu_config"),
    [
        ("not-a-model", "b200"),
        ("Qwen/Qwen3.8-27B", "not-a-config"),
        ("Qwen/Qwen3.8-27B", "h200"),
    ],
)
def test_model_max_concurrency_endpoint_returns_not_found_for_invalid_pairs(
    client, model_name, gpu_config,
):
    response = client.get(
        "/api/model-max-concurrency",
        params={"model_name": model_name, "gpu_config": gpu_config},
    )

    assert response.status_code == 404


def test_start_model_uses_resource_tensor_parallelism_and_extra_args(monkeypatch):
    manager = object.__new__(NebiusInstanceManager)
    commands = []

    async def get_instance(_instance_name):
        return {
            "spec": {
                "resources": {
                    "platform": "gpu-b200-sxm",
                    "preset": "8gpu-160vcpu-1792gb",
                },
            },
        }

    async def instance_exec(**kwargs):
        commands.append(kwargs["command"])

    monkeypatch.setattr(manager, "get_instance", get_instance)
    monkeypatch.setattr(manager, "instance_exec", instance_exec)

    asyncio.run(manager.start_model("instance", "RedHatAI/DeepSeek-V4-Flash"))

    shell_command = commands[0][2]
    assert "--tensor-parallel-size 8" in shell_command
    assert "--moe-backend deep_gemm_mega_moe" in shell_command
    assert "--attention_config.use_fp4_indexer_cache True" in shell_command


def test_create_instance_uses_preemptible_pricing_policy(monkeypatch):
    manager = object.__new__(NebiusInstanceManager)
    manager._user = "tester"
    manager._ssh_key = "ssh-rsa test-key"
    manager._parent_id = "project-1"
    manager._service_account_id = "service-account-1"
    commands = []

    async def instance_exec(args):
        commands.append(args)

    monkeypatch.setattr(manager, "exec", instance_exec)

    asyncio.run(manager.create_instance("worker-1", "subnet-1", gpu_config="h200"))

    assert len(commands) == 1
    for argument in (
        "--preemptible-on-preemption",
        "stop",
        "--recovery-policy",
        "fail",
        "--follows-spot-price=true",
    ):
        assert argument in commands[0]


def test_create_instance_omits_preemptible_policy_when_unsupported(monkeypatch):
    manager = object.__new__(NebiusInstanceManager)
    manager._user = "tester"
    manager._ssh_key = "ssh-rsa test-key"
    manager._parent_id = "project-1"
    manager._service_account_id = "service-account-1"
    commands = []

    class NonPreemptibleResource:
        platform = "gpu-test"
        preset = "test-preset"
        additional_args = []
        is_preemptible = False

    async def instance_exec(args):
        commands.append(args)

    monkeypatch.setattr(manager, "exec", instance_exec)
    monkeypatch.setattr(
        "coding_agent_bench.nebius_utils.RESOURCE_CONFIG_REGISTRY",
        {"test": NonPreemptibleResource},
    )

    asyncio.run(manager.create_instance("worker-1", "subnet-1", gpu_config="test"))

    assert not any(
        argument in commands[0]
        for argument in (
            "--preemptible-on-preemption",
            "--recovery-policy",
            "--follows-spot-price=true",
        )
    )


def test_create_profile_updates_existing_profile(monkeypatch, tmp_path):
    manager = object.__new__(NebiusInstanceManager)
    manager._parent_id = "project-1"
    manager._tenant_id = "tenant-1"
    commands = []

    async def instance_exec(args):
        commands.append(args)
        if args == ["profile", "list"]:
            return "service-account [default]"
        return "updated"

    monkeypatch.setattr(manager, "exec", instance_exec)

    result = asyncio.run(
        manager.create_profile("service-account", tmp_path / "credentials.json")
    )

    assert result == "updated"
    assert commands == [
        ["profile", "list"],
        [
            "profile",
            "update",
            "service-account",
            "--endpoint",
            "api.nebius.cloud",
            "--parent-id",
            "project-1",
            "--tenant-id",
            "tenant-1",
            "--service-account-file",
            str(tmp_path / "credentials.json"),
        ],
    ]


def test_create_profile_creates_missing_profile(monkeypatch, tmp_path):
    manager = object.__new__(NebiusInstanceManager)
    manager._parent_id = "project-1"
    manager._tenant_id = "tenant-1"
    commands = []

    async def instance_exec(args):
        commands.append(args)
        return "created"

    monkeypatch.setattr(manager, "exec", instance_exec)

    result = asyncio.run(
        manager.create_profile("service-account", tmp_path / "credentials.json")
    )

    assert result == "created"
    assert commands[0] == ["profile", "list"]
    assert commands[1][-2:] == ["--profile", "service-account"]


def test_init_activates_the_profile(monkeypatch, tmp_path):
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}")
    ssh_key_path = tmp_path / "id.pub"
    ssh_key_path.write_text("ssh-rsa test-key")
    manager = NebiusInstanceManager(
        credentials_path=credentials_path,
        user="tester",
        ssh_public_key_path=ssh_key_path,
        ssh_private_key_path=tmp_path / "id",
        parent_id="project-1",
        tenant_id="tenant-1",
        service_account_id="service-account-1",
    )
    calls = []

    async def create_profile(name, credentials_path):
        calls.append(("create", name, credentials_path))
        return "profile-created"

    async def activate_profile(name):
        calls.append(("activate", name))

    monkeypatch.setattr(manager, "create_profile", create_profile)
    monkeypatch.setattr(manager, "activate_profile", activate_profile)

    asyncio.run(manager.init())

    assert calls == [
        ("create", "default-service-account", credentials_path),
        ("activate", "default-service-account"),
    ]


def test_get_instance_scopes_lookup_to_parent(monkeypatch):
    manager = object.__new__(NebiusInstanceManager)
    manager._parent_id = "project-1"
    commands = []

    async def instance_exec(args):
        commands.append(args)
        return json.dumps({"metadata": {"name": "worker-1"}})

    monkeypatch.setattr(manager, "exec", instance_exec)

    assert asyncio.run(manager.get_instance("worker-1")) == {
        "metadata": {"name": "worker-1"}
    }
    assert commands == [
        [
            "compute",
            "instance",
            "get-by-name",
            "--name",
            "worker-1",
            "--parent-id",
            "project-1",
            "--format",
            "json",
        ]
    ]
