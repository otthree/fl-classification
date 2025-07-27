"""Unit tests for adni_classification.config.fl_config module."""

import os
from unittest.mock import patch

from adni_classification.config.fl_config import (
    ClientMachineConfig,
    FLConfig,
    MultiMachineConfig,
    ServerMachineConfig,
    SSHConfig,
)


class TestSSHConfig:
    """Test cases for SSHConfig dataclass."""

    def test_ssh_config_creation_with_defaults(self):
        """Test SSHConfig creation with default values."""
        config = SSHConfig()

        assert config.timeout == 30
        assert config.banner_timeout == 30
        assert config.auth_timeout == 30

    def test_ssh_config_creation_with_custom_values(self):
        """Test SSHConfig creation with custom values."""
        config = SSHConfig(
            timeout=60,
            banner_timeout=45,
            auth_timeout=90,
        )

        assert config.timeout == 60
        assert config.banner_timeout == 45
        assert config.auth_timeout == 90


class TestClientMachineConfig:
    """Test cases for ClientMachineConfig dataclass."""

    def test_client_machine_config_creation_with_required_fields(self):
        """Test ClientMachineConfig creation with minimal required fields."""
        config = ClientMachineConfig(
            host="client1.example.com",
            username="testuser",
        )

        assert config.host == "client1.example.com"
        assert config.username == "testuser"
        assert config.password is None
        assert config.partition_id == 0
        assert config.project_dir is None
        assert config.config_file is None
        assert config.sequential_experiment is False
        assert config.train_sequential_labels is None
        assert config.val_sequential_labels is None

    def test_client_machine_config_creation_with_custom_values(self):
        """Test ClientMachineConfig creation with custom values."""
        config = ClientMachineConfig(
            host="client2.example.com",
            username="clientuser",
            password="clientpass",
            partition_id=1,
            project_dir="/home/clientuser/project",
            config_file="client2.yaml",
            sequential_experiment=True,
            train_sequential_labels=["CN", "AD"],
            val_sequential_labels=["MCI"],
        )

        assert config.host == "client2.example.com"
        assert config.username == "clientuser"
        assert config.password == "clientpass"
        assert config.partition_id == 1
        assert config.project_dir == "/home/clientuser/project"
        assert config.config_file == "client2.yaml"
        assert config.sequential_experiment is True
        assert config.train_sequential_labels == ["CN", "AD"]
        assert config.val_sequential_labels == ["MCI"]


class TestServerMachineConfig:
    """Test cases for ServerMachineConfig dataclass."""

    def test_server_machine_config_creation_with_required_fields(self):
        """Test ServerMachineConfig creation with minimal required fields."""
        config = ServerMachineConfig(
            host="server.example.com",
            username="serveruser",
        )

        assert config.host == "server.example.com"
        assert config.username == "serveruser"
        assert config.password is None
        assert config.port == 9092
        assert config.config_file is None
        assert config.sequential_experiment is False
        assert config.train_sequential_labels is None
        assert config.val_sequential_labels is None

    def test_server_machine_config_creation_with_custom_values(self):
        """Test ServerMachineConfig creation with custom values."""
        config = ServerMachineConfig(
            host="server2.example.com",
            username="serveruser2",
            password="serverpass",
            port=8080,
            config_file="server.yaml",
            sequential_experiment=True,
            train_sequential_labels=["CN", "AD", "MCI"],
            val_sequential_labels=["CN", "AD"],
        )

        assert config.host == "server2.example.com"
        assert config.username == "serveruser2"
        assert config.password == "serverpass"
        assert config.port == 8080
        assert config.config_file == "server.yaml"
        assert config.sequential_experiment is True
        assert config.train_sequential_labels == ["CN", "AD", "MCI"]
        assert config.val_sequential_labels == ["CN", "AD"]


class TestMultiMachineConfig:
    """Test cases for MultiMachineConfig dataclass."""

    def test_multi_machine_config_creation_with_defaults(self):
        """Test MultiMachineConfig creation with default values."""
        config = MultiMachineConfig()

        assert config.server is None
        assert config.clients == []
        assert config.project_dir is None
        assert config.venv_path is None
        assert config.venv_activate is None
        assert isinstance(config.ssh, SSHConfig)

    def test_multi_machine_config_creation_with_custom_values(self):
        """Test MultiMachineConfig creation with custom configurations."""
        server_config = ServerMachineConfig(
            host="server.example.com",
            username="serveruser",
        )
        client_config = ClientMachineConfig(
            host="client1.example.com",
            username="clientuser",
        )
        ssh_config = SSHConfig(timeout=60)

        config = MultiMachineConfig(
            server=server_config,
            clients=[client_config],
            project_dir="/home/user/project",
            venv_path="/home/user/.venv",
            venv_activate="source /home/user/.venv/bin/activate",
            ssh=ssh_config,
        )

        assert config.server == server_config
        assert config.clients == [client_config]
        assert config.project_dir == "/home/user/project"
        assert config.venv_path == "/home/user/.venv"
        assert config.venv_activate == "source /home/user/.venv/bin/activate"
        assert config.ssh == ssh_config

    def test_get_server_config_dict_with_server(self):
        """Test get_server_config_dict with server configuration."""
        server_config = ServerMachineConfig(
            host="server.example.com",
            username="serveruser",
            password="serverpass",
            port=8080,
            config_file="server.yaml",
            sequential_experiment=True,
            train_sequential_labels=["CN", "AD"],
            val_sequential_labels=["MCI"],
        )

        config = MultiMachineConfig(server=server_config)
        result = config.get_server_config_dict()

        expected = {
            "host": "server.example.com",
            "username": "serveruser",
            "password": "serverpass",
            "port": 8080,
            "config_file": "server.yaml",
            "sequential_experiment": True,
            "train_sequential_labels": ["CN", "AD"],
            "val_sequential_labels": ["MCI"],
        }

        assert result == expected

    def test_get_server_config_dict_without_server(self):
        """Test get_server_config_dict without server configuration."""
        config = MultiMachineConfig()
        result = config.get_server_config_dict()

        assert result == {}

    @patch.dict(os.environ, {"FL_PASSWORD": "env_password"})
    def test_get_server_config_dict_with_env_password(self):
        """Test get_server_config_dict with password from environment."""
        server_config = ServerMachineConfig(
            host="server.example.com",
            username="serveruser",
            password=None,  # No password set, should use env
        )

        config = MultiMachineConfig(server=server_config)
        result = config.get_server_config_dict()

        assert result["password"] == "env_password"

    def test_get_clients_config_dict_with_clients(self):
        """Test get_clients_config_dict with client configurations."""
        client1_config = ClientMachineConfig(
            host="client1.example.com",
            username="client1user",
            password="client1pass",
            partition_id=0,
            project_dir="/client1/project",
            config_file="client1.yaml",
            sequential_experiment=True,
            train_sequential_labels=["CN"],
            val_sequential_labels=["AD"],
        )
        client2_config = ClientMachineConfig(
            host="client2.example.com",
            username="client2user",
            partition_id=1,
        )

        config = MultiMachineConfig(
            clients=[client1_config, client2_config],
            project_dir="/default/project",
        )
        result = config.get_clients_config_dict()

        expected = [
            {
                "host": "client1.example.com",
                "username": "client1user",
                "password": "client1pass",
                "partition_id": 0,
                "project_dir": "/client1/project",
                "config_file": "client1.yaml",
                "sequential_experiment": True,
                "train_sequential_labels": ["CN"],
                "val_sequential_labels": ["AD"],
            },
            {
                "host": "client2.example.com",
                "username": "client2user",
                "password": None,
                "partition_id": 1,
                "project_dir": "/default/project",  # Uses default from MultiMachineConfig
            },
        ]

        assert result == expected

    def test_get_clients_config_dict_empty(self):
        """Test get_clients_config_dict with no clients."""
        config = MultiMachineConfig()
        result = config.get_clients_config_dict()

        assert result == []

    @patch.dict(os.environ, {"FL_PASSWORD": "env_client_pass"})
    def test_get_clients_config_dict_with_env_password(self):
        """Test get_clients_config_dict with password from environment."""
        client_config = ClientMachineConfig(
            host="client1.example.com",
            username="client1user",
            password=None,  # No password set, should use env
            partition_id=0,
        )

        config = MultiMachineConfig(clients=[client_config])
        result = config.get_clients_config_dict()

        assert result[0]["password"] == "env_client_pass"


class TestFLConfig:
    """Test cases for FLConfig dataclass."""

    def test_fl_config_creation_with_defaults(self):
        """Test FLConfig creation with default values."""
        config = FLConfig()

        assert config.num_rounds == 10
        assert config.strategy == "fedavg"
        assert config.fraction_fit == 1.0
        assert config.fraction_evaluate == 1.0
        assert config.min_fit_clients == 2
        assert config.min_evaluate_clients == 2
        assert config.min_available_clients == 2
        assert config.local_epochs == 1
        assert config.client_config_files is None
        assert config.evaluate_frequency == 1
        assert config.fedprox_mu == 0.01
        assert config.dp_noise_multiplier == 0.1
        assert config.dp_dropout_rate == 0.0
        assert config.dp_clipping_norm == 1.0
        assert config.secagg_num_shares == 3
        assert config.secagg_reconstruction_threshold == 3
        assert config.secagg_max_weight == 16777216
        assert config.secagg_timeout == 30.0
        assert config.secagg_clipping_range == 1.0
        assert config.secagg_quantization_range == 1048576
        assert config.client_id is None
        assert config.multi_machine is None

    def test_fl_config_creation_with_custom_values(self):
        """Test FLConfig creation with custom values."""
        multi_machine_config = MultiMachineConfig()

        config = FLConfig(
            num_rounds=50,
            strategy="fedprox",
            fraction_fit=0.8,
            fraction_evaluate=0.6,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=5,
            local_epochs=5,
            client_config_files=["client1.yaml", "client2.yaml"],
            evaluate_frequency=2,
            fedprox_mu=0.05,
            dp_noise_multiplier=0.2,
            dp_dropout_rate=0.1,
            dp_clipping_norm=2.0,
            secagg_num_shares=5,
            secagg_reconstruction_threshold=4,
            secagg_max_weight=32768,
            secagg_timeout=60.0,
            secagg_clipping_range=2.0,
            secagg_quantization_range=2048,
            client_id=1,
            multi_machine=multi_machine_config,
        )

        assert config.num_rounds == 50
        assert config.strategy == "fedprox"
        assert config.fraction_fit == 0.8
        assert config.fraction_evaluate == 0.6
        assert config.min_fit_clients == 3
        assert config.min_evaluate_clients == 3
        assert config.min_available_clients == 5
        assert config.local_epochs == 5
        assert config.client_config_files == ["client1.yaml", "client2.yaml"]
        assert config.evaluate_frequency == 2
        assert config.fedprox_mu == 0.05
        assert config.dp_noise_multiplier == 0.2
        assert config.dp_dropout_rate == 0.1
        assert config.dp_clipping_norm == 2.0
        assert config.secagg_num_shares == 5
        assert config.secagg_reconstruction_threshold == 4
        assert config.secagg_max_weight == 32768
        assert config.secagg_timeout == 60.0
        assert config.secagg_clipping_range == 2.0
        assert config.secagg_quantization_range == 2048
        assert config.client_id == 1
        assert config.multi_machine == multi_machine_config

    def test_fl_config_differential_privacy_strategy(self):
        """Test FLConfig with differential privacy strategy."""
        config = FLConfig(
            strategy="differential_privacy",
            dp_noise_multiplier=0.5,
            dp_dropout_rate=0.2,
            dp_clipping_norm=1.5,
        )

        assert config.strategy == "differential_privacy"
        assert config.dp_noise_multiplier == 0.5
        assert config.dp_dropout_rate == 0.2
        assert config.dp_clipping_norm == 1.5

    def test_fl_config_secagg_plus_strategy(self):
        """Test FLConfig with SecAgg+ strategy."""
        config = FLConfig(
            strategy="secaggplus",
            secagg_num_shares=7,
            secagg_reconstruction_threshold=5,
            secagg_max_weight=65536,
            secagg_timeout=120.0,
            secagg_clipping_range=3.0,
            secagg_quantization_range=4096,
        )

        assert config.strategy == "secaggplus"
        assert config.secagg_num_shares == 7
        assert config.secagg_reconstruction_threshold == 5
        assert config.secagg_max_weight == 65536
        assert config.secagg_timeout == 120.0
        assert config.secagg_clipping_range == 3.0
        assert config.secagg_quantization_range == 4096

    def test_fl_config_with_multi_machine_full_setup(self):
        """Test FLConfig with complete multi-machine setup."""
        server_config = ServerMachineConfig(
            host="server.example.com",
            username="serveruser",
            password="serverpass",
        )
        client_configs = [
            ClientMachineConfig(
                host="client1.example.com",
                username="client1user",
                partition_id=0,
            ),
            ClientMachineConfig(
                host="client2.example.com",
                username="client2user",
                partition_id=1,
            ),
        ]
        ssh_config = SSHConfig(timeout=120)

        multi_machine_config = MultiMachineConfig(
            server=server_config,
            clients=client_configs,
            project_dir="/shared/project",
            venv_path="/shared/.venv",
            venv_activate="source /shared/.venv/bin/activate",
            ssh=ssh_config,
        )

        config = FLConfig(
            num_rounds=100,
            strategy="fedavg",
            multi_machine=multi_machine_config,
        )

        assert config.multi_machine == multi_machine_config
        assert config.multi_machine.server == server_config
        assert len(config.multi_machine.clients) == 2
        assert config.multi_machine.ssh.timeout == 120

    def test_fl_config_client_config_files_validation(self):
        """Test FLConfig with client config files."""
        config = FLConfig(client_config_files=["configs/client1.yaml", "configs/client2.yaml", "configs/client3.yaml"])

        assert len(config.client_config_files) == 3
        assert "configs/client1.yaml" in config.client_config_files
        assert "configs/client2.yaml" in config.client_config_files
        assert "configs/client3.yaml" in config.client_config_files
