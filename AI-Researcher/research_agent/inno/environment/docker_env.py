import os
import os.path as osp
import subprocess
from research_agent.constant import BASE_IMAGES, AI_USER, GITHUB_AI_TOKEN, GPUS, PLATFORM
import time
import socket
import json
from pathlib import Path
import shutil
wd = Path(__file__).parent.resolve()
from dataclasses import dataclass, field
from typing import Optional, Union, Dict
from functools import update_wrapper
from inspect import signature
from integration_helper import get_directory_manager

@dataclass
class DockerConfig: 
    container_name: str
    workplace_name: str 
    communication_port: int # 12345
    test_pull_name: str = field(default='main')
    task_name: Optional[str] = field(default=None)
    git_clone: bool = field(default=False)
    setup_package: Optional[str] = field(default=None)
    local_root: str = field(default=os.getcwd())
    singularity_image: str = field(default='/nfs/home/<Username>/Projects/phd/airesearcher.sif')
    

class DockerEnv:
    def __init__(self, config: Union[DockerConfig, Dict]):
        if isinstance(config, Dict):
            config = DockerConfig(**config)
        self.workplace_name = config.workplace_name
        manager = get_directory_manager()
        if manager.current_run_dir:
            # Use centralized directory
            self.local_workplace = str(manager.get_workplace_dir())
        else:
            # Fallback to original behavior
            self.local_workplace = osp.join(config.local_root, config.workplace_name)
        self.docker_workplace = f"/{config.workplace_name}"
        self.container_name = config.container_name
        self.test_pull_name = config.test_pull_name
        self.task_name = config.task_name
        self.git_clone = config.git_clone
        self.setup_package = config.setup_package
        self.communication_port = config.communication_port
        self.singularity_image = config.singularity_image
    def init_container(self):
        """Initialize Singularity instance (replaces Docker container initialization)"""
        # Check if instance already exists
        instance_exists = self._check_instance_exists()
        os.makedirs(self.local_workplace, exist_ok=True)
        
        if self.setup_package is not None:
            unzip_command = ["tar", "-xzvf", f"packages/{self.setup_package}.tar.gz", "-C", self.local_workplace]
            subprocess.run(unzip_command)
            
        if self.git_clone:
            if not os.path.exists(os.path.join(self.local_workplace, 'metachain')):
                git_command = ["cd", self.local_workplace, "&&", "git", "clone", "-b", self.test_pull_name, f"https://{AI_USER}:{GITHUB_AI_TOKEN}@github.com/tjb-tech/metachain.git"]
                git_command = " ".join(git_command)
                
                result = subprocess.run(git_command, shell=True)
                if result.returncode != 0:
                    raise Exception(f"Failed to clone the repository. Please check your internet connection and try again.")
                    
            # create a new branch
            new_branch_name = f"{self.test_pull_name}_{self.task_name}"
            create_branch_command = f"cd {self.local_workplace}/metachain && git checkout -b {new_branch_name}"
            result = subprocess.run(create_branch_command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(Exception(f"Failed to create and switch to new branch. Error: {result.stderr}"))
                switch_branch_command = f"cd {self.local_workplace}/metachain && git checkout {new_branch_name}"
                result = subprocess.run(switch_branch_command, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f"Failed to switch to new branch. Error: {result.stderr}")
                else:
                    print(f"Successfully switched to new branch: {new_branch_name}")
            else:
                print(f"Successfully created and switched to new branch: {new_branch_name}")

        if instance_exists:
            print(f"Singularity instance '{self.container_name}' is already running. Skipping creation.")
            return
        
        # Start new Singularity instance
        print(f"Starting Singularity instance '{self.container_name}'...")
        singularity_command = [
            "singularity", "instance", "start",
            "--nv",
            "--env", "SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt",  # Changed path
            "--env", "REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt",  # Changed path
            "--bind", f"{self.local_workplace}:{self.docker_workplace}",
            "--bind", "/etc/pki:/etc/pki:ro",  # Mount entire PKI directory
            "--bind", "/etc/ssl:/etc/ssl:ro",  # Keep this too for other tools
            self.singularity_image,
            self.container_name
        ]

        print(" ".join(singularity_command))
        result = subprocess.run(singularity_command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to start Singularity instance: {result.stderr}")
            
        # Wait for instance to be ready and start TCP server
        if self.wait_for_container_ready(timeout=60):
            print(f"Singularity instance '{self.container_name}' has been created and started.")

            git_ssl_config = """
git config --global http.sslCAInfo /etc/pki/tls/certs/ca-bundle.crt
git config --global http.sslVerify true
"""
            result = subprocess.run(
                ["singularity", "exec", f"instance://{self.container_name}", "bash", "-c", git_ssl_config],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✓ Configured git SSL certificates")
            else:
                print(f"⚠ Warning: Could not configure git SSL: {result.stderr}")

            
            # Configure git with GitHub token for authentication
            github_token = os.getenv('GITHUB_AI_TOKEN', '')
            if github_token:
                # Disable VSCode credential helper and configure token auth
                git_config_cmds = f"""
git config --global --unset-all credential.helper 2>/dev/null || true
git config --global credential.helper '!f() {{ echo "username=token"; echo "password={github_token}"; }}; f'
"""
                result = subprocess.run(
                    ["singularity", "exec", f"instance://{self.container_name}", "bash", "-c", git_config_cmds],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"✓ Configured git authentication in container")
                else:
                    print(f"⚠ Warning: Could not configure git auth: {result.stderr}")
            else:
                print(f"⚠ Warning: GITHUB_TOKEN not set in environment, git clones may fail")

    
    def _check_instance_exists(self):
        """Check if Singularity instance exists"""
        result = subprocess.run(
            ["singularity", "instance", "list"],
            capture_output=True,
            text=True
        )
        return self.container_name in result.stdout
    
    def wait_for_container_ready(self, timeout=30):
        """Wait for Singularity instance to be ready and TCP server to start"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if instance is running
            if not self._check_instance_exists():
                time.sleep(1)
                continue
            
            # Start TCP server inside the instance
            try:
                # Start the TCP server in background using setsid (creates new session, prevents hanging)
                start_server_cmd = f"setsid python /app/tcp_server.py > /tmp/tcp_server.log 2>&1 < /dev/null &"
                result = subprocess.run(
                    ["singularity", "exec", f"instance://{self.container_name}", "bash", "-c", start_server_cmd],
                    capture_output=True,
                    text=True,
                    timeout=5  # Should return almost immediately
                )
                
                # Give server time to start
                time.sleep(2)
                
                # Test if TCP server is actually responding by connecting to it
                try:
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(2)
                    test_socket.connect(('localhost', self.communication_port))
                    test_socket.close()
                    print(f"TCP server started successfully on port {self.communication_port}")
                    return True
                except (ConnectionRefusedError, socket.timeout, OSError) as e:
                    print(f"Connection test failed: {e}")
                    # Server not ready yet, continue waiting
                    pass
                    
            except Exception as e:
                print(f"Waiting for TCP server to start: {e}")
                
            time.sleep(1)
            
        raise TimeoutError(f"Singularity instance {self.container_name} failed to start within {timeout} seconds")
    
    def stop_container(self):
        """Stop Singularity instance"""
        if not self._check_instance_exists():
            print(f"Instance '{self.container_name}' is not running.")
            return
            
        result = subprocess.run(
            ["singularity", "instance", "stop", self.container_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to stop Singularity instance: {result.stderr}")
        else:
            print(f"Successfully stopped instance '{self.container_name}'")
    
    def run_command(self, command, stream_callback=None):
        """
        Communicate with Singularity instance and execute command, support stream output
        
        Args:
            command: the command to execute
            stream_callback: optional callback function, for handling stream output
                            the function signature should be callback(text: str)
        
        Returns:
            dict: the complete JSON result returned by the container
        """
        hostname = 'localhost'
        port = self.communication_port
        buffer_size = 4096
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((hostname, port))
            s.sendall(command.encode())
            
            partial_line = ""
            while True:
                chunk = s.recv(buffer_size)
                if not chunk:
                    break
                    
                # add new received data to the unfinished data
                data = partial_line + chunk.decode('utf-8')
                lines = data.split('\n')
                
                # except the last line, process all complete lines
                for line in lines[:-1]:
                    if line:
                        try:
                            response = json.loads(line)
                            if response['type'] == 'chunk':
                                # process stream output
                                if stream_callback:
                                    stream_callback(response['data'])
                            elif response['type'] == 'final':
                                # return the final result
                                return {
                                    'status': response['status'],
                                    'result': response['result']
                                }
                        except json.JSONDecodeError:
                            print(f"Invalid JSON: {line}")
                
                # save the possibly unfinished last line
                partial_line = lines[-1]
                
        # if the loop ends normally without receiving a final response
        return {
            'status': -1,
            'result': 'Connection closed without final response'
        }
    
def with_env(env: DockerEnv):
    """将env注入到工具函数中的装饰器"""
    def decorator(func):
        def wrapped(*args, **kwargs):
            return func(env=env, *args, **kwargs)
        
        # 保留原始函数的所有属性
        update_wrapper(wrapped, func)
        # 修改signature，移除env参数
        wrapped.__signature__ = signature(func).replace(
            parameters=[p for p in signature(func).parameters.values() if p.name != 'env']
        )
        if func.__doc__:
            try:
                if '{docker_workplace}' in func.__doc__:
                    wrapped.__doc__ = func.__doc__.format(docker_workplace=env.docker_workplace)
                else:
                    wrapped.__doc__ = func.__doc__
                if '{local_workplace}' in func.__doc__:
                    wrapped.__doc__ = func.__doc__.format(local_workplace=env.local_workplace)
                else:
                    wrapped.__doc__ = func.__doc__
            except (KeyError, IndexError, ValueError):
                # 如果格式化失败（没有占位符），保持原始文档
                wrapped.__doc__ = func.__doc__
        return wrapped
    return decorator

def check_container_ports(container_name: str):
    """
    Check Singularity instance status (compatibility function for Docker version)
    Returns None if instance doesn't exist, or (port, port) tuple if it does
    """
    result = subprocess.run(
        ["singularity", "instance", "list"],
        capture_output=True,
        text=True
    )
    
    if container_name in result.stdout:
        # Singularity doesn't do port mapping like Docker, but return consistent format
        # Default port is 8000 inside the container
        return (8000, 8000)
    return None

def check_container_exist(container_name: str):
    """Check if Singularity instance exists"""
    result = subprocess.run(
        ["singularity", "instance", "list"],
        capture_output=True,
        text=True
    )
    return container_name in result.stdout

def check_container_running(container_name: str):
    """Check if Singularity instance is running (same as exists for Singularity)"""
    return check_container_exist(container_name)