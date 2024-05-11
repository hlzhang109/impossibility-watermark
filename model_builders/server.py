import os
import time
import requests
import subprocess
import logging
import signal

from langchain_community.llms import VLLMOpenAI

log = logging.getLogger(__name__)

class ServerBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=self.cfg.base_url,
            model_name=self.cfg.model_name_or_path,
            presence_penalty=self.cfg.repetition_penalty,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
        )
        self.cuda_device = cfg.cuda
                    
    def acquire_lock(self):
        """Attempt to acquire the lock by creating a lock file. Returns True if successful, False otherwise."""
        if os.path.exists(self.cfg.lock_file_path):
            return False
        else:
            with open(self.cfg.lock_file_path, 'w') as f:
                f.write("locked")
            return True

    def release_lock(self):
        """Release the lock by deleting the lock file."""
        if os.path.exists(self.cfg.lock_file_path):
            os.remove(self.cfg.lock_file_path)

    def wait_for_lock(self):
        """Wait for the lock to be released."""
        while os.path.exists(self.cfg.lock_file_path):
            time.sleep(1)

    def start_server(self):
        """Start the server."""
        server_command = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.cfg.model_name_or_path,
            "--host", self.cfg.vllm_host,
            "--port", str(self.cfg.vllm_port),
            "--download-dir", self.cfg.model_cache_dir,
            "--dtype", "float16",
        ]

        if self.cfg.trust_remote_code:
            server_command.extend(["--trust-remote-code"])

        if "AWQ" in self.cfg.model_name_or_path:
            server_command.extend(["--quantization", "awq"])

        if "GPTQ" in self.cfg.model_name_or_path:
            server_command.extend(["--quantization", "gptq"])
            
        log.info(f"Initializing vLLM api server:\n{server_command}")
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = f"{self.cuda_device}"
        self.server_process = subprocess.Popen(server_command, env=env)
        self.server_pid = self.server_process.pid  # Store the server's PID
        log.info(f"Server process started with PID: {self.server_pid}")
        self.wait_for_server_to_load()
        log.info("Server started!")

    def kill_server(self):
        """Method to kill the server process using its PID."""
        if self.server_pid:
            os.kill(self.server_pid, signal.SIGTERM)
            log.info(f"Server process with PID {self.server_pid} has been terminated.")
            self.server_pid = None

    def is_server_running(self):
        """Check if the model server is running."""
        try:
            response = requests.get(self.cfg.base_url + "/models")
            response.raise_for_status()  # Raises an error for bad responses
            return True
        except requests.exceptions.RequestException:
            return False

    def wait_for_server_to_load(self):
        """Wait for the model to load into memory."""
        while not self.is_server_running():
            time.sleep(1)

    def ensure_server_running(self):
        """Ensure the server is running, starting it if necessary."""
        if not self.is_server_running():
            if self.acquire_lock():
                try:
                    self.start_server()
                finally:
                    self.release_lock()
            else:
                log.info("Server is starting up by another process. Waiting...")
                self.wait_for_lock()
                log.info("Server started by another process.")

    def __call__(self, prompt):
        """Invoke the model with the provided prompt."""
        self.ensure_server_running()
        return self.client.invoke(prompt)

if __name__ == "__main__": 

    import hydra
    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def main(cfg):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.attack_args.cuda)
        os.environ["WORLD_SIZE"] = str(len(str(cfg.attack_args.cuda).split(",")))
        server = ServerBuilder(cfg.generator_args)
        try:
            print(server("Rome is"))
        finally:
            server.kill_server()

    main()