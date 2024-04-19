import subprocess


def run_shell_command(cmd: str) -> str:
    return subprocess.run(cmd, text=True, shell=True, check=True, capture_output=True).stdout
