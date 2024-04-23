import os
import subprocess

from pathlib import Path

from loguru import logger

from config import cfg
from src.utils.run_command import run_shell_command


def is_dvc_initialized() -> bool:
    return (Path().cwd() / ".dvc").exists()


def initialize_dvc() -> None:
    if is_dvc_initialized():
        logger.info("DVC is already initialized")
        return None
    logger.info("Initializing DVC...")
    # try:
    #     run_shell_command("git checkout -b dev")
    # except subprocess.CalledProcessError:
    #     run_shell_command("git checkout dev")
    run_shell_command("poetry run dvc init --subdir")
    run_shell_command("poetry run dvc config core.analytics false")
    run_shell_command("poetry run dvc config core.autostage true")
    run_shell_command("git add .dvc")
    run_shell_command("git commit -m 'feat: Initialize DVC'")


def initialize_dvc_storage(dvc_remote_name: str, dvc_remote_url: str) -> None:
    if not run_shell_command("poetry run dvc remote list"):
        logger.info("Initialize DVC storage...")
        run_shell_command(f"poetry run dvc remote add -d {dvc_remote_name} {dvc_remote_url}")
        run_shell_command("git add .dvc/config")
        run_shell_command(f"git commit -m 'Configure remote storage at: {dvc_remote_url}'")
    else:
        logger.info("DVC storage was already initialized.")


def get_current_data_version() -> str:
    current_version = run_shell_command(
        "git tag --list | sort -t v -k 2 -g | tail -1 | sed 's/v//'"
    )
    return current_version


def get_gitignore_file_created_by_dvc(raw_data_folder: str) -> str:
    seperator = os.sep
    return os.path.join(seperator.join(raw_data_folder.split(seperator)[:-1]), ".gitignore")


def commit_new_data_version_to_dvc(raw_data_folder: str, dvc_remote_name: str) -> None:
    current_version = get_current_data_version().strip()
    if not current_version:
        current_version = "0"
    next_version = f"v{int(current_version)+1}"
    logger.info("Add data to dvc")
    run_shell_command(f"poetry run dvc add {raw_data_folder}")

    gitignore_file = get_gitignore_file_created_by_dvc(raw_data_folder)
    run_shell_command(f"git add {raw_data_folder}.dvc {gitignore_file}")
    run_shell_command(
        f"git commit -m 'Updated data version from v{current_version} to {next_version}'"
    )
    run_shell_command(f"git tag -a {next_version} -m 'Versioning data: {next_version}'")
    logger.info(f"Push data version {next_version} to {dvc_remote_name}")
    run_shell_command(f"poetry run dvc push {raw_data_folder}.dvc --remote {dvc_remote_name}")
    run_shell_command("git push --follow-tags origin dev")
    run_shell_command("git push -f --tags origin dev")


def update_data_version(raw_data_folder: str, dvc_remote_name: str) -> None:
    try:
        dvc_status = run_shell_command(f"poetry run dvc status {raw_data_folder}.dvc")
        if dvc_status == "Data and pipelines are up to date.\n":
            logger.info(dvc_status)
            return None
        commit_new_data_version_to_dvc(raw_data_folder, dvc_remote_name)
    except subprocess.CalledProcessError:
        logger.info("Versioning data the first time.")
        commit_new_data_version_to_dvc(raw_data_folder, dvc_remote_name)


if __name__ == "__main__":
    initialize_dvc()
    initialize_dvc_storage(
        dvc_remote_name=cfg.dvc_remote_name,
        dvc_remote_url=cfg.dvc_remote_url,
    )
    object_version = os.path.join(cfg.data_root_path, cfg.label_subset_file)
    update_data_version(
        raw_data_folder=object_version,
        dvc_remote_name=cfg.dvc_remote_name,
    )
