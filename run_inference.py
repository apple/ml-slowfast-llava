#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import subprocess
from argparse import ArgumentParser
from pathlib import Path
import yaml


def get_args():
    parser = ArgumentParser(description="SlowFast-LLaVA")
    parser.add_argument(
        "--exp_config",
        type=str,
        required=True,
        help="path to exp config file",
    )
    return parser.parse_args()


def main():
    # Load exp config
    args = get_args()
    with open(args.exp_config, "r") as f:
        exp_config = yaml.safe_load(f)
    if exp_config["CONFIG_NAME"] == "auto":
        exp_config["CONFIG_NAME"] = Path(args.exp_config).stem

    # Get commands
    commands = exp_config.pop("SCRIPT", None)
    if commands is None:
        raise RuntimeError("Script was not found in the config")
    if type(commands) is not list:
        commands = [commands]

    # Get parameters
    parameters = []
    for k, v in exp_config.items():
        if type(v) is not list:
            v = [v] * len(commands)
        else:
            assert len(v) == len(commands), \
                f"The number of parameters in {k} must match the number of SCRIPT"
        parameters.append((k, v))

    print(f":::: Start Inference ::::")

    # Iterate all scripts
    for idx, cmd in enumerate(commands):
        params = ""
        for k, v in parameters:
            params += f"{k}={v[idx]} "
        cmd = params + cmd

        # Run command
        subprocess.check_call(["bash", "-c", cmd])


if __name__ == "__main__":
    main()