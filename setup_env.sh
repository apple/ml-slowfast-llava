#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
#!/bin/bash

pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir

apt-get update
apt-get install git-lfs
git-lfs install
