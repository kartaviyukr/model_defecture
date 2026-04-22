#!/usr/bin/env bash
# Run the test suite.
# pytest is installed via uv tools; yaml is from system dist-packages.
PYTHONPATH="${PWD}:/root/.local/share/uv/tools/pytest/lib/python3.11/site-packages:/usr/lib/python3/dist-packages" \
    /root/.local/share/uv/tools/pytest/bin/pytest "$@"
