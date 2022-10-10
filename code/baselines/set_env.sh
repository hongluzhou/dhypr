#!/usr/bin/env bash
export DHYPR_HOME=$(pwd)
export LOG_DIR="$DHYPR_HOME/logs"
export PYTHONPATH="$DHYPR_HOME:$PYTHONPATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
source activate dhypr