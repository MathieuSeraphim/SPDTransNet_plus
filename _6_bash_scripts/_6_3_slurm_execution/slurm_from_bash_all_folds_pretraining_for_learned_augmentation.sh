#!/bin/bash

CURRENT_SCRIPT_DIRECTORY=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
slurm_script=slurm_all_folds_pretraining_for_learned_augmentation.sl
run_script="$CURRENT_SCRIPT_DIRECTORY"/_bash_run_any_slurm_script.sh

bash "$run_script" $slurm_script




