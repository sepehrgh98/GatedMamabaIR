#!/bin/bash
# Submit the full job chain of one same-setup baseline (docs/BASELINES.md):
#   x2 500k train -> auto-resume (afternotok, only runs if the 7-day limit / a node kills it) -> x2 test
#                 -> x3 / x4 ft250k (start when x2 succeeds) -> x3 / x4 tests
# usage (from /speed-scratch/se_gham/GatedMambaIR): bash tests/launch_baselines.sh <Model> <account>
#   e.g. bash tests/launch_baselines.sh SwinIR swamy
# Prints one line per job id; paste into docs/BASELINES.md.
# If the x2 run finishes OK, cancel its resume job (afternotok is then never satisfied).
set -e
M=$1
ACC=${2:-omair}
TR=options/train/baselines
TE=options/test/baselines
TEST_GPU="--gres=gpu:1 --constraint=Model-V100"  # whole-image tests; 32GB
EXC="--exclude=speed-[01,05,17,38]"

sub() { sbatch --parsable -A "$ACC" $EXC "$@"; }

t2=$(sub --job-name=BL_${M}_x2 train.sh $TR/train_${M}_SR_x2_500k.yml)
r2=$(sub --job-name=BL_${M}_x2_resume --dependency=afternotok:$t2 train.sh $TR/train_${M}_SR_x2_500k.yml)
ok2="afterok:$t2?afterok:$r2"
e2=$(sub --job-name=BL_${M}_x2_test $TEST_GPU --dependency=$ok2 tests/test_cfg.sh $TE/test_${M}_SR_x2_500k.yml)
t3=$(sub --job-name=BL_${M}_x3 --dependency=$ok2 train.sh $TR/train_${M}_SR_x3_ft250k.yml)
t4=$(sub --job-name=BL_${M}_x4 --dependency=$ok2 train.sh $TR/train_${M}_SR_x4_ft250k.yml)
e3=$(sub --job-name=BL_${M}_x3_test $TEST_GPU --dependency=afterok:$t3 tests/test_cfg.sh $TE/test_${M}_SR_x3_ft250k.yml)
e4=$(sub --job-name=BL_${M}_x4_test $TEST_GPU --dependency=afterok:$t4 tests/test_cfg.sh $TE/test_${M}_SR_x4_ft250k.yml)
echo "$M account=$ACC x2=$t2 x2_resume=$r2 x2_test=$e2 x3=$t3 x4=$t4 x3_test=$e3 x4_test=$e4"
