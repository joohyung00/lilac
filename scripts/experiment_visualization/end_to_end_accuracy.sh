eval "$(conda shell.bash hook)"
conda activate baseline

python3 src_experiment/evaluator/end_to_end_accuracy_evaluator.py