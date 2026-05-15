eval "$(conda shell.bash hook)"
conda activate baseline

python3 src/experiment/evaluator/end_to_end_accuracy_evaluator.py