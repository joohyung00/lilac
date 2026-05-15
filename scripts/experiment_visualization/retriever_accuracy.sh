eval "$(conda shell.bash hook)"
conda activate baseline

python3 src/experiment/experiments/retrieval_accuracy_experiment.py     --mode benchmark     --dataset MP-DocVQA
python3 src/experiment/experiments/retrieval_accuracy_experiment.py     --mode benchmark     --dataset SlideVQA
python3 src/experiment/experiments/retrieval_accuracy_experiment.py     --mode benchmark     --dataset InfoVQA
python3 src/experiment/experiments/retrieval_accuracy_experiment.py     --mode benchmark     --dataset MMCoQA
python3 src/experiment/experiments/retrieval_accuracy_experiment.py     --mode benchmark     --dataset MultimodalQA