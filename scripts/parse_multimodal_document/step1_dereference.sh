eval "$(conda shell.bash hook)"
conda activate baseline

python3 src/multimodal_document_manager/step1_dereference.py --target_data MMWebQA
python3 src/multimodal_document_manager/step1_dereference.py --target_data MMCoQA