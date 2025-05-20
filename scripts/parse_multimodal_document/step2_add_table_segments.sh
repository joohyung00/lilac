eval "$(conda shell.bash hook)"
conda activate baseline

# python3 src/multimodal_document_manager/step2_add_table_segments.py --target_data MMWebQA 
python3 src/multimodal_document_manager/step2_add_table_segments.py --target_data MMCoQA
python3 src/multimodal_document_manager/step2_add_table_segments.py --target_data InfoVQA
python3 src/multimodal_document_manager/step2_add_table_segments.py --target_data SlideVQA
python3 src/multimodal_document_manager/step2_add_table_segments.py --target_data MP-DocVQA