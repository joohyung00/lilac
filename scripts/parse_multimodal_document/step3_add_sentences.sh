eval "$(conda shell.bash hook)"
conda activate baseline

# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data MMWebQA   --mode extract_text
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data MMCoQA    --mode extract_text
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data SlideVQA  --mode extract_text
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data InfoVQA   --mode extract_text
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data MP-DocVQA --mode extract_text

# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data MMWebQA   --mode split_sentences
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data MMCoQA    --mode split_sentences
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data SlideVQA  --mode split_sentences
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data InfoVQA   --mode split_sentences
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data MP-DocVQA --mode split_sentences

# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data MMWebQA   --mode add_to_parsed_documents
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data MMCoQA    --mode add_to_parsed_documents
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data SlideVQA  --mode add_to_parsed_documents
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data InfoVQA   --mode add_to_parsed_documents
# python3 src/multimodal_document_manager/step3_add_sentences.py --target_data MP-DocVQA --mode add_to_parsed_documents


python3 src/multimodal_document_manager/step3_add_sentences.py --target_data MMCoQA    --mode embed --start_idx 27000 --end_idx 36000