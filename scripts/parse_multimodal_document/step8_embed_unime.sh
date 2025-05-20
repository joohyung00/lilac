
eval "$(conda shell.bash hook)"
conda activate unime



## MP-DocVQA

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/text.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/text.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/text.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/sentence.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/sentence.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/sentence.json \
    --num_gpus 4 \
    --max_length 128 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/table.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/table.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/table_segment+image.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/table_segment+image.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/table_segment+image.json \
    --num_gpus 4 \
    --max_length 128 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/image.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/image.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/image.json \
    --num_gpus 4 \
    --max_length 4096 \
    --batch_size 1

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/subimage.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/subimage.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/subimage.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/page.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/page.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/UniME/page.json \
    --num_gpus 4 \
    --max_length 4096 \
    --batch_size 1













# SlideVQA

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/text.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/UniME/text.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/UniME/text.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/sentence.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/UniME/sentence.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/UniME/sentence.json \
    --num_gpus 4 \
    --max_length 128 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/UniME/table.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/UniME/table.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/table_segment+image.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/UniME/table_segment+image.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/UniME/table_segment+image.json \
    --num_gpus 4 \
    --max_length 128 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/image.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/UniME/image.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/UniME/image.json \
    --num_gpus 4 \
    --max_length 4096 \
    --batch_size 1
    
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/subimage.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/UniME/subimage.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/UniME/subimage.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/page.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/UniME/page.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/UniME/page.json \
    --num_gpus 4 \
    --max_length 4096 \
    --batch_size 1





# InfoVQA

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/text.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/UniME/text.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/UniME/text.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/sentence.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/UniME/sentence.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/UniME/sentence.json \
    --num_gpus 4 \
    --max_length 128 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/UniME/table.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/UniME/table.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/table_segment+image.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/UniME/table_segment+image.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/UniME/table_segment+image.json \
    --num_gpus 4 \
    --max_length 128 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/image.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/UniME/image.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/UniME/image.json \
    --num_gpus 4 \
    --max_length 4096 \
    --batch_size 1

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/subimage.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/UniME/subimage.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/UniME/subimage.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/page.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/UniME/page.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/UniME/page.json \
    --num_gpus 4 \
    --max_length 4096 \
    --batch_size 1





## MMCoQA

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/text.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/UniME/text.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/UniME/text.json \
    --num_gpus 4 \
    --max_length 1024 \
    --batch_size 16

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/sentence.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/UniME/sentence.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/UniME/sentence.json \
    --num_gpus 4 \
    --max_length 128 \
    --batch_size 64

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/UniME/table.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/UniME/table.json \
    --num_gpus 4 \
    --max_length 2048 \
    --batch_size 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/UniME/table.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/UniME/table.json \
    --num_gpus 4 \
    --max_length 512 \
    --batch_size 10

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/image.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/UniME/image.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/UniME/image.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/subimage.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/UniME/subimage.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/UniME/subimage.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/page.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/UniME/page.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/UniME/page.json \
    --num_gpus 4 \
    --max_length 4096 \
    --batch_size 1




## MMWebQA

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/text.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/UniME/text.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/UniME/text.json \
    --num_gpus 4 \
    --max_length 1024 \
    --batch_size 16

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/sentence.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/UniME/sentence.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/UniME/sentence.json \
    --num_gpus 4 \
    --max_length 128 \
    --batch_size 64

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/UniME/table.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/UniME/table.json \
    --num_gpus 4 \
    --max_length 2048 \
    --batch_size 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/UniME/table.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/UniME/table.json \
    --num_gpus 4 \
    --max_length 512 \
    --batch_size 10

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/image.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/UniME/image.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/UniME/image.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/subimage.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/UniME/subimage.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/UniME/subimage.json \
    --num_gpus 4 \
    --max_length 256 \
    --batch_size 8

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_unime.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/page.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/UniME/page.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/UniME/page.json \
    --num_gpus 4 \
    --max_length 4096 \
    --batch_size 1