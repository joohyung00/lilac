HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir-use-symlinks False \
    --local-dir /root/LILaC/Models/Qwen2.5-VL-7B-Instruct \
    # --exclude *.pth