HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    nvidia/MM-Embed \
    --local-dir-use-symlinks False \
    --local-dir /root/LILaC/Models/MM-Embed \
    # --exclude *.pth

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    DeepGlint-AI/UniME-LLaVA-OneVision-7B \
    --local-dir-use-symlinks False \
    --local-dir /root/LILaC/Models/UniME \
    # --exclude *.pth

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    intfloat/mmE5-mllama-11b-instruct \
    --local-dir-use-symlinks False \
    --local-dir /root/LILaC/Models/MMe5 \
    # --exclude *.pth
    