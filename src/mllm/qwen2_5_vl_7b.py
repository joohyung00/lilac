# qwen2_5_vl_no_vllm_single.py
import math
from typing import List, Dict

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info   # pip install qwen-vl-utils
import contextlib, traceback

QWEN_2_5_VL_PATH = "/root/LILaC/Models/Qwen2.5-VL-7B-Instruct"
# or simply "Qwen/Qwen2.5-VL-7B-Instruct" when loading from the Hub


class Qwen2_5_VL:
    """
    vLLM-free wrapper around Qwen2.5-VL-7B-Instruct that matches the original
    API but performs *one* generation per call (no internal batching).
    """

    def __init__(
        self,
        model_name_or_path: str = QWEN_2_5_VL_PATH,
        device: str = "cuda",
        limit_mm_per_prompt: Dict[str, int] = {"image": 8, "video": 0},
    ):
        self.device_requested = device
        self.limit_mm_per_prompt = limit_mm_per_prompt

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",         # HF will shard if several GPUs exist
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # choose one “main” device for small tensors
        if device == "cpu" or not torch.cuda.is_available():
            self._tensor_device = torch.device("cpu")
        else:
            first_cuda = next(
                (d for d in self.model.hf_device_map.values() if "cuda" in str(d)),
                "cuda:0",
            )
            self._tensor_device = torch.device(first_cuda)

    # ------------------------------------------------------------------ #
    # public API (unchanged)                                             #
    # ------------------------------------------------------------------ #
    def _obj_to_messages(
        self,
        obj: Dict
    ) -> List[Dict]:
        """
        Convert *either* (1) simple {text, images, videos} objects
        *or*      (2) already-formatted Qwen chat messages
        into the standard chat-message list required by `apply_chat_template`.
        """
        # Case 2 – already in chat-message shape
        if "role" in obj:
            return [obj] if isinstance(obj, dict) else obj

        # Case 1 – simple shape
        images = obj.get("images", [])[: self.limit_mm_per_prompt.get("image", math.inf)]
        videos = obj.get("videos", [])[: self.limit_mm_per_prompt.get("video", math.inf)]
        text   = obj.get("text", "")

        content = (
            [{"type": "image", "image": p}  for p in images] +
            [{"type": "video", "video": p}  for p in videos] +
            [{"type": "text",  "text":  text}]
        )
        return [{"role": "user", "content": content}]


    @torch.inference_mode()
    def infer(
        self,
        objects: List[Dict],
        batch_size: int = 1,        # ignored – kept for API compatibility
        temperature: float = 0.1,
        top_p: float = 0.001,
        repetition_penalty: float = 1.05,
        max_tokens: int = 256,
    ) -> List[str]:
        """
        Same behaviour as before, but each prompt is OOM-safe:
        on OOM ⇒ "" (empty answer) is appended.
        """
        answers: List[str] = []

        for obj in objects:
            messages = self._obj_to_messages(obj)

            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            img_inputs, vid_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_prompt],
                images=img_inputs,
                videos=vid_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._tensor_device)

            # ---------- OOM-safe call ------------------------------------
            out_ids = self._safe_generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if out_ids is None:                # hit OOM → empty answer
                answers.append("")
                continue

            new_tokens = out_ids[0, inputs.input_ids.shape[1]:]
            answers.append(
                self.processor.decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return answers

    def _safe_generate(self, inputs, **gen_kwargs):
        """
        Call self.model.generate() and catch CUDA OOM.
        On success  → tensor
        On OOM      → None
        """
        try:
            return self.model.generate(**inputs, **gen_kwargs)

        except torch.cuda.OutOfMemoryError:
            pass
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise

        # ---- OOM path ----------------------------------------------------
        print("[warn] CUDA OOM during generation → returning empty answer")
        traceback.print_exc(limit=1)
        torch.cuda.empty_cache()
        return None          # ← was ""  (string).  Must be None.

# ---------------------------------------------------------------------- #
# quick sanity-check                                                     #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    qwen = Qwen2_5_VL(
        device="cuda" if torch.cuda.is_available() else "cpu",
        limit_mm_per_prompt={"image": 8, "video": 0},       # keep at most 4 imgs
    )

    # ---------------------------------------------------------------------
    # 2. Build a *simple-object* prompt
    # ---------------------------------------------------------------------
    simple_prompt = {
        "text": (
            "You are shown a picture. "
            "Briefly describe what this picture is."
        ),
        "images": [
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            "/root/LILaC/%2B_908_wurde_St_Andrews_bereits_Bischohfssitz._13_%28cropped%29.png",
            
        ]
    }

    # ---------------------------------------------------------------------
    # 3. Run inference — note: we still pass a *list* of objects
    # ---------------------------------------------------------------------
    answer = qwen.infer(
        objects=[simple_prompt],   # <- list of 1 simple object
        temperature=0.0,
        top_p=0.9,
        max_tokens=32,
    )[0]

    print("🪄 Generated answer:\n", answer)
    
    answer = qwen.infer(
        objects=[simple_prompt],   # <- list of 1 simple object
        temperature=0.0,
        top_p=0.9,
        max_tokens=32,
    )[0]

    print("🪄 Generated answer:\n", answer)
    
    answer = qwen.infer(
        objects=[simple_prompt],   # <- list of 1 simple object
        temperature=0.0,
        top_p=0.9,
        max_tokens=32,
    )[0]

    print("🪄 Generated answer:\n", answer)
    
    answer = qwen.infer(
        objects=[simple_prompt],   # <- list of 1 simple object
        temperature=0.0,
        top_p=0.9,
        max_tokens=32,
    )[0]

    print("🪄 Generated answer:\n", answer)

