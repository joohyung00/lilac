import os
import json
from typing import List, Sequence, Union

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

from src.embedder.embedder import Embedder

# -------------------------------------------------------------------------
#  Constants
# -------------------------------------------------------------------------
UNI_ME_PATH = "/root/LILaC/Models/UniME-LLaVA-OneVision-7B"


class UniME(Embedder):
    """Wrapper that exposes the same API as :class:`MMEmbed` for UniME.

    *Only one‑by‑one inference is implemented.*  Add real batching later if you
    need higher throughput – the `batch_size` parameters are kept solely for
    call‑site compatibility.
    """

    def __init__(self, cuda_num: int, show_progress: bool = True):
        self.cuda_num = cuda_num
        self.show_progress = show_progress
        self.model_path = UNI_ME_PATH

        self.processor = None  # Will be set in `load_model()`
        self.model = None

    # ------------------------------------------------------------------
    #  Model lifecycle
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load processor + UniME LLaVA‑OneVision model onto the given GPU."""
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map={"": f"cuda:{self.cuda_num}"},  # all on a single GPU
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Ensure left-padding so the last token is the EOS-like summary token
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.padding = True

        self.model.eval()
        print("UniME model loaded successfully.")

    def delete_model(self) -> None:
        """Free CUDA memory and processor handles."""
        self.processor = None
        self.model = None
        torch.cuda.empty_cache()
        print("UniME model deleted successfully.")

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chat_template(*, text: str | None = None, image=None):
        """Build a chat‑template conversation accepted by `apply_chat_template()`.

        Supports **three** scenarios:
        1. *image only*   –  user supplies an image.
        2. *text only*    –  plain sentence/paragraph.
        3. *image + text* –  both provided; ask the model to consider *both*.
        """
        if image is not None and text is not None:
            # image + supplemental text
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": f"{text}\nSummary above image and the following text in one word:\n",
                        },
                    ],
                }
            ]
        elif image is not None:
            # image‑only prompt
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Summary above image in one word:\n"},
                    ],
                }
            ]
        else:
            # text‑only prompt
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{text}\nSummary above sentence in one word:\n",
                        }
                    ],
                }
            ]

    # ------------------------------------------------------------------
    #  Query encoding (text‑only, one‑by‑one)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_queries(
        self,
        queries: List[str],
        output_filepath: str | None = None,
        instruction: Union[str, Sequence[str]] | None = None,
        batch_size: int = 4,  # kept for API compatibility (ignored)
        max_length: int = 4096,
    ) -> torch.Tensor:
        """Return a `(N × hidden_size)` *CPU* tensor of embeddings."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Call `load_model()` before `encode_queries()`")

        # Normalise instruction list (unused by UniME)
        if isinstance(instruction, str) or instruction is None:
            instruction = [None] * len(queries)
        elif len(instruction) != len(queries):
            raise ValueError("`instruction` length must match `queries` length")

        embeddings: List[torch.Tensor] = []
        iterator = tqdm(range(len(queries)), desc="UniME‑EncodeQ", disable=not self.show_progress)

        for i in iterator:
            q = queries[i]
            # Build text‑only prompt
            conv = self._chat_template(text=q)

            # Prepare model inputs
            inputs = self.processor.apply_chat_template(
                [conv],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,    # ensure mapping
                return_tensors="pt",
                padding=True,
            ).to(f"cuda:{self.cuda_num}")

            try:
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[-1][:, -1, :]  # (1 × H)
                emb = F.normalize(hidden, dim=-1).squeeze(0).cpu()
            except Exception as e:
                print(f"⚠️ UniME failed to encode query {i}: {e}")
                # fallback to model's embed_dim or LM hidden size
                fallback_dim = getattr(self.model, "embed_dim", None)
                if fallback_dim is None:
                    fallback_dim = getattr(
                        self.model.language_model.config, "hidden_size", 0
                    )
                emb = torch.zeros(fallback_dim, dtype=torch.float32)

            embeddings.append(emb)

        tensor = torch.stack(embeddings, dim=0)

        if output_filepath:
            torch.save(tensor, output_filepath)
            print(f"💾 Saved UniME query embeddings → {output_filepath}  shape={tensor.shape}")

        return tensor

    # ------------------------------------------------------------------
    #  Corpus encoding (text ± image, one‑by‑one)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_corpus(
        self,
        data: list[dict],
        out_embeddings_path: str,
        out_index_path: str,
        start_idx: int | None = None,
        end_idx: int | None = None,
        batch_size: int = 1,  # not implemented
        max_length: int = 4096,
    ) -> torch.Tensor:
        if self.model is None or self.processor is None:
            raise RuntimeError("Call `load_model()` before `encode_corpus()`")

        if start_idx is None:
            start_idx = 0
        if end_idx is None or end_idx > len(data):
            end_idx = len(data)

        embeddings: List[torch.Tensor] = []
        idx_to_data_id: dict[int, str] = {}

        iterator = tqdm(range(start_idx, end_idx), desc="UniME‑EncodeCorpus", disable=not self.show_progress)

        for idx in iterator:
            item = data[idx]
            data_id = item["id"]
            text = item["target"].get("text", "").strip()
            image_paths = item["target"].get("images", [])

            # Try first loadable image (if any)
            image = None
            for p in image_paths:
                try:
                    image = Image.open(p).convert("RGB")
                    break
                except Exception as e:
                    print(f"⚠️ Unable to open image '{p}': {e}")

            # Build prompt depending on availability of text/image
            if image is not None and text:
                conv = self._chat_template(text=text, image=image)
            elif image is not None:
                conv = self._chat_template(image=image)
            else:
                conv = self._chat_template(text=text)

            # Prepare model inputs
            inputs = self.processor.apply_chat_template(
                [conv],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(f"cuda:{self.cuda_num}")

            try:
                out = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = out.hidden_states[-1][:, -1, :]
                emb = F.normalize(hidden, dim=-1).squeeze(0).cpu()
            except Exception as e:
                print(f"⚠️ encode_corpus failed at idx={idx}: {e}")
                # fallback to model's embed_dim or LM hidden size
                fallback_dim = getattr(self.model, "embed_dim", None)
                if fallback_dim is None:
                    fallback_dim = getattr(
                        self.model.language_model.config, "hidden_size", 0
                    )
                emb = torch.zeros(fallback_dim, dtype=torch.float32)

            embeddings.append(emb)
            idx_to_data_id[len(embeddings) - 1] = data_id

        # Final tensor
        tensor = torch.stack(embeddings, dim=0)

        torch.save(tensor, out_embeddings_path)
        with open(out_index_path, "w", encoding="utf-8") as f:
            json.dump(idx_to_data_id, f, indent=2)

        print(
            f"💾 Saved {tensor.shape[0]} embeddings → {out_embeddings_path}; index → {out_index_path}"
        )
        return tensor


# -------------------------------------------------------------------------
#  Testing (optional direct run)
# -------------------------------------------------------------------------
if __name__ == "__main__":

    import torch
    from PIL import Image
    from torch.nn import functional as F
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    def appply_chat_template(image=None, text=None):
        """Minimal direct-check function (not used by the class)."""
        conversation_image = []
        if image is not None:
            conversation_image = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Summary above image in one word:\n"},
                    ],
                }
            ]
        elif text is not None:
            conversation_image = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{text}\nSummary above sentence in one word:\n",
                        }
                    ],
                }
            ]
        return conversation_image

    base_model_path = UNI_ME_PATH

    text = "A man is crossing the street with a red car parked nearby."
    image_path = "/root/LILaC/cat_dog.webp"
    test_image = Image.open(image_path).convert("RGB")

    # Basic test
    transform = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        base_model_path,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    transform.tokenizer.padding_side = "left"
    transform.tokenizer.padding = True

    inputs_text = transform.apply_chat_template(
        [appply_chat_template(text=text)],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    inputs_image = transform.apply_chat_template(
        [appply_chat_template(image=test_image)],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    with torch.no_grad():
        emb_text = model(**inputs_text, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        emb_image = model(**inputs_image, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        emb_text = F.normalize(emb_text, dim=-1)
        emb_image = F.normalize(emb_image, dim=-1)
        score = (emb_image @ emb_text.T).item()
        print("Sample similarity score (image vs text):", score)
