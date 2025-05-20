import argparse
from typing import List, Dict, Any

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


QWEN_2_5_72B_PATH = "/root/LILaC/Models/Qwen2.5-72B-Instruct"

class Qwen2_5_72B:
    """
    Light wrapper around Qwen‑2.5‑72B‑Instruct with a batch‑friendly `infer`
    method.  API intentionally matches `Qwen2_5_VL` (no multimodal kwargs).
    """

    def __init__(
        self,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.9,
        system_prompt: str = "You are a helpful assistant.",
    ):
        # Chat template / tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_2_5_72B_PATH, trust_remote_code = True
        )
        # vLLM engine
        self.llm = LLM(
            model=QWEN_2_5_72B_PATH,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=30000,
            max_num_seqs=32,  # Reduce from default 256 to something manageable
        )
        self.system_prompt = system_prompt

    # ────────────────────────────────────────────────────────────────────────
    # Public inference API
    # -----------------------------------------------------------------------

    def infer(
        self,
        objects: List[Dict[str, Any]],
        *,
        max_tokens: int = 256,
        batch_size: int = 4,
        temperature: float = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
    ) -> List[str]:
        """
        Args
        ----
        objects : List[Dict]
            Each dict must have a `"text"` field with the prompt content.
            Any other keys are ignored.
        Returns
        -------
        List[str] – generated responses, same order as inputs.
        """
        self.sampling_params = SamplingParams(
            temperature = temperature,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            max_tokens = max_tokens
        )

        results: List[str] = []

        for start in tqdm(range(0, len(objects), batch_size), desc = "Inference Batches"):
            
            batch_objs = objects[start : start + batch_size]

            llm_inputs = []
            for obj in batch_objs:
                # Build standard chat message list
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": obj["text"]},
                ]
                # Convert to single prompt string via chat template
                prompt = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
                llm_inputs.append({"prompt": prompt})

            batch_outputs = self.llm.generate(llm_inputs, sampling_params = self.sampling_params)
            for out in batch_outputs:
                results.append(out.outputs[0].text)

        return results


# ───────────────────────────────────────────────────────────────────────────────
# CLI demo (optional)
# ───────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    
    
    qwen = Qwen2_5_72B()
    
    prompts = [
        {"text": "What is the capital of France?"},
        {"text": "What is the capital of Japan?"},
        {"text": "What is the capital of South Korea?"},
        {"text": "What is the capital of China?"},
        {"text": "What is the capital of the United States?"},
        {"text": "What is the capital of Canada?"},
        {"text": "What is the capital of Australia?"},
        {"text": "What is the capital of Germany?"},
        {"text": "What is the capital of Italy?"},
        {"text": "What is the capital of Spain?"},
    ] * 100

    outputs = qwen.infer(prompts, batch_size = 32, max_tokens = 50)

    for i, out in enumerate(outputs, 1):
        print(f"\n===== Output {i} =====\n{out}\n")

    pass