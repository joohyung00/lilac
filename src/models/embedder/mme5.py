import os
import json
import torch
from PIL import Image
from typing import List, Union, Sequence, Dict, Any

from tqdm import tqdm


from src.models.embedder.embedder import Embedder
from src.utils.utils import REPO_ROOT
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
)

# You might already have a base Embedder class; import it here:
# from src.models.embedder.embedder import Embedder

################################################################################
# Some constants or strings:
################################################################################
MODALITY_AGNOSTIC_MME5_RETRIEVAL_INSTRUCTION = (
    "Represent the given text or image in a shared embedding space."
)
# Adjust model path to your local location:
MME5_PATH = f"{REPO_ROOT}/models/mmE5-mllama-11b-instruct"


################################################################################
# Utility: final pooling
################################################################################
def last_pooling(last_hidden_state: torch.Tensor,
                 attention_mask: torch.Tensor,
                 normalize: bool = True) -> torch.Tensor:
    """
    Extract a single embedding from the last hidden state by taking the hidden
    vector at the position of the last valid token per sequence, then optionally
    L2-normalize across the embedding dimension.
    """
    # For each sequence, find the index of its last non‑padding token:
    sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
    batch_size = last_hidden_state.shape[0]

    # Gather the embedding at that index
    reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device),
                             sequence_lengths]
    if normalize:
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps






class MME5(Embedder): 
    def __init__(
        self,
        cuda_num: int,
        show_progress: bool = True
    ):
        """
        Args:
            cuda_num: Which GPU device index to load the model onto.
            show_progress: Whether to display progress bars (tqdm).
        """
        self.model_path = MME5_PATH
        self.cuda_num = cuda_num
        self.show_progress = show_progress

        self.model = None
        self.processor = None

    def load_model(self):
        """
        Load MME5 model + processor onto the specified device.
        """
        print("Loading MME5 model from:", self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        self.model.to(f"cuda:{self.cuda_num}")
        self.model.eval()
        print("MME5 model loaded successfully.")

    def delete_model(self):
        """
        Delete model from memory by setting references to None.
        """
        self.model = None
        self.processor = None
        print("Model deleted successfully.")

    @torch.no_grad()
    def encode_queries(
        self,
        queries: List[str],
        output_filepath: Union[str, None] = None,
        instruction: Union[str, Sequence[str]] = MODALITY_AGNOSTIC_MME5_RETRIEVAL_INSTRUCTION,
        batch_size: int = 4,  # Kept for API compatibility; not used here.
        max_length: int = 256
    ) -> torch.Tensor:
        """
        Encode text-only queries, returning a CPU tensor of shape [len(queries), hidden_dim].
        
        Args:
            queries: List of text queries (strings).
            output_filepath: Path to save the tensor (via torch.save) if provided.
            instruction: Either a single string or list of strings (same length as queries).
                         It will be prepended to each query text.
            batch_size: Not used (we do one-by-one).
            max_length: Maximum sequence length for the model.
        
        Returns:
            A torch.Tensor on CPU of shape [len(queries), hidden_dim].
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load_model() before encode_queries().")

        # Normalize instructions to a list
        if isinstance(instruction, str):
            instruction_list = [instruction] * len(queries)
        else:
            if len(instruction) != len(queries):
                raise ValueError("instruction length must match queries length.")
            instruction_list = list(instruction)

        embeddings = []
        device = f"cuda:{self.cuda_num}"

        # For each query, run model inference one by one
        iterator = range(len(queries))
        if self.show_progress:
            iterator = tqdm(iterator, desc="EncodeQ-one-by-one (MME5)")

        for i in iterator:
            text = queries[i]
            instr = instruction_list[i]

            # Combine the instruction + text for MME5
            # For text-only: just "instr + text"
            prompt = f"{instr}\n{text}"

            # Prepare processor inputs
            proc_inp = self.processor(
                text=prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            )
            proc_inp = {k: v.to(device) for k, v in proc_inp.items()}

            # Forward through model
            try:
                out = self.model(
                    **proc_inp,
                    return_dict=True,
                    output_hidden_states=True
                )
                hidden = out.hidden_states[-1]  # last hidden state
                # pool to single vector
                emb = last_pooling(hidden, proc_inp["attention_mask"])
                emb = emb.squeeze(0).cpu()
            except Exception as e:
                print(f"❗ Encoding failed for query index {i}: {e} — using zeros.")
                # If there's an error, produce a zero vector
                # Attempt dimension from hidden size if known, else pick large guess
                emb_size = self.model.config.hidden_size
                emb = torch.zeros(emb_size, dtype=torch.float32)

            embeddings.append(emb)

        tensor = torch.stack(embeddings, dim=0)

        if output_filepath:
            torch.save(tensor, output_filepath)
            print(f"Saved embeddings to {output_filepath} [shape={tensor.shape}]")

        return tensor

    @torch.no_grad()
    def encode_corpus(
        self,
        data: List[Dict[str, Any]],
        out_embeddings_path: str,
        out_index_path: str,
        start_idx: Union[int, None],
        end_idx: Union[int, None],
        batch_size: int = 4,  # not used, but kept for signature compatibility
        max_length: int = 256
    ) -> torch.Tensor:
        """
        Encode a corpus of items (each possibly containing text and images)
        into a single 2D tensor, and write an index JSON file mapping row→doc_id.

        Args:
            data: A list of dicts, each with structure:
                  {
                    "id": str,
                    "target": {
                       "text": "...",
                       "images": [list of image paths, can be empty or more than one]
                    }
                  }
            out_embeddings_path: Path where the .pt file of embeddings is saved.
            out_index_path: Path where the .json index is saved.
            start_idx: If not None, the starting offset in `data` to encode.
            end_idx:   If not None, the last offset (exclusive) in `data`.
            batch_size: Unused; we do one-by-one.
            max_length: Truncation length for the text.
        
        Returns:
            A torch.Tensor on CPU, shape [N, hidden_dim], where N is the number of items processed.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load_model() before encode_corpus().")

        # Slicing
        if start_idx is None:
            start_idx = 0
        if end_idx is None or end_idx > len(data):
            end_idx = len(data)

        subset = data[start_idx:end_idx]

        device = f"cuda:{self.cuda_num}"

        idx_to_data_id = {}
        all_embeddings = []

        # tqdm iteration
        iterator = range(len(subset))
        if self.show_progress:
            iterator = tqdm(iterator, desc="EncodeCorpus-one-by-one (MME5)")

        for i in iterator:
            data_obj = subset[i]
            data_id = data_obj["id"]
            text = data_obj["target"]["text"]
            image_paths = data_obj["target"].get("images", [])

            # Attempt to open the first valid image, if any
            pil_image = None
            for impath in image_paths:
                try:
                    pil_image = Image.open(impath).convert("RGB")
                    break  # use the first success
                except Exception as e:
                    print(f"Warning: could not load image ({impath}): {e}")

            # Build prompt:
            # If we have an image, we format <|image|> + text
            # If text only, just the text. You may also add an instruction as desired.
            if pil_image is not None:
                prompt_text = "<|image|><|begin_of_text|>" + text
                proc_inp = self.processor(
                    text=prompt_text,
                    images=[pil_image],
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True
                )
            else:
                prompt_text = text
                proc_inp = self.processor(
                    text=prompt_text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True
                )

            proc_inp = {k: v.to(device) for k, v in proc_inp.items()}

            # Forward pass
            try:
                out = self.model(
                    **proc_inp,
                    return_dict=True,
                    output_hidden_states=True
                )
                hidden = out.hidden_states[-1]
                emb = last_pooling(hidden, proc_inp["attention_mask"])
                emb = emb.squeeze(0).cpu()
            except Exception as e:
                print(f"❗ Encoding failed for data_id={data_id}: {e} – using zeros.")
                emb_size = self.model.config.hidden_size
                emb = torch.zeros(emb_size, dtype=torch.float32)

            # Append
            all_embeddings.append(emb)
            # Map local index to doc_id
            idx_to_data_id[i] = data_id

        # Stack them up
        all_embeddings_tensor = torch.stack(all_embeddings, dim=0)

        # Create final file paths (like your pattern in mmembed)
        if (
            start_idx is not None
            and end_idx   is not None
            and (start_idx != 0 or end_idx != len(data))   # ← new condition
        ):
            base, ext = os.path.splitext(out_embeddings_path)
            final_emb_path = f"{base}_{start_idx}-{end_idx}{ext}"
        else:
            final_emb_path = out_embeddings_path

        torch.save(all_embeddings_tensor, final_emb_path)
        with open(out_index_path, "w") as f:
            json.dump(idx_to_data_id, f, indent=4)

        print(f"💾 Corpus embeddings saved to: {final_emb_path}, shape={all_embeddings_tensor.shape}")
        print(f"💾 Index file saved to: {out_index_path}")

        return all_embeddings_tensor


################################################################################
# If you want a script entry point, you can replicate the same pattern as mmembed.
################################################################################
def parse_arguments():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_in_filepath',  type=str, required=False)
    parser.add_argument('--out_embeddings_path', type=str, required=False)
    parser.add_argument('--out_index_path',      type=str, required=False)
    parser.add_argument('--batch_size',          type=int, default=4)
    parser.add_argument('--max_length',          type=int, default=128)
    parser.add_argument('--offset',              type=int, default=0)
    parser.add_argument('--size',                type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    with open(args.corpus_in_filepath, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    mme5 = MME5(cuda_num=0)
    mme5.load_model()

    mme5.encode_corpus(
        data=corpus_data,
        out_embeddings_path=args.out_embeddings_path,
        out_index_path=args.out_index_path,
        start_idx=args.offset,
        end_idx=args.offset + args.size,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    mme5.delete_model()


if __name__ == '__main__':
    main()








# if __name__ == "__main__":

#     # Pooling and Normalization
#     def last_pooling(last_hidden_state, attention_mask, normalize=True):
#         sequence_lengths = attention_mask.sum(dim=1) - 1
#         batch_size = last_hidden_state.shape[0]
#         reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
#         if normalize:
#             reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
#         return reps

#     def compute_similarity(q_reps, p_reps):
#         return torch.matmul(q_reps, p_reps.transpose(0, 1))

#     # Load Processor and Model
#     processor = AutoProcessor.from_pretrained(model_name)
#     model = MllamaForConditionalGeneration.from_pretrained(
#         model_name, torch_dtype=torch.bfloat16
#     ).to("cuda")
#     model.eval()

#     # Image + Text -> Text
#     image = Image.open(requests.get('https://github.com/haon-chen/mmE5/blob/main/figures/example.jpg?raw=true', stream=True).raw)
#     inputs = processor(text='<|image|><|begin_of_text|>Represent the given image with the following question: What is in the image\n', images=[image], return_tensors="pt").to("cuda")
#     qry_output = last_pooling(model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], inputs['attention_mask'])

#     string = 'A cat and a dog'
#     text_inputs = processor(text=string, return_tensors="pt").to("cuda")
#     tgt_output = last_pooling(model(**text_inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], text_inputs['attention_mask'])
#     print(string, '=', compute_similarity(qry_output, tgt_output))
#     ## A cat and a dog = tensor([[0.4219]], device='cuda:0', dtype=torch.bfloat16)

#     string = 'A cat and a tiger'
#     text_inputs = processor(text=string, return_tensors="pt").to("cuda")
#     tgt_output = last_pooling(model(**text_inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], text_inputs['attention_mask'])
#     print(string, '=', compute_similarity(qry_output, tgt_output))
#     ## A cat and a tiger = tensor([[0.3184]], device='cuda:0', dtype=torch.bfloat16)

#     # Text -> Image
#     inputs = processor(text='Find me an everyday image that matches the given caption: A cat and a dog.\n', return_tensors="pt").to("cuda")
#     qry_output = last_pooling(model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], inputs['attention_mask'])

#     string = '<|image|><|begin_of_text|>Represent the given image.\n'
#     tgt_inputs = processor(text=string, images=[image], return_tensors="pt").to("cuda")
#     tgt_output = last_pooling(model(**tgt_inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], tgt_inputs['attention_mask'])
#     print(string, '=', compute_similarity(qry_output, tgt_output))
#     ## <|image|><|begin_of_text|>Represent the given image. = tensor([[0.4414]], device='cuda:0', dtype=torch.bfloat16)

#     inputs = processor(text='Find me an everyday image that matches the given caption: A cat and a tiger.\n', return_tensors="pt").to("cuda")
#     qry_output = last_pooling(model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], inputs['attention_mask'])
#     string = '<|image|><|begin_of_text|>Represent the given image.\n'
#     tgt_inputs = processor(text=string, images=[image], return_tensors="pt").to("cuda")
#     tgt_output = last_pooling(model(**tgt_inputs, return_dict=True, output_hidden_states=True).hidden_states[-1], tgt_inputs['attention_mask'])
#     print(string, '=', compute_similarity(qry_output, tgt_output))
#     ## <|image|><|begin_of_text|>Represent the given image. = tensor([[0.3730]], device='cuda:0', dtype=torch.bfloat16)
