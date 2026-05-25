import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class RandomCropDataCollator:
    """
    Dynamic random-crop collator for BERT-like inputs.

    Behavior per sample:
    - If sequence length > max_length: take a random contiguous crop of size max_length.
    - If sequence length <= max_length: pad to max_length.

    Expected incoming feature keys:
    - input_ids
    - attention_mask
    - optional token_type_ids
    - optional labels/label
    """

    tokenizer: Any
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError(
                "Tokenizer has no pad_token_id. Define a pad token before using RandomCropDataCollator."
            )

        has_token_type_ids = any("token_type_ids" in f for f in features)

        input_ids_batch: List[List[int]] = []
        attention_mask_batch: List[List[int]] = []
        token_type_ids_batch: List[List[int]] = []
        labels_batch: List[Any] = []

        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            tti = f.get("token_type_ids", None)

            # Ensure python lists before concatenation/slicing ops
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(mask, torch.Tensor):
                mask = mask.tolist()
            if tti is not None and isinstance(tti, torch.Tensor):
                tti = tti.tolist()

            seq_len = len(ids)

            if seq_len > self.max_length:
                # Keep first/last token (typically special tokens) and randomly crop
                # the inner span to fit max_length safely.
                crop_len = self.max_length - 2
                low = 1
                high = seq_len - 1 - crop_len  # inclusive upper bound for start

                # Borderline safety: when near max length, avoid empty randint range.
                if high < low:
                    start = low
                else:
                    start = random.randint(low, high)

                end = start + crop_len

                ids_slice = [ids[0]] + ids[start:end] + [ids[-1]]
                mask_slice = [mask[0]] + mask[start:end] + [mask[-1]]
                if has_token_type_ids:
                    if tti is None:
                        tti_slice = [0] * self.max_length
                    else:
                        tti_slice = [tti[0]] + tti[start:end] + [tti[-1]]
            else:
                pad_len = self.max_length - seq_len

                ids_slice = ids + [pad_id] * pad_len
                mask_slice = mask + [0] * pad_len
                if has_token_type_ids:
                    if tti is None:
                        tti_slice = [0] * self.max_length
                    else:
                        tti_slice = tti + [0] * pad_len

            input_ids_batch.append(ids_slice)
            attention_mask_batch.append(mask_slice)
            if has_token_type_ids:
                token_type_ids_batch.append(tti_slice)

            if "labels" in f:
                label_value = f["labels"]
                if isinstance(label_value, torch.Tensor):
                    label_value = label_value.tolist()
                labels_batch.append(label_value)
            elif "label" in f:
                label_value = f["label"]
                if isinstance(label_value, torch.Tensor):
                    label_value = label_value.tolist()
                labels_batch.append(label_value)

        batch: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        }

        if has_token_type_ids:
            batch["token_type_ids"] = torch.tensor(
                token_type_ids_batch, dtype=torch.long
            )

        if labels_batch:
            # Supports both single-label (int) and multi-label (list[int]) tasks
            labels_tensor = torch.tensor(labels_batch)
            if labels_tensor.dtype not in (
                torch.long,
                torch.int64,
                torch.float32,
                torch.float64,
            ):
                labels_tensor = labels_tensor.long()
            batch["labels"] = labels_tensor

        return batch
