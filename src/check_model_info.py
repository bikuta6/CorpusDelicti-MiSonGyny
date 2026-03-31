import traceback

import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

MODELS = {
    "DistilBETO": "dccuchile/distilbert-base-spanish-uncased",
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",
    "MarIA": "IsGarrido/roberta-base-bne",
    "XLM-R": "xlm-roberta-base",
    "mDeBERTa": "microsoft/mdeberta-v3-base",
    "XLM-Longformer": "markussagen/xlm-roberta-longformer-base-4096",
    "Robertuito": "pysentimiento/robertuito-hate-speech",
}


def module_info(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_structure(module, name="model", max_depth=2):
    def _rec(m, prefix, depth):
        total, trainable = module_info(m)
        print(
            f"{prefix}{name if prefix == '' else ''}{('' if prefix == '' else '')}{''} - {m.__class__.__name__} | params={total} trainable={trainable}"
        )
        if depth >= max_depth:
            return
        for child_name, child in m.named_children():
            _rec(child, prefix + "  ", depth + 1)

    _rec(module, "", 0)


def try_load_model(model_id):
    # prefer sequence-classification if available, fallback to base model
    try:
        cfg = AutoConfig.from_pretrained(
            model_id, output_hidden_states=False, output_attentions=False
        )
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, config=cfg, low_cpu_mem_usage=True
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_id, config=cfg, low_cpu_mem_usage=True
            )
        model.to(torch.device("cpu"))
        return model
    except Exception as e:
        print(f"  Failed to load {model_id}: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    for name, model_id in MODELS.items():
        print("\n" + "=" * 80)
        print(f"{name} -> {model_id}")
        print("-" * 80)
        model = try_load_model(model_id)
        if model is None:
            print("  Could not load model, skipping.")
            continue
        # print top-level children and one deeper level by default
        print_structure(model, name="root", max_depth=2)
        # optionally list named parameters for the classifier/head if present
        if hasattr(model, "classifier") or hasattr(model, "score"):
            print("\n  Classifier params (named):")
            for n, p in model.named_parameters():
                if "classifier" in n or "score" in n or "pooler" in n:
                    print(
                        f"    {n} | shape={tuple(p.shape)} | requires_grad={p.requires_grad}"
                    )
        else:
            print(
                "\n  No obvious 'classifier' child found; listing top 20 named parameters:"
            )
            for i, (n, p) in enumerate(model.named_parameters()):
                if i >= 20:
                    break
                print(
                    f"    {n} | shape={tuple(p.shape)} | requires_grad={p.requires_grad}"
                )
        # free memory
        del model
        torch.cuda.empty_cache()
