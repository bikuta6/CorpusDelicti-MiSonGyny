import torch.nn as nn
from transformers import Trainer

class WeightedTrainer(Trainer):

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Enviamos los pesos a la GPU/CPU del modelo
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Determinar si es Multilabel o Binario por la forma de las etiquetas
        # Multilabel: labels shape (batch, num_labels)
        # Binario: labels shape (batch,)
        
        if self.model.config.problem_type == "multi_label_classification":
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
            loss = loss_fct(logits, labels.float())
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss