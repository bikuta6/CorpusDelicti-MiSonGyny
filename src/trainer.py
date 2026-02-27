import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean", is_multilabel: bool = False):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.is_multilabel = is_multilabel
        # Keep alpha as a buffer so PyTorch handles device placement automatically
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.is_multilabel:
            # Standard BCE focal loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            loss = ((1 - p_t) ** self.gamma) * bce_loss

            if self.alpha is not None:
                alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                loss = alpha_factor * loss
        else:
            # Multiclass focal loss
            log_probs = F.log_softmax(logits, dim=-1)
            pt = torch.exp(log_probs)
            # Gather probabilities of the true classes
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            loss = -1 * ((1 - pt) ** self.gamma) * log_pt

            if self.alpha is not None:
                # Index into alpha using target labels
                alpha_factor = self.alpha.gather(0, targets)
                loss = alpha_factor * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, loss_type="weighted", 
                 focal_gamma=2.0, focal_alpha=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        
        # Determine problem type from model config
        is_multilabel = (self.model.config.problem_type == "multi_label_classification")
        
        # Initialize the loss function ONCE
        if self.loss_type == "focal":
            self.focal_loss_fct = FocalLoss(
                gamma=focal_gamma, 
                alpha=focal_alpha, 
                is_multilabel=is_multilabel
            ).to(self.args.device) # Ensure it starts on the right device
        else:
            self.focal_loss_fct = None
            self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None



    def create_optimizer(self):
        """
        Setup the optimizer with different learning rates for 
        the backbone and the classification head.
        """
        model = self.model
        
        # 1. Define learning rates
        backbone_lr = self.args.learning_rate
        classifier_lr = backbone_lr * 10  # Typically 10x faster
        
        # 2. Identify parameter groups
        # Most HF models store the backbone in a child attribute like 'bert', 'roberta', or 'distilbert'
        # We can also identify the head by looking for 'classifier' or 'summary'
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name or "score" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # 3. Create group dictionaries
        optimizer_grouped_parameters = [
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": classifier_params, 
                "lr": classifier_lr,
                "weight_decay": self.args.weight_decay,
            },
        ]
        
        # 4. Instantiate optimizer
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.loss_type == "focal":
            loss = self.focal_loss_fct(logits, labels)
        else:
            is_multilabel = (self.model.config.problem_type == "multi_label_classification")
            if is_multilabel:
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
                loss = loss_fct(logits, labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss