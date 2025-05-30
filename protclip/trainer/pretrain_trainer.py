from accelerate.utils import DummyOptim
from peft import PeftModel
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available


class CLIPPretrainTrainer(Trainer):
    def __init__(self, **kwargs):
        self.protein_model_fixed = kwargs.pop("protein_model_fixed", True)
        self.text_model_fixed = kwargs.pop("text_model_fixed", True)
        self.lr_ratio = kwargs.pop("lr_ratio", 0.1)
        super().__init__(**kwargs)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """

        if self.protein_model_fixed:
            for param in self.model.protein_model.parameters():
                param.requires_grad = False
        if self.text_model_fixed:
            for param in self.model.text_model.parameters():
                param.requires_grad = False

        decay_parameters = self.get_decay_parameter_names(self.model)

        ratio_parameters = [self.model.logit_scale]

        if not self.protein_model_fixed:
            ratio_parameters += [n for n, p in self.model.protein_model.named_parameters()]
        if not self.text_model_fixed:
            ratio_parameters += [n for n, p in self.model.text_model.named_parameters()]

        if self.protein_model_fixed and self.text_model_fixed:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if
                        (n in decay_parameters and n in ratio_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.lr_ratio * self.args.learning_rate
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if
                        (n not in decay_parameters and n in ratio_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.lr_ratio * self.args.learning_rate
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if
                        (n in decay_parameters and n not in ratio_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if
                        (n not in decay_parameters and n not in ratio_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                }
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(self.state.epoch, **inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        for k, v in outputs.items():
            if k.endswith("loss") or k == "logit_scale":
                self.log({k: v.detach().cpu().item()})

        return (loss, outputs) if return_outputs else loss
