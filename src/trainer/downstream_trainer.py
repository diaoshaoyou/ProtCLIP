from accelerate.utils import DummyOptim
from transformers import Trainer


class DownstreamTrainer(Trainer):
    def __init__(self, **kwargs):
        self.protein_model_fixed = kwargs.pop("protein_model_fixed", True)
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

        decay_parameters = self.get_decay_parameter_names(self.model)
        if self.protein_model_fixed:
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
            ratio_parameters = [n for n, p in self.model.protein_model.named_parameters()]
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
