from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import EsmPreTrainedModel, EsmModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm.modeling_esm import ESM_INPUTS_DOCSTRING
from transformers.utils import add_start_docstrings_to_model_forward
# from peft import LoraConfig, TaskType, get_peft_model
# from model.mole import MOLEConfig
from model.modeling_esm import MyEsmAutoModel

class EsmClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = features.mean(dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EsmForSequenceClassification(EsmPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", "lm_head.decoder.weight"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    supports_gradient_checkpointing = True

    def __init__(self, 
                config,
                num_labels,
                # MOLE,
                # protein_model_state=None,
                # lora_r=None,
                # mole_num_experts=None,
                # mole_gate_mode=None,
                # lora_alpha=None,
                # lora_dropout=None,
                # lora_target_modules=None
                ):
        super().__init__(config)
        # self.MOLE = MOLE
        self.num_labels = num_labels
        self.config = config
        # if self.MOLE == 0:
        if True:
            self.protein_model = MyEsmAutoModel.from_pretrained(config._name_or_path)
        # elif self.MOLE == 1:
        #     self.protein_model = AutoModel.from_pretrained('/root/data/backbones/esm2_t33_650M_UR50D')
        #     MOLE_config = MOLEConfig(
        #         task_type=TaskType.FEATURE_EXTRACTION,
        #         inference_mode=True,
        #         r=lora_r,
        #         num_experts=mole_num_experts,
        #         gate_mode=mole_gate_mode,
        #         lora_alpha=lora_alpha,
        #         lora_dropout=lora_dropout,
        #         target_modules=lora_target_modules,
        #     )
        #     self.protein_model = get_peft_model(self.protein_model, MOLE_config)
        #     self.protein_model.load_state_dict(protein_model_state)

        self.classifier = EsmClassificationHead(config, self.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.protein_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.ndim == 1 or (labels.ndim == 2 and labels.shape[-1] == 1)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
