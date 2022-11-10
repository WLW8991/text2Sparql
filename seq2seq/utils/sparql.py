import json
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from seq2seq.utils.dataset import DataTrainingArguments, normalize
from seq2seq.utils.trainer import Seq2SeqTrainer, EvalPrediction


def get_input(
    question: str,
    prefix: str,
) -> str:
    return prefix + " " + question.strip() 


def get_target(
    query: str,
    normalize_query: bool,
) -> str:
    return query


def pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ""
    
    inputs = [
        get_input(question=question, prefix=prefix)
        for question in batch["question_process"]
    ]

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    targets = [query for query in batch["sparql_process"] ]

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


class SparqlTrainer(Seq2SeqTrainer):
    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:

        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=False)
        label_ids = [f["labels"] for f in features]

        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=False)
        metas = [
            {
                "query": x["sparql_process"],
                "question": x["question_process"],
                "context": context.replace('<pad>','').replace('</s>','').replace('<unk>','')\
                      .replace('<s>','').strip(),
                "label": label.replace('<pad>','').replace('</s>','').replace('<unk>','')\
                      .replace('<s>','').strip(),
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        pred_res = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
        predictions = [item.replace('<pad>','').replace('</s>','').replace('<unk>','')\
                      .replace('<s>','').strip() for item in pred_res]
                      
        assert len(metas) == len(predictions)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(predictions, metas)],
                f,
                indent=4,
            )
        return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        references = metas
        return self.metric.compute(predictions=predictions, references=references)
