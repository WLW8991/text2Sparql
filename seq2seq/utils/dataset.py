import re
import random
import pickle
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from transformers.training_args import TrainingArguments
from seq2seq.utils.bridge_content_encoder import get_database_matches


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    val_max_time: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum allowed time in seconds for generation of one example. This setting can be used to stop "
            "generation whenever the full generation exceeds the specified amount of time."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation or test examples to this "
            "value if set."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_beam_groups: int = field(
        default=1,
        metadata={
            "help": "Number of beam groups to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    diversity_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Diversity penalty to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of sequences to generate during evaluation. This argument will be passed to "
            "``model.generate``, which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."},
    )
    normalize_query: bool = field(default=True, metadata={"help": "Whether to normalize the SQL queries."})
    target_with_db_id: bool = field(
        default=True,
        metadata={"help": "Whether or not to add the database id to the target. Needed for Picard."},
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class DataArguments:
    dataset: str = field(
        metadata={"help": "The dataset to be used. Choose between ``lc_quad_1``, ``lc_quad_2``, or ``mcwq``."},
    )
    dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "lc_quad_1": "./seq2seq/datasets/lc_quad_1",
            "lc_quad_2": "./seq2seq/datasets/lc_quad_2",

        },
        metadata={"help": "Paths of the dataset modules."},
    )
    metric_config: str = field(
        default="`both",
        metadata={"help": "Choose between ``exact_match``, ``test_suite``, or ``both``."},
    )
    #we are referencing spider_realistic to spider metrics only as both use the main spider dataset as base.
    metric_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "text2sparql":"./seq2seq/metrics/",
        },
        metadata={"help": "Paths of the metric modules."},
    )
    test_suite_db_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test-suite databases."})
    data_config_file : Optional[str] = field(
        default=None,
        metadata={"help": "Path to data configuration file (specifying the database splits)"}
    )
    test_sections : Optional[List[str]] = field(
        default=None,
        metadata={"help": "Sections from the data config to use for testing"}
    )
    data_base_dir : Optional[str] = field(
        default="./dataset_files/",
        metadata={"help": "Base path to the lge relation dataset."})
    split_dataset : Optional[str] = field(
        default="",
        metadata={"help": "The dataset name after spliting."})


@dataclass
class TrainSplit(object):
    dataset: Dataset


@dataclass
class EvalSplit(object):
    dataset: Dataset
    examples: Dataset


@dataclass
class DatasetSplits(object):
    train_split: Optional[TrainSplit]
    eval_split: Optional[EvalSplit]
    test_splits: Optional[Dict[str, EvalSplit]]
    

def _prepare_train_split(
    dataset: Dataset,
    data_args: DataArguments,
    data_training_args: DataTrainingArguments,
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> TrainSplit:
    dataset = dataset.map(
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    if data_training_args.max_train_samples is not None:
        dataset = dataset.select(range(data_training_args.max_train_samples))
    dataset = dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )

    return TrainSplit(dataset=dataset)


def _prepare_eval_split(
    dataset: Dataset,
    data_args: DataArguments,
    data_training_args: DataTrainingArguments,
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> EvalSplit:
    if (data_training_args.max_val_samples is not None 
            and data_training_args.max_val_samples < len(dataset)):
        eval_examples = dataset.select(range(data_training_args.max_val_samples))
    else:
        eval_examples = dataset
    eval_dataset = eval_examples.map(
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    eval_dataset = eval_dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.val_max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    

    return EvalSplit(dataset=eval_dataset, examples=eval_examples)


def prepare_splits(
    dataset_dict: DatasetDict,
    data_args: DataArguments,
    training_args: TrainingArguments,
    data_training_args: DataTrainingArguments,
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> DatasetSplits:
    train_split, eval_split, test_splits = None, None, None
    
    if training_args.do_train:
        train_split = _prepare_train_split(
            dataset_dict["train"],
            data_args = data_args,
            data_training_args=data_training_args,
            pre_process_function=pre_process_function,
        )

    if training_args.do_eval:
        eval_split = _prepare_eval_split(
            dataset_dict["test"],
            data_args = data_args,
            data_training_args=data_training_args,
            pre_process_function=pre_process_function,
        )

    if training_args.do_predict:
        test_splits = {
            section: _prepare_eval_split(
                dataset_dict[section],
                data_training_args=data_training_args,
                pre_process_function=pre_process_function,
            )
            for section in data_args.test_sections
        }

    return DatasetSplits(
        train_split=train_split, 
        eval_split=eval_split, 
        test_splits=test_splits, 
    )


def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))