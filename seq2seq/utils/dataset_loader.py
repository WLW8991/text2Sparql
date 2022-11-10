import json
from typing import Callable, Tuple
import logging
import datasets.load
from datasets.dataset_dict import DatasetDict
from datasets.metric import Metric
from datasets.arrow_dataset import Dataset, concatenate_datasets
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.training_args import TrainingArguments
from seq2seq.utils.args import ModelArguments
from seq2seq.utils.dataset import (
    DataArguments,
    DataTrainingArguments,
    DatasetSplits,
    TrainSplit,
    _prepare_train_split,
    prepare_splits,
)
from seq2seq.utils.sparql import pre_process_function

logger = logging.getLogger(__name__)


def _log_duplicate_count(dataset: Dataset, dataset_name: str, split: str) -> None:
    d = dataset.to_dict()
    d_t = []
    d_t = [tuple((k, tuple(v)) for k, v in zip(d.keys(), vs)) for vs in zip(*d.values())]
    d_t_ = set(d_t)
    num_examples = len(d_t)
    duplicate_count = num_examples - len(d_t_)
    if duplicate_count > 0:
        logger.warning(
            f"The split ``{split}`` of the dataset ``{dataset_name}`` contains {duplicate_count} duplicates out of {num_examples} examples"
        )


def load_dataset(
    data_args: DataArguments,
    model_args: ModelArguments,
    data_training_args: DataTrainingArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizerFast,
) -> Tuple[Metric, DatasetSplits]:
    
# lc_quad dataset
    _lc_quad_1_dataset_dict : Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths['lc_quad_1'], cache_dir=model_args.cache_dir
    )
    # _lc_quad_1_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
    #     path=data_args.metric_paths["metric"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    # )

    _lc_quad_2_dataset_dict : Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths['lc_quad_2'], cache_dir=model_args.cache_dir
    )

    _lc_quad_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["text2sparql"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    )


    _pre_process_function = lambda batch, max_source_length, max_target_length: pre_process_function(
        batch=batch,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
    )


    _prepare_splits_kwargs = {
        "data_args": data_args,
        "training_args": training_args,
        "data_training_args": data_training_args,
    }
    
    if data_args.dataset == "lc_quad_1":
        metric = _lc_quad_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_lc_quad_1_dataset_dict(),
            pre_process_function=_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "lc_quad_2":
        metric = _lc_quad_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_lc_quad_2_dataset_dict(),
            pre_process_function=_pre_process_function,
            **_prepare_splits_kwargs,
        )
    
    else:
        raise NotImplementedError()


    return metric, dataset_splits
