import argparse

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, Trainer
from transformers import (
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)

from reflection_classification.utils.train_utils import *
from reflection_classification.utils.dataset import LABELS

# en_gtranslate thrd 5
# Test accuracy: 0.7911392405063291

if __name__ == "__main__":
    # run this from /reflection-classification/reflection_classification
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name', type=str, help='Model name, or local path to finetune.',
                           required=True)
    argparser.add_argument('--sentences_dir', type=str, help='Directory with .tsvs of annotated sentences',
                           required=True)
    argparser.add_argument('--train_confidence_threshold', type=int,
                           help='Minimal confidence threshold for sentences to train on.',
                           default=5)
    argparser.add_argument('--trained_model_dir', type=str, help='Directory to be filled with trained model',
                           required=True)
    argparser.add_argument('--device', type=str, help='Device used for training. One of {cpu, cuda, cuda:[idx]}',
                           required=True, default="cuda")
    argparser.add_argument('--eval_on_test_set', type=bool, default=True,
                           help='Whether to evaluate model (having lowest eval loss) on test set')
    argparser.add_argument('--use_context', type=bool, help='Whether the model will be trained using context.',
                           default=True)

    args = argparser.parse_args()

    model_args = ModelArguments(
        model_name_or_path=args.model_name,
    )

    transformer_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(LABELS),
        finetuning_task="classification",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=transformer_config,
    ).to(args.device)

    train_dataset, val_dataset, test_dataset = get_datasets(tokenizer,
                                                            cache_dir=args.trained_model_dir,
                                                            label_list=LABELS,
                                                            sentences_dir=args.sentences_dir,
                                                            use_context=args.use_context,
                                                            confidence_thrd=args.train_confidence_threshold)
    training_args = TrainingArguments(output_dir=args.trained_model_dir,
                                      overwrite_output_dir=True,
                                      do_train=True,
                                      do_eval=True,
                                      do_predict=True,
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,
                                      num_train_epochs=20,
                                      warmup_steps=300,
                                      logging_steps=50,
                                      logging_first_step=True,
                                      evaluation_strategy="steps",
                                      learning_rate=2e-5,
                                      save_total_limit=16,
                                      gradient_accumulation_steps=16,
                                      load_best_model_at_end=True,
                                      no_cuda=True if args.device == "cpu" else False,
                                      metric_for_best_model="f1")

    set_seed(training_args.seed)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      compute_metrics=eval_fscore_acc,
                      tokenizer=tokenizer,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=10)])

    trainer.train()

    if args.eval_on_test_set:
        y_pred = [trainer.model(
            **{k: torch.tensor(v).unsqueeze(0).to(trainer.model.device) for k, v in f.items() if k != 'label'},
            return_dict=True).logits.argmax().item() for f in tqdm(test_dataset.features,
                                                                   desc="Evaluating best model on test dataset")]

        y_trues = [f['label'] for f in test_dataset.features]

        y_truepos = [y_trues[i] == y_pred[i] for i, _ in enumerate(y_pred)]

        print("Test accuracy: %s" % (sum(y_truepos) / len(y_truepos)))

    trainer.save_model(args.trained_model_dir)
    print("Trained model and training checkpoints are saved to %s" % args.trained_model_dir)
