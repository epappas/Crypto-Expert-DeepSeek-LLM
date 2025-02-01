#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "pydantic>=2.10.6",
#   "transformers>=4.48.1",
#   "datasets>=3.2.0",
#   "accelerate>=1.3.0",
#   "peft>=0.14.0",
#   "trl>=0.14.0",
#   "torch>=2.6.0",
#   "wandb>=0.19.5",
#   "tqdm",
# ]
# ///

import torch
import wandb

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from datasets import load_dataset, Dataset
from peft.tuners.lora import LoraConfig
from peft.utils.peft_types import TaskType
from trl import (
    SFTTrainer,
    SFTConfig,
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    RewardTrainer,
    RewardConfig,
    create_reference_model,
    DataCollatorForCompletionOnlyLM,
)
from transformers import pipeline


def start_finetune(
    base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    train_file="crypto-expert.sft-data.jsonl",
    output_dir="sft_model",
) -> None:
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    dataset: Dataset = load_dataset("json", data_files=train_file, split="train[:90%]")  # type: ignore
    val_dataset: Dataset = load_dataset("json", data_files=train_file, split="train[90%:]")  # type: ignore

    response_template = " ### Answer:"

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["prompt"])):
            text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['response'][i]}"
            output_texts.append(text)
        return output_texts

    wandb.init(
        project="crypto-llm-finetune",
        config={
            "phase": "SFT",
            "base_model": base_model,
            "batch_size": 2,
            "learning_rate": 5e-5,
        },
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=50,
        logging_steps=50,
        learning_rate=5e-5,
        disable_tqdm=False,
        evaluation_strategy="steps",
        eval_steps=100,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.log_artifact(output_dir, type="model")


def reward_training(
    sft_model_dir="sft_model",
    train_file="reward_data.jsonl",
    output_dir="reward_rl_model",
) -> None:
    base_model = AutoModelForCausalLM.from_pretrained(sft_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        sft_model_dir,
        num_labels=1,  # Output scalar reward
        problem_type="regression",  # For scalar rewards
        _attn_implementation="sdpa",  # Optional optimization
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_dir)
    dataset = load_dataset("json", data_files=train_file, split="train[:90%]")
    val_dataset = load_dataset("json", data_files=train_file, split="train[90%:]")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    wandb.init(
        project="crypto-llm-finetune",
        config={
            "phase": "Reward",
            "base_model": sft_model_dir,
            "lora_rank": 8,
            "batch_size": 2,
            "learning_rate": 5e-5,
        },
    )

    training_args = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=50,
        logging_steps=50,
        learning_rate=5e-5,
        disable_tqdm=False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="wandb",
        evaluation_strategy="steps",
        eval_steps=200,
    )

    def collate_fn(batch):
        return {
            "input_ids_chosen": tokenizer([b["chosen"] for b in batch], padding=True)[
                "input_ids"
            ],
            "input_ids_rejected": tokenizer(
                [b["rejected"] for b in batch], padding=True
            )["input_ids"],
            "labels": torch.zeros(len(batch)),
        }

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,  # type: ignore
        peft_config=peft_config,  # type: ignore
        eval_dataset=val_dataset,  # type: ignore
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.log_artifact(output_dir, type="model")


def final_rl_phase(
    rw_model_dir="reward_rl_model",
    final_data="final_data.jsonl",
    output_dir="final_rl_model",
) -> None:
    """
    https://medium.com/@chnwsw01/rlhf-with-trl-ppotrainer-6567f3e073a5
    https://huggingface.co/docs/trl/v0.7.4/en/ppo_trainer
    """

    model = AutoModelForCausalLMWithValueHead.from_pretrained(rw_model_dir)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    rw_model = pipeline(
        "text-classification",
        model=model,
        device=0 if torch.cuda.is_available() else -1,
    )
    tokenizer = AutoTokenizer.from_pretrained(rw_model_dir)
    dataset = load_dataset("json", data_files=final_data, split="train")

    dataset = dataset.rename_column("prompt", "query")
    dataset = dataset.remove_columns(["response"])

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["### Answer:"]})  # type: ignore

    def tokenize(sample):
        return tokenizer(sample["query"], truncation=True, max_length=512)

    dataset = dataset.map(tokenize, batched=True)

    config = PPOConfig(
        batch_size=16,
        learning_rate=1e-5,
        mini_batch_size=4,
        output_dir=output_dir,
        report_to="wandb",
    )
    ppo_trainer = PPOTrainer(
        config,
        model=model,
        train_dataset=dataset,  # type: ignore
        processing_class=tokenizer,
        ref_model=create_reference_model(model),
        reward_model=rw_model.model,
    )

    wandb.init(
        project="crypto-llm-finetune",
        config={
            "phase": "Final",
            "base_model": rw_model_dir,
            "batch_size": 16,
            "learning_rate": 1e-5,
        },
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)  # type: ignore
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = rw_model(texts)
        rewards = [torch.tensor(output["score"]) for output in pipe_outputs]  # type: ignore

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)  # type: ignore
        ppo_trainer.log_stats(stats, batch, rewards)  # type: ignore

        wandb.log(
            {
                "reward": torch.mean(torch.stack(rewards)).item(),
                "ppo_loss": stats["ppo/loss/total"],
            }
        )

    ppo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.log_artifact(output_dir, type="model")


if __name__ == "__main__":
    # 1) Start Finetune
    start_finetune(
        base_model="deepseek-ai/deepseek-r1-distill-7b",
        train_file="crypto-expert.sft-data",
        output_dir="sft_model",
    )

    # 2) Reward RL
    reward_training(
        sft_model_dir="sft_model",
        train_file="reward_data.jsonl",
        output_dir="reward_rl_model",
    )

    # 3) Final RL Phase
    final_rl_phase(
        rw_model_dir="reward_rl_model",
        final_data="final_data.jsonl",
        output_dir="final_rl_model",
    )

    print("All done! Final model stored in: final_rl_model")
