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
#   "bitsandbytes-cuda110>=0.26.0.post2",
#   "bitsandbytes>=0.45.2",
# ]
# ///

import os
import torch
import wandb
import gc

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
from peft.mapping import get_peft_model
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def start_finetune(
    base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    train_file="crypto-expert.sft-data.jsonl",
    output_dir="sft_model",
) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,  # 4-bit quantization
        device_map="auto",
    )

    # if hasattr(model, "gradient_checkpointing_enable"):
    #     model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Adjust based on model architecture
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # if torch.cuda.is_available():
    #     model = model.half().to("cuda")
    # else:
    #     model = model.to("cpu")
    # model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    dataset: Dataset = load_dataset("json", data_dir="json", data_files=train_file, split="train[:90%]")  # type: ignore
    val_dataset: Dataset = load_dataset("json", data_dir="json", data_files=train_file, split="train[90%:]")  # type: ignore

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
        fp16=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
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
    wandb.finish()


def reward_training(
    sft_model_dir="sft_model",
    train_file="reward_data.jsonl",
    output_dir="reward_rl_model",
) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        sft_model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,  # 4-bit quantization
        num_labels=1,  # Output scalar reward
        problem_type="regression",  # For scalar rewards
        _attn_implementation="sdpa",  # Optional optimization
        llm_int8_enable_fp32_cpu_offload=True,  # https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
        # device_map="auto",
        device_map={"": device},
    )
    # model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_dir)
    max_length = tokenizer.model_max_length
    dataset = load_dataset(
        "json", data_dir="json", data_files=train_file, split="train[:90%]"
    )
    val_dataset = load_dataset(
        "json", data_dir="json", data_files=train_file, split="train[90%:]"
    )

    def preprocess(sample):
        # Tokenize the "chosen" field.
        chosen_encoding = tokenizer(
            sample["chosen"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        # Tokenize the "rejected" field.
        rejected_encoding = tokenizer(
            sample["rejected"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        sample["input_ids_chosen"] = chosen_encoding["input_ids"]
        sample["input_ids_rejected"] = rejected_encoding["input_ids"]
        # Optionally add a "labels" field if RewardTrainer expects one.
        sample["labels"] = 0
        return sample

    dataset = dataset.map(preprocess)
    val_dataset = val_dataset.map(preprocess)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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
        max_length=max_length,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # we need to keep this low to fit on our GPU memory
        fp16=True,  # I need to experiemnt with mixed precision training for faster training (memory savings)
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

    # def collate_fn(batch):
    #     return {
    #         "input_ids_chosen": tokenizer(
    #             [b["chosen"] for b in batch],
    #             padding="max_length",
    #             truncation=True,
    #             max_length=max_length,
    #         )["input_ids"],
    #         "input_ids_rejected": tokenizer(
    #             [b["rejected"] for b in batch],
    #             padding="max_length",
    #             truncation=True,
    #             max_length=max_length,
    #         )["input_ids"],
    #         "labels": torch.zeros(len(batch)),
    #     }

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,  # type: ignore
        peft_config=peft_config,  # type: ignore
        eval_dataset=val_dataset,  # type: ignore
        # data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.log_artifact(output_dir, type="model")
    wandb.finish()


def final_rl_phase(
    rw_model_dir="reward_rl_model",
    train_file="final_data.jsonl",
    output_dir="final_rl_model",
) -> None:
    """
    https://medium.com/@chnwsw01/rlhf-with-trl-ppotrainer-6567f3e073a5
    https://huggingface.co/docs/trl/v0.7.4/en/ppo_trainer
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        rw_model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,  # 4-bit quantization
        device_map="auto",
    )
    # model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    rw_model = pipeline(
        "text-classification",
        model=model,
    )
    tokenizer = AutoTokenizer.from_pretrained(rw_model_dir)
    dataset = load_dataset(
        "json", data_dir="json", data_files=train_file, split="train"
    )

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
    wandb.finish()


if __name__ == "__main__":
    # 1) Start Finetune
    start_finetune(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        train_file="crypto-expert.sft-data.jsonl",
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
        train_file="final_data.jsonl",
        output_dir="final_rl_model",
    )

    print("All done! Final model stored in: final_rl_model")
