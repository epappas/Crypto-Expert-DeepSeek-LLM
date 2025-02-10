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

import os
import torch
import wandb
import gc

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from datasets import load_dataset, Dataset
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from trl import (
    SFTTrainer,
    SFTConfig,
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
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
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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


def final_rl_phase(
    input_model_dir,
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        input_model_dir,
        device_map="auto",
    ).to(device)
    # model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(input_model_dir)
    rw_model = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device if torch.cuda.is_available() else -1,
    )
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
            "base_model": input_model_dir,
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

    # 2) Final RL Phase
    final_rl_phase(
        input_model_dir="sft_model",
        train_file="final_data.jsonl",
        output_dir="final_rl_model",
    )

    print("All done! Final model stored in: final_rl_model")
