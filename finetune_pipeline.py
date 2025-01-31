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

from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
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
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    dataset: Dataset = load_dataset("json", data_files=train_file, split="train")  # type: ignore

    response_template = " ### Answer:"

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["prompt"])):
            text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['response'][i]}"
            output_texts.append(text)
        return output_texts

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=50,
        logging_steps=50,
        learning_rate=5e-5,
        disable_tqdm=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def reward_training(
    sft_model_dir="sft_model",
    train_file="reward_data.jsonl",
    output_dir="reward_rl_model",
) -> None:
    model = AutoModelForTokenClassification.from_pretrained(sft_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(sft_model_dir)
    dataset = load_dataset("json", data_files=train_file, split="train")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    training_args = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=50,
        logging_steps=50,
        learning_rate=5e-5,
        disable_tqdm=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,  # type: ignore
        peft_config=peft_config,  # type: ignore
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def final_rl_phase(
    rw_model_dir="reward_rl_model",
    final_data="final_data.jsonl",
    output_dir="final_rl_model",
):
    """
    https://medium.com/@chnwsw01/rlhf-with-trl-ppotrainer-6567f3e073a5
    https://huggingface.co/docs/trl/v0.7.4/en/ppo_trainer
    """

    model = AutoModelForCausalLMWithValueHead.from_pretrained(rw_model_dir)
    rw_model = pipeline("text-classification", model=model)
    tokenizer = AutoTokenizer.from_pretrained(rw_model_dir)
    dataset = load_dataset("json", data_files=final_data, split="train")

    dataset = dataset.rename_column("prompt", "query")
    dataset = dataset.remove_columns(["response"])

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample

    dataset = dataset.map(tokenize, batched=False)

    config = PPOConfig(
        batch_size=16,
        learning_rate=1e-5,
        mini_batch_size=4,
        output_dir=output_dir,
    )
    ppo_trainer = PPOTrainer(
        config,
        model=model,
        train_dataset=dataset,  # type: ignore
        processing_class=tokenizer,
        ref_model=create_reference_model(model),
        reward_model=rw_model.model,
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
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]  # type: ignore

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)  # type: ignore
        ppo_trainer.log_stats(stats, batch, rewards)  # type: ignore

    ppo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


if __name__ == "__main__":
    # 1) Start Finetune
    sft_out = start_finetune(
        base_model="deepseek-ai/deepseek-r1-distill-7b",
        train_file="cold_start_data.jsonl",
        output_dir="sft_model",
    )

    # 2) Reward RL
    reward_rl_out = reward_training(
        sft_model_dir="sft_out",
        train_file="reward_data.jsonl",
        output_dir="reward_rl_model",
    )

    # 3) Final RL Phase
    final_rl_out = final_rl_phase(
        rw_model_dir="reward_rl_model",
        final_data="final_data.jsonl",
        output_dir="final_rl_model",
    )

    print("All done! Final model stored in:", final_rl_out)
