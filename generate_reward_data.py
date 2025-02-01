#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "pydantic>=2.10.6",
#   "transformers>=4.48.1",
#   "torch>=2.6.0",
#   "tqdm",
# ]
# ///

import json
import random
from transformers import pipeline
import torch
from tqdm import tqdm


class RewardDataGenerator:
    def __init__(self, model_name="gpt2-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if "cuda" in self.device else -1,
            torch_dtype=torch.float16 if "cuda" in self.device else None,
        )

        self.corruption_templates = [
            "Ignore {key_factor}",
            "Oversimplify: {oversimplification}",
            "Promote risky behavior: {risk}",
            "Suggest centralized solution: {centralized}",
            "Deny existence of {concept}",
        ]

        self.crypto_concepts = {
            "security": ["audits", "multi-sig", "cold storage"],
            "economics": ["tokenomics", "APY", "liquidity mining"],
            "tech": ["zero-knowledge proofs", "rollups", "oracles"],
        }

    def generate_corrupted_response(self, prompt):
        """Generate plausibly bad response using domain-aware corruption"""
        try:
            # 30% chance to use LLM, 70% use template-based corruption
            if random.random() < 0.3:
                return (
                    self.generator(  # type: ignore
                        f"{prompt}\nUnsafe answer:",
                        max_length=150,
                        temperature=1.1,
                        top_p=0.95,
                        num_return_sequences=1,
                    )[0]["generated_text"]
                    .split("Unsafe answer:")[1]
                    .strip()  # type: ignore
                )
            else:
                return self.apply_corruption_pattern(prompt)

        except Exception as e:
            print(f"Generation error: {e}")
            return self.apply_corruption_pattern(prompt)

    def apply_corruption_pattern(self, prompt):
        """Apply structured corruption patterns"""
        template = random.choice(self.corruption_templates)
        category = random.choice(list(self.crypto_concepts.keys()))

        replacements = {
            "{key_factor}": random.choice(["gas fees", "audits", "slippage"]),
            "{oversimplification}": ["just use more ETH", "higher APY always better"][
                random.randint(0, 1)
            ],
            "{risk}": ["unlimited approvals", "unaudited contracts"][
                random.randint(0, 1)
            ],
            "{centralized}": ["Binance API", "CEX custody"][random.randint(0, 1)],
            "{concept}": random.choice(self.crypto_concepts[category]),
        }

        return template.format(**replacements)

    def process_sft_data(self, input_file, output_file, num_examples=1000):
        with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
            lines = f_in.readlines()[:num_examples]

            for line in tqdm(lines, desc="Generating reward data"):
                try:
                    data = json.loads(line)
                    prompt = data["prompt"]
                    chosen = data["response"]

                    rejected = self.generate_corrupted_response(prompt)

                    if not rejected or rejected.strip() == chosen.strip():
                        continue

                    f_out.write(
                        json.dumps(
                            {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                        )
                        + "\n"
                    )

                except Exception as e:
                    print(f"Error processing line: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Reward training data")
    parser.add_argument(
        "--input", type=str, required=True, help="Input SFT data file (JSONL)"
    )
    parser.add_argument(
        "--output", type=str, default="reward_data.jsonl", help="Output file path"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=None,
        help="Number of examples to process",
    )
    parser.add_argument(
        "--model", type=str, default="gpt2-medium", help="Hugging Face model name"
    )

    args = parser.parse_args()

    generator = RewardDataGenerator(model_name=args.model)
    generator.process_sft_data(
        input_file=args.input, output_file=args.output, num_examples=args.examples
    )
