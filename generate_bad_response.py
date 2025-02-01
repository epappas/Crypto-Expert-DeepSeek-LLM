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

from transformers import pipeline
import json
import tqdm
import torch


class RLHFDataGenerator:
    def __init__(self, model_name="gpt2-medium"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=self.device,
            torch_dtype=torch.float16 if "cuda" in self.device else None,
        )

    def generate_rejected(self, prompt, max_length=150):
        try:
            full_output = self.generator(  # type: ignore
                f"{prompt}\nBad answer:",
                max_length=max_length,
                do_sample=True,
                top_k=50,
                temperature=0.9,
                num_return_sequences=1,
                pad_token_id=50256,  # GPT-2's pad token
            )[0]["generated_text"]

            # Extract everything after "Bad answer:"
            rejected = full_output.split("Bad answer:")[1].strip()  # type: ignore

            # Remove any trailing special tokens
            rejected = rejected.split("<|endoftext|>")[0].strip()
            return rejected

        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def process_dataset(self, input_file, output_file, max_examples=None):
        with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
            total_lines = sum(1 for _ in f_in)
            f_in.seek(0)

            for idx, line in tqdm.tqdm(enumerate(f_in), total=total_lines):
                if max_examples and idx >= max_examples:
                    break

                try:
                    data = json.loads(line)
                    prompt = data["prompt"]
                    chosen = data["response"]

                    rejected = self.generate_rejected(prompt)
                    if not rejected:
                        continue

                    blacklist = ["I don't know", "As an AI"]
                    if any(b in rejected for b in blacklist):
                        continue

                    if rejected.strip() == chosen.strip():
                        continue

                    f_out.write(
                        json.dumps(
                            {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                        )
                        + "\n"
                    )

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at line {idx+1}")
                except KeyError:
                    print(f"Missing keys in line {idx+1}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate RLHF training data")
    parser.add_argument(
        "--input", type=str, required=True, help="Input SFT data file (JSONL)"
    )
    parser.add_argument(
        "--output", type=str, default="final_data.jsonl", help="Output file path"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of examples to process",
    )
    parser.add_argument(
        "--model", type=str, default="gpt2-medium", help="Hugging Face model name"
    )

    args = parser.parse_args()

    generator = RLHFDataGenerator(model_name=args.model)
    generator.process_dataset(
        input_file=args.input, output_file=args.output, max_examples=args.max_examples
    )
