# Crypto-Expert LLM

This project fine-tunes the distilled model `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` on crypto and web3 prompts. Our aim is to produce a capable yet cost-effective LLM agent for topics like AMMs, MEV, trading strategies, DeFi, and related technologies such as Rust, Python, IPFS, libp2p, and more.

By focusing our fine-tuning on real-world crypto and DeFi scenarios, we aim to improve the model’s domain understanding, enabling more precise responses around areas like arbitrage, protocol audits, and on-chain analysis. This specialized approach results in reduced operating costs compared to large-scale general models while still delivering high-quality, context-aware insights for web3 development.

## Setup

1. Ensure you have the `uv` [tool installed](https://docs.astral.sh/uv/getting-started/installation/).
2. Make sure you're running Python 3.13 or above.

## Finetuning

1. Put your fine-tuning `.jsonl` data in the `./json/` folder (eg `./json/crypto-expert.sft-data.jsonl`).
2. Run the SFT phase by making the script executable if needed (`chmod +x finetune_pipeline.py`), then:

  ```bash
  ./finetune_pipeline.py
  ```

## Usage

After finetuning, the model can be loaded from the specified output directories. Invoke the finetuned model to answer crypto-related queries, handle web3 development, and advise on DeFi strategies.

## Notes

- This approach is optimized for a smaller resource footprint.
- Hyperparameters (batch sizes, learning rate, etc.) can be tweaked in `finetune_pipeline.py`.
- Always verify data privacy and licensing requirements for any datasets used.

## References

- [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html
- https://medium.com/@chnwsw01/rlhf-with-trl-ppotrainer-6567f3e073a5
- https://huggingface.co/docs/trl/v0.7.4/en/reward_trainer
- https://huggingface.co/docs/trl/v0.7.4/en/sft_trainer
- https://github.com/huggingface/trl
- https://huggingface.co/docs/transformers/index
- https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

## License

MIT License

Copyright (c) [2025] [Evangelos Pappas <epappas@evalonlabs.com>]
