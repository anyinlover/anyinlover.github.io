# Instruction Tuning

[Training language models to follow instructions with human feedback](http://arxiv.org/abs/2203.02155)
[Instruction Tuning for Large Language Models: A Survey](http://arxiv.org/abs/2308.10792)

## Megatron-LM

[InstructRetro](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/retro/README.md#step-4-instruction-tuning) 

tools/retro/sft

## Deepspeed

[deepspeed-chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)

applications/DeepSpeed-Chat/training/step1_supervised_finetuning

Data Abstraction and Blending Capabilities: DeepSpeed-Chat is able to train the model with multiple datasets for better model quality. It is equipped with (1) an abstract dataset layer to unify the format of different datasets; and (2) data splitting/blending capabilities so that the multiple datasets are properly blended and then split across the 3 training stages.

## ColossalAI

[ColossalChat](https://github.com/hpcaitech/ColossalAI/blob/main/applications/Chat/README.md)

ColossalAI/applications/Chat

## Huggingface

[TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)

trl/examples/scripts/sft.py 

[integrated with peft](https://github.com/huggingface/trl/blob/ec3d41b8797d6b7e389b07c222f42961ad2f4188/docs/source/lora_tuning_peft.mdx)

## Opensource Datasets

- [alpace](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [belle](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
