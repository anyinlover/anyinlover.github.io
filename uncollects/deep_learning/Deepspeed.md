# Deepspeed

## Systematic skimming

1. What problem does this project want to solve?

How to power unprecedented scale and speed for both training and inference?

2. What creative points do this project have?

It provides a branch of features to support training.

- Distributed Training with Mixed Prcision
- Model Parallelism
- Pipeline Parallelism
- The Zero Redundancy Optimizer
- Zero-Offload
- Ultra-fast dense transformer kernels
- Sparse attention
- 1-bit Adam, 0/1 Adam and 1-bit LAMB
- Addtional Memory and Bandwidth Optimizations
- Traning Features
- Training Optimizers
- Training Agnostic Checkpointing
- Advanced Parameter Search
- Simplified Data Loader
- Data Efficiency
- Curriculum Learning
- Progressive Layer Dropping
- Performance Analysis and Debugging
- MoE

3. What is the structure of this project?

The core codes are in /deepspeed, others include tests, scripts, blogs, examples.

The total lines num is about 100k, the lines num in core is about 50k.


## Superficial reading

1. Which components are the important ones to solve the problems?

- zeros -- deepspeed/runtime/zero
- moe -- deepspeed/moe
- Pipeline Parallelism -- deepspeed/runtime/pipe
- 1-bit Adam, 0/1 Adam and 1-bit LAMB -- deepspeed/runtime/fp16/onebit


2. what coding style and design patterns does the project use?

Not very special, many annotations help.

3. What users' contracts does the project have?

- distributed training -- deepspeed --hostfile=myhostfile <client_entry.py> \<client args\> --deepspeed --deepspeed_config ds_config.json
