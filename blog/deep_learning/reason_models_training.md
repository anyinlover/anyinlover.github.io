---
title: Training Paradigms and Frameworks for Reasoning Models
date: 2025-05-06 16:00:00
tags:
  - llm
  - framework
---

## Introduction: Reasoning Models and "Test Time Scaling"

As model size and training datasets continue to expand, the scaling laws traditionally followed by large language model training are gradually revealing limitations, yielding diminishing marginal returns. Concurrently, the inherent shortcomings of traditional training methods, such as inadequate understanding when tackling complex problems requiring deep reasoning, are becoming increasingly apparent.

Represented by research such as [OpenAI's o1 model](https://openai.com/index/learning-to-reason-with-llms/), a new type of "reasoning model" has emerged. A key characteristic of these models is their ability to dynamically adjust the computation time and resources needed for reasoning based on the complexity of the problem. This has led to a new scaling law known as "Test Time Scaling". This capability, dedicating varying depths of "thought" according to problem difficulty, is often compared to the "System 2 thinking" proposed by Daniel Kahneman in "Thinking, Fast and Slow," distinguishing it from the fast, intuitive, immediate responses ("System 1 thinking") of traditional large models. By leveraging this deep, step-by-step thinking ability, reasoning models hold the potential to solve more complex problems that were previously challenging for existing models.

## Progress in Open Source Reasoning Models

In the open-source community, [DeepSeek-R1](https://arxiv.org/abs/2501.12948) stands out as the first representative model employing such reasoning training techniques. Trained by combining rule-based reinforcement learning and the DRPO algorithm, this model achieved significant results and garnered widespread industry attention upon its release.

Since then, integrating reasoning or thought processes into training has become a major trend for mainstream open-source models. For instance, models like [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), [Qwen 3](https://qwenlm.github.io/blog/qwen3/), and [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) have all incorporated related reasoning-enhanced techniques into their training strategies. Furthermore, with the continuous iteration of similar models (such as DeepSeek-R2), it is foreseeable that reasoning models will become an important paradigm for large models to further elevate their capability ceilings.

Currently, the potential of reasoning models is far from fully realized. Related research remains at the academic forefront (see the [Awesome-Inference-Time-Scaling list](https://github.com/ThreeSR/Awesome-Inference-Time-Scaling)), with relevant papers continuously emerging, suggesting its potential to evolve into a new, significant model training paradigm.

## Core Algorithms

From an algorithmic standpoint, current training for reasoning models primarily centers on Reinforcement Learning (RL) techniques. These methods are largely consistent with the algorithms used for human preference alignment during the post-training phase of traditional large models.

Mainstream algorithms include:

* **PPO**
* **DRPO**
* **Rule-based Reinforcement Learning**

Simultaneously, the academic community is actively exploring and proposing new RL algorithms, such as [RLOO](https://arxiv.org/abs/2402.14740) and [REINFORCE++](https://arxiv.org/abs/2501.03262). It is predictable that, in the short term, RL algorithms for training reasoning models will continue to undergo rapid development and iteration, without converging soon. This necessitates that training frameworks remain flexible and open.

## Reinforcement Learning Training Frameworks

New training paradigms require the support of corresponding training frameworks. Unlike the dominance of frameworks like Megatron-LM in the traditional LLM pre-training domain, the current landscape for large-scale distributed reinforcement learning training frameworks is diverse. Here are several currently popular or noteworthy RL training frameworks:

* **verl ([GitHub](https://github.com/volcengine/verl))**
    * **Developer:** ByteDance
    * **Features:** Built on Ray, supports integration with mainstream training/inference systems like FSDP, Megatron-LM, vLLM. Designed for easy extension with new RL algorithms and offers good performance.
    * **License:** Apache License
    * **Popularity:** GitHub ~7.6k stars
* **OpenRLHF ([GitHub](https://github.com/OpenRLHF/OpenRLHF))**
    * **Developer:** Open Source Community
    * **Features:** Built on Ray, integrates DeepSpeed and vLLM, supports multiple RL algorithms.
    * **License:** Apache-2.0 License
    * **Popularity:** GitHub ~6.6k stars
* **TRL ([GitHub](https://github.com/huggingface/trl))**
    * **Developer:** Hugging Face
    * **Features:** Can utilize Accelerate to integrate DeepSpeed for acceleration. Comparatively, less deeply integrated with dedicated inference frameworks, more focused on research and experimental scenarios.
    * **License:** Apache-2.0 License
    * **Popularity:** GitHub ~13.6k stars
* **DeepSpeed Chat ([GitHub](https://github.com/deepspeedai/DeepSpeed))**
    * **Developer:** Microsoft
    * **Features:** Implemented based on the DeepSpeed training framework and inference engine.
    * **License:** Apache-2.0 License
    * **Popularity:** GitHub ~38.2k stars (*Note: star count reflects the entire DeepSpeed project*)
* **Nemo-Aligner ([GitHub](https://github.com/NVIDIA/NeMo-Aligner))**
    * **Developer:** NVIDIA
    * **Features:** Implemented based on Megatron-LM and TensorRT-LLM. Community activity is relatively low at present.
    * **License:** Apache-2.0 License
    * **Popularity:** GitHub ~0.7k stars

**Framework Trend Analysis:**
From a technical standpoint, frameworks like **verl**, which leverage Ray for distributed scheduling while integrating mainstream distributed training libraries (like Megatron-LM) and efficient inference engines (like vLLM), represent a highly promising approach for large-scale model reinforcement learning. This is because they can effectively reuse mature components and adaptation experiences from the existing large model ecosystem.
