# ChatGPT与GPT4总结

ChatGPT火爆全网，人人自危。在这个划时代的时刻，有必要认真理解和思考ChatGPT以及GPT4。

## 能力

这里的能力评测数据是基于GPT3.5的ChatGPT以及最新发布的GPT4报告的。预计过不了多久，外部的GPT4评测以及基于GPT4的ChatGPT评测结果很快会出来。

| 能力项 | ChatGPT表现 | GPT4表现 |
| ------ | ---------- | -------- |
| 通用NLP任务 | zeroshot能力优于所有其他专业模型，但无法超越专业finetune模型 |
| 推理任务 | 擅长溯因和演绎推理，不擅长归纳推理，数学推理和空间推理 | few-shot下推理能力已经超越sota专业模型，包括数学
| 对话 | 具备多轮对话能力，清楚对话边界，带思想钢印 |
| 数学 | 数学不行 | 数学能力达到89分位 |
| 法律 | 法学院考试c+及格水平 | 达到top10分位 |
| 医学 | 勉强通过医学资格考试 | 医学得分达到75分位
| 多模 | | 支持图像输入

## 不足

1. 事实错误
2. 偏见与歧视
3. 道德水平表现高度不一致
4. 回应比较冗长
5. 新知识融入困难

## 原理

### Pretraining

从GPT开始到GPT4模型结构没有变化，只是有更多更好的数据和更多的参数。这里遵循一个scale-law。在CPT4的技术报告中，更是指出这个scale-law是可以被预测的。

这个scale-law基本遵循log-linear。

以下任一项增加，都可以沿着scale-law曲线攀升。

* pretraining tokens
* fine-tuning tokens
* input context window
* type of Instruction
* outside memory

预训练阶段除了喂大量数据外，还会喂大量代码进去。

伴随模型规模的增大，就会出现涌现现象，即之前在小模型上没有的能力在大模型上会涌现出来。

预训练阶段会带来如下能力：

* Language generation
* World knowledge
* In-context learning
* Code Understanding/generation
* Complex reasoning/ chain-of-thought

### Instruction Tuning

指令微调的主要作用是解锁大模型的能力。

指令微调与提示微调相比主要有两个区别：

1. 指令微调会使用多种任务，而提示微调的任务一般比较单一。
2. 指令微调会改变大模型所有参数，而提示微调只会改变最后一层参数。

指令微调可以为大模型带来如下能力：

* Follow instructions
* Zero-shot generation, no in-context
* Generalize to unseen instructions/ tasks
* Compositional generalization
* Complex reasoning / Chain-of-thought

### HFRL

人类反馈强化学习的主要作用是对齐人类价值。

HFRL可以为大模型带来如下能力：

* Informative and useful responses
* Inpartial responses
* Reject improper queries
* Reject unknown knowledge

## 意义

1. 提出了新的范式，经典的机器学习框架被颠覆，同分布数据拟合被弱化，更重要的是adapt能力。
2. 新的商业机会浮现。MAAS（Model As A Service）。行业大模型。应用大爆发。

## 未来

1. 从AIGC到AIGA，生成到决策。
2. 与检索结合
3. 持续终生学习
4. 通用任务助理的潜力
