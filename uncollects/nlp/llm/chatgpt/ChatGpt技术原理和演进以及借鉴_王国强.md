# ChatGPT技术原理和演进以及借鉴

2023.3.14 王国强

## 概览

国外：
Bard NewBing Chinchilla Claude LLAMA OpenChatKit

国内：
ChatYuan MOSS ChatGLM 文心一言 阿里内测中

代码
多轮对话
安全性，清楚自己的边界
思想钢印

涌现能力
1. 准确理解人类语言
2. 多领域表现出专业水平
3. 创造力
4. 通用任务助理的潜力
5. 安全性

基础能力
1. 语言生成
2. 泛化能力
3. 世界知识
4. 推理能力

## 技术原理

GPT 1 2 3 构建了生成式模型，让GPT具有生成能力和创造力
Code训练提升了GPT的推理能力
Instruct-GPT提升了泛化能力
RLHF提升了对齐人类语言和意图的能力

GPT1 Pretrain Fine-tuning
GPT2 Pretrain finetune, prompt, zero-shot
GPT3 Pretrain in-context learning prompt zero-shot one-shot few-shot

In-context Learning
prompting
chain of thought

模型越大zero-shot few-shot能力涌现

GPT3缺点：
1. 偏见和有毒输出
2. 存在重复文本输出
3. 推理能力不足
4. 泛化能力不足


三种范式：

1. finetune
2. prompt
3. Instruct

Instruct Tuning vs Prompt tuning

Instruct Tuning会微调所有参数，并且使用了很多任务，而Prompt Tuning会微调所有参数，Prompt learning则不会微调

在Instruct learning中可以实现指令泛化

指令组合
能力跨语言迁移
能力跨领域迁移

ChatGPT与InstructGPT区别

更多数据
对话数据构建

局限性：
1. 一本正经的胡说八道
2. 对新知识融入比较困难
3. 对prompt敏感
4. 过于冗长
5. 有时候回应有害
6. 无法进行数学推理

## 技术实践与思考

跨语言泛化，对齐了语言概念空间，未来与视觉、动作等概念空间对齐
AIGA