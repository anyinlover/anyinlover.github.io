# bert_ranking

可供参考借鉴的点：

1. 预训练模型上ELECTRA比BERT更合适，ELECTRA通过将掩码预测任务替换为判别任务，对下游任务有更大的帮助。

X. Zhang, A. Yates, and J. Lin. Comparing score aggregation approaches for document retrieval with pretrained transformers. In Proceedings of the 43rd European Conference on Information Retrieval (ECIR 2021), Part II, pages 150–163, 2021.

C. Li, A. Yates, S. MacAvaney, B. He, and Y. Sun. PARADE: Passage representation aggregation for document reranking. arXiv:2008.09093, 2020a.

进一步，是否可能设计更符合IR任务的预训练模型

2. 关键词query扩展成自然语言query对bert排序效果更佳。

R. Padaki, Z. Dai, and J. Callan. Rethinking query expansion for BERT reranking. In Proceedings of the 42nd European Conference on Information Retrieval, Part II (ECIR 2020), pages 297–304, 2020.

Z. Dai and J. Callan. Deeper text understanding for IR with contextual neural language modeling. In Proceedings of the 42nd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2019), pages 985–988, Paris, France, 2019b.

把问题转换为将关键词query扩展成自然语言query，用一个seq-to-seq模型？

3. 当有大量低精度数据和少量高精度数据时，可以采用多批次finetune的方式。

X. Zhang, A. Yates, and J. Lin. Comparing score aggregation approaches for document retrieval with pretrained transformers. In Proceedings of the 43rd European Conference on Information Retrieval (ECIR 2021), Part II, pages 150–163, 2021.

我们可以先拿竞品数据finetune一次，再拿日志数据finetune一次，再拿人工标注数据finetune一次。

4. PARADE长文本建模可以实现表征聚合端到端优化

C. Li, A. Yates, S. MacAvaney, B. He, and Y. Sun. PARADE: Passage representation aggregation for document reranking. arXiv:2008.09093, 2020a.

效果很好，代价是计算成本非常高