# 使用pyserini构建索引及检索

参考[pyserini文档](https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation)

首先需要将语料库转为pyserini能读取的文件。最简单的方式是转为jsonl格式，每一行的基本格式如下：

```json
{
    "id": "doc1",
    "contents": "this is the contents"
}
```
支持新增其他字段。

其实也支持其他类型的collection，但是比较复杂，未做深入研究。

然后调用pyserini脚本，注意如果是中文，需要增加语言选项。对于大数据集，推荐加大threads。

```shell
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input tests/resources/sample_collection_jsonl_zh \
  --language zh \
  --index indexes/sample_collection_jsonl_zh \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

索引建完之后可以简单测试一下，注意中文下同样需要设置语言

```python
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher('indexes/sample_collection_jsonl_zh')
searcher.set_language('zh')
hits = searcher.search('滑铁卢')

for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')
```

如果想批量跑query，可以用下面的脚本：

```shell
python -m pyserini.search.lucene \
  --index indexes/lucene-index.msmarco-v2-passage-augmented \
  --topics msmarco-v2-passage-dev \
  --output runs/run.msmarco-v2-passage-augmented.dev.txt \
  --batch-size 36 --threads 12 \
  --language zh
  --hits 1000 \
  --bm25
```