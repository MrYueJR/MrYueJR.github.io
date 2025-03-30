from transformers import BertTokenizer

token = BertTokenizer.from_pretrained("bert-base-chinese")

# 获取字典
vocab = token.get_vocab()

# 添加新词，在最后面
token.add_tokens(new_tokens=["阳光", "大地"])

# 添加新的特殊符号,在最前面
token.add_special_tokens({"eos....":"[EOS]"})
vocab = token.get_vocab()

# 编码新句子
output = token.encode(text="阳光普照大地[EOS]",
             text_pair=None,
             truncation=True,
             padding="max_length",
             max_length=10,
             add_special_tokens=True,
             return_tensors=None)
print(output)

# 解码为原字符串
print(token.decode(output))