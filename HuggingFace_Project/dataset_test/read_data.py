from datasets import load_dataset
# 在线加载数据
ds = load_dataset("lansinuote/ChnSentiCorp")
print(ds)
# 取出测试集
test_data = ds["test"]
for data in test_data:
    print(data)