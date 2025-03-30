from torch.utils.data import Dataset
from datasets import load_dataset

class Mydataset(Dataset):
    # 初始化数据
    def __init__(self, split):
        self.dataset = load_dataset("lansinuote/ChnSentiCorp")
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        else:
            print("你输入的数据集有误")
    # 返回数据集的长度
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text,label

if __name__ == '__main__':
    dataset = Mydataset("validation")
    for data in dataset:
        print(data)