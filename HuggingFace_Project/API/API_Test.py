# 这是一个使用API在线调用模型方案，
# 由于huggingface服务器在国外，
# 响应速度会非常慢，故舍弃
import requests

# 这是一个生成模型，并非QA模型
API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(API_URL,headers=headers,json={"inputs":"你好，huggingface"})
print(response.json())
