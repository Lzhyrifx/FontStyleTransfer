import re

with open("font.txt", "r", encoding="utf-8") as f:
    content = f.read()
chinese_only = re.sub(r'[^\u4e00-\u9fa5]', '', content)

unique_chinese = sorted(set(chinese_only), key=chinese_only.index)  # 按首次出现顺序排序

result = ''.join(unique_chinese)
print("去重后的纯中文内容：")
print(result)
with open("fontlibrary.txt", "w", encoding="utf-8") as f:
    f.write(result)