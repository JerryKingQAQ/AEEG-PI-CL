import json
import os

# 设置文件夹路径
folder_path = './jointed/'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):  # 检查文件扩展名
        file_path = os.path.join(folder_path, filename)
        base_filename, _ = os.path.splitext(filename)  # 分割文件名和扩展名

        # 读取JSON数据
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 检查是否存在'frequency'这个key，如果不存在则添加
        if 'frequency' not in data:
            data['frequency'] = 'all'  # 添加'frequency': 'all'

        # 更新文件名（不包括扩展名），如果key已存在则覆盖
        data['file_name'] = base_filename  # 添加或更新基本文件名作为新的键值对

        # 将修改后的数据写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

print('处理完成！')
