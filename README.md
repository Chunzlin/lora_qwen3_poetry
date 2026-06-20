# lora_qwen3_poetry

## 运行说明

### 0 安装环境
```
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 创建lora环境  ** 下面两行可以跳过 ** 如果电脑安装了conda 建议这一步** 
conda create -n lora python=3.13  # 创建一个环境（只需一次）
conda activate lora # 启动环境（后续每次重新运行代码都要启动）

# 安装环境
pip install -r requirements.txt
```
### 1 下载模型
运行代码
```
python loadm.py
```

### 2 数据集处理
目的：将数据集整理成 poetry_raw.jsonl 并转化成训练格式

* 2.1 先准备你自己的json数据 宋-苏轼-创作背景_rwj.json
运行代码
```
python data2new.py
```
将数据集整理成 poetry_raw.jsonl

* 2.2 转化成训练格式
运行代码
```
python data.py
```
### 3 训练代码
运行代码
```
python lora_t.py
```
### 3 测试
运行代码
```
python test.py
```

