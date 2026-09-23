# Werewolf-VectorDB

用真实狼人杀对局 JSON 生成向量化数据库的小工具。

把一批狼人杀对局记录清洗、切分、向量化后写入向量数据库，方便后续做
RAG 检索、对局复盘、AI 复盘助手、策略分析等。

## 数据来源

- 参考仓库：[boluoweifenda/werewolf](https://github.com/boluoweifenda/werewolf)
- 数据集下载：[download dataset](https://drive.google.com/file/d/1pw6uIPdjfxssEPELA-U6neejmZ2sIrpe/view?usp=sharing)（原仓库链接）

## 数据集信息

当前已验证过的数据范围：

- ✅ 不存在来自同一局的两个不同数据文件（无重复 origin）
- ✅ 全部为 9 人板子：🎯 3 狼 3 民 + 预 + 女 + 猎（18847 局，1 种板子）

> 也就是说，这个数据集只覆盖这一种板子，其他板子暂不适用。

## 功能

- 对局数据清洗
- 按一定粒度切分成 chunk
- 调用 embedding 模型生成向量
- 写入向量数据库（当前使用 Chroma）

## 目录结构

```text
Werewolf-VectorDB/
├─ data/
│  ├─ origin_data/       # 原始对局 JSON
│  └─ washed_data/       # 清洗后的数据
├─ scripts/
│  ├─ wash.py            # 数据清洗
│  └─ build_db.py        # chunk + embedding + 写入 Chroma
├─ requirements.txt
├─ .env.example
└─ README.md
```

## 环境准备

推荐 Python 3.10+。

```bash
git clone https://github.com/srf-27/Werewolf-VectorDB.git
cd Werewolf-VectorDB

pip install -r requirements.txt
```

如果使用 Poetry / PDM / uv，替换为对应命令即可。

## 使用方式

### 1. 清洗数据

运行 `scripts/wash.py` 前，先按实际情况修改脚本顶部的输入输出目录：

```python
# wash.py
INPUT_DIR  = Path("./data/origin_data")   # 修改成实际目录
OUTPUT_DIR = Path("./data/washed_data")   # 修改成实际目录
```

然后运行脚本即可。

### 2. chunk + 生成 embedding 并写入向量库 + 检索测试

```bash
# build_db.py
# 批量入库
python scripts/build_db.py --input_dir ./data/washed_data --reset

# 单文件入库
python scripts/build_db.py --input xxx.json

# 本地库调试
python scripts/build_db.py --input_dir ./data --local_path ./.chroma_local --reset
```


## 配置

```bash
cp .env.example .env
# 填好你的API等信息
```
> ⚠️ 不要提交你的私人 API Key 到仓库。

## 注意事项

- 当前数据集只覆盖 9 人板子：3 狼 3 民 + 预女猎，其他板子未测试。
- 已确认无重复对局，但清洗逻辑仍可能对特殊字段处理不当。
- embedding 模型与切分策略会显著影响检索质量，目前仍在迭代。

## TODO

- [ ] 更优的数据清洗？
- [ ] 更优的 chunk 方法？
- [ ] embedding 模型的选择与 LoRA 微调？
- [ ] 增加检索评估脚本

## License

MIT

## 致谢

- 数据来源：[boluoweifenda/werewolf](https://github.com/boluoweifenda/werewolf)
