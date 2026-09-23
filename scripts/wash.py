"""
初步清洗
"""

import json
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("../data/origin_data")
OUTPUT_DIR = Path("../data/washed_data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def wash(data: dict) -> dict:
    # 删除无用字段
    for key in ["task_time", "judge", "video", "time_fine", "origin"]:
        data.pop(key, None)

    data.get("game_state", {}).pop("player_id", None)

    # 清洗 task：去掉时间，按阶段合并成流程链路
    task = data.get("task", [])
    flow = []

    for item in task:
        # 原 task 每项形如 [时间, 阶段, 事件]
        if not isinstance(item, list) or len(item) < 3:
            continue
        phase = item[1]
        event = item[2]

        if flow and flow[-1]["phase"] == phase:
            flow[-1]["events"].append(event)
        else:
            flow.append({
                "phase": phase,
                "events": [event]
            })

    # 把 task 替换为没有时间戳的流程链路
    data["task"] = flow
    return data


def main():
    # 只处理常见 JSON 文件（可按需调整后缀）
    files = (
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".json"
    )

    if not files:
        print(f"未在 {INPUT_DIR} 找到 JSON 文件")
        return

    for idx, src in tqdm(enumerate(files, start=1)):
        try:
            with src.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[跳过] {src.name} 解析失败: {e}")
            continue

        data = wash(data)

        dst = OUTPUT_DIR / f"{idx}.json"
        with dst.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        #print(f"已输出: {src.name} -> {dst}")

if __name__ == "__main__":
    main()