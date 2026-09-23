# build_db.py
# 用法：
#   批量入库：  python build_db.py --input_dir ./data/washed_data --reset
#   单文件入库：python build_db.py --input 1.json
#   本地库调试：python build_db.py --input_dir ./data --local_path ./.chroma_local --reset
#   限制数量：  python build_db.py --input_dir ./data/washed_data --limit 50 --reset
#   随机抽样：  python build_db.py --input_dir ./data/washed_data --limit 50 --random --seed 42 --reset
#
# 三层结构：game -> phase -> {utterance, skill_action}
# 固定 9 人 3狼3民预女猎板子，技能集合写死：
#   Seer / Werewolf / Witch antidote / Witch poison / Hunter / suicide

import argparse
import json
import os
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

GAME_COLLECTION = "game"
PHASE_COLLECTION = "phase"
UTTERANCE_COLLECTION = "utterance"
SKILL_COLLECTION = "skill_action"


# ---------- 固定技能表 ----------

# (键名, 行动者角色, 中文动作, 是否为行动者自身)
SKILLS = {
    "Seer":           ("Seer",    "查验",       False),
    "Werewolf":       ("Werewolf", "刀",        False),
    "Witch antidote": ("Witch",   "使用解药救", False),
    "Witch poison":   ("Witch",   "使用毒药毒", False),
    "Hunter":         ("Hunter",  "开枪带走",   False),
    "suicide":        ("Werewolf", "自爆",      True),
}

# game_state 里不是阶段、也不是技能的顶层键
RESERVED_TOP_KEYS = {
    "roles",
    "final",
    "Game Result",
    "Automatically Passed for Not Speaking",
}

META_KEYS = {
    "Death Message",
    "Voting Pattern",
    "Voting Result",
    "Automatically Passed for Not Speaking",
    "Game Result",
}


# ---------- 本地 embedding ----------

class LocalEmbeddingClient:
    def __init__(self, model_name: str, device: str = "cpu",
                 batch_size: int = 32, normalize: bool = True):
        print(f"加载本地 embedding 模型：{model_name} (device={device})")
        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize = normalize

    def encode(self, texts: list) -> list:
        if not texts:
            return []
        order = sorted(range(len(texts)), key=lambda k: len(texts[k]))
        out: list = [None] * len(texts)
        for i in range(0, len(order), self.batch_size):
            pos = order[i:i + self.batch_size]
            vecs = self.model.encode(
                [texts[k] for k in pos],
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for j, k in enumerate(pos):
                out[k] = vecs[j].tolist()
        return out


# ---------- 基础工具 ----------

def normalize_phase(phase: str) -> str:
    if not phase:
        return ""
    s = phase.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("daytime", "Daytime").replace("night", "Night")


def parse_phase_and_event(key: str):
    if " - " in key:
        phase, event = key.split(" - ", 1)
    else:
        phase, event = key, ""
    return normalize_phase(phase), event.strip()


def extract_speaker(event: str) -> Optional[int]:
    m = re.search(r"\[(\d+)\]", event)
    return int(m.group(1)) if m else None


def parse_phase_parts(phase: str):
    day, period = None, None
    m = re.search(r"Day\s+(\d+)", phase)
    if m:
        day = int(m.group(1))
    if "Night" in phase:
        period = "Night"
    elif "Daytime" in phase:
        period = "Daytime"
    elif "preparation" in phase.lower():
        period = "Preparation"
    return day, period


def safe_parse_format(fmt_str: str) -> Optional[Any]:
    if not isinstance(fmt_str, str) or not fmt_str.strip():
        return None
    try:
        return json.loads(fmt_str)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r"\[([^\[\]\"]*?->[^\[\]\"]*?)\]", r'["\1"]', fmt_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def naturalize_format(fmt_str: str) -> str:
    obj = safe_parse_format(fmt_str)
    if not isinstance(obj, dict):
        return ""
    parts = []
    for group_key, group_val in obj.items():
        if not isinstance(group_val, dict) or not group_val:
            continue
        items = [f"{k}[{','.join(map(str, v)) if isinstance(v, list) else v}]"
                 for k, v in group_val.items()]
        if items:
            parts.append(f"{group_key}：" + "，".join(items) + "。")
    return "".join(parts)


def extract_intents(fmt_str: str):
    obj = safe_parse_format(fmt_str)
    if not isinstance(obj, dict):
        return []
    action = obj.get("动作类型")
    if not isinstance(action, dict):
        return []
    return [k for k in action.keys() if k and k != "无结果"]


def is_meaningful_speech(val: dict) -> bool:
    return bool(
        (val.get("text") or "").strip()
        or (val.get("summary") or "").strip()
        or (val.get("format") or "").strip()
    )


def clean_metadata(md: dict) -> dict:
    out = {}
    for k, v in md.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = ",".join(str(x) for x in v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


# ---------- 构建 game ----------

def build_game_doc(data: dict, game_id: str):
    state = data.get("game_state", {}) or {}
    roles = state.get("roles", {}) or {}
    final = state.get("final", {}) or {}
    result = state.get("Game Result", "")

    roles_str = "，".join(f"{k}号{v}" for k, v in roles.items()) if roles else "无"
    final_str = "，".join(f"{k}号{v}" for k, v in final.items()) if final else "无"

    doc_text = (
        f"对局：{game_id}\n"
        f"角色分配：{roles_str}\n"
        f"最终状态：{final_str}\n"
        f"游戏结果：{result or '未知'}"
    )

    return {
        "id": f"{game_id}:game",
        "document": doc_text,
        "metadata": {
            "game_id": game_id,
            "game_result": result or "Unknown",
            "doc_type": "game",
        },
    }


# ---------- 收集全部阶段 ----------

def collect_all_phases(data: dict) -> list:
    phases: list = []
    seen = set()

    def add(p: str):
        if p and p not in seen:
            seen.add(p)
            phases.append(p)

    for item in data.get("task", []):
        add(normalize_phase(item.get("phase", "")))
    for key in (data.get("audio", {}) or {}):
        phase, _ = parse_phase_and_event(key)
        add(phase)
    for key, val in (data.get("game_state", {}) or {}).items():
        if key in RESERVED_TOP_KEYS:
            continue
        if isinstance(val, dict):
            add(normalize_phase(key))
    return phases


# ---------- 构建 phase ----------

def build_phase_docs(data: dict, game_id: str):
    game_doc_id = f"{game_id}:game"

    phase_events: dict[str, list] = {}
    for item in data.get("task", []):
        p = normalize_phase(item.get("phase", ""))
        if p:
            phase_events[p] = item.get("events", [])

    phase_to_speeches: dict[str, list] = {}
    for key, val in (data.get("audio", {}) or {}).items():
        phase, event = parse_phase_and_event(key)
        if not phase or not is_meaningful_speech(val):
            continue
        phase_to_speeches.setdefault(phase, []).append({
            "event": event,
            "val": val,
            "speaker": extract_speaker(event),
        })

    phase_docs = []
    for phase in collect_all_phases(data):
        speeches = phase_to_speeches.get(phase, [])
        events = phase_events.get(phase, [])
        events_str = "；".join(events) if events else "无公开事件。"

        speakers = [s["speaker"] for s in speeches if s["speaker"] is not None]
        speakers_str = "、".join(f"{s}号" for s in speakers) if speakers else "无"

        summary_lines = [
            f"{s['speaker']}号：{(s['val'].get('summary') or '').strip()}"
            for s in speeches
            if (s['val'].get("summary") or "").strip()
        ]
        summary_str = "\n".join(summary_lines) if summary_lines else "无发言摘要。"

        doc_text = (
            f"对局：{game_id}\n"
            f"阶段：{phase}\n"
            f"公开事件：{events_str}\n"
            f"本阶段发言者：{speakers_str}\n"
            f"阶段内发言摘要：\n{summary_str}"
        )

        day, period = parse_phase_parts(phase)
        phase_docs.append({
            "id": f"{game_id}:phase:{phase}",
            "document": doc_text,
            "metadata": {
                "game_id": game_id,
                "parent_id": game_doc_id,
                "phase": phase,
                "day": day if day is not None else -1,
                "period": period or "Unknown",
                "speech_count": len(speeches),
                "doc_type": "phase",
            },
            "phase": phase,
            "speeches": speeches,
        })

    return phase_docs


# ---------- 构建 utterance ----------

def build_utterance_docs(phase_docs: list, data: dict, game_id: str):
    roles = data.get("game_state", {}).get("roles", {}) or {}

    utterance_docs = []
    for phase_doc in phase_docs:
        phase = phase_doc["phase"]
        parent_id = phase_doc["id"]
        for idx, s in enumerate(phase_doc["speeches"]):
            speaker = s["speaker"]
            val = s["val"]
            summary = (val.get("summary") or "").strip()
            naturalized = naturalize_format(val.get("format") or "")
            intents = extract_intents(val.get("format") or "")
            role = roles.get(str(speaker)) if speaker is not None else None

            doc_text = (
                f"阶段：{phase}\n"
                f"发言者：{speaker}号\n"
                f"视角：{role or '未知'}\n"
                f"意图：{'、'.join(intents) if intents else '未标注'}\n"
                f"摘要：{summary}\n"
                f"结构化观点：{naturalized if naturalized else '无'}"
            )

            utterance_docs.append({
                "id": f"{game_id}:utt:{phase}:{idx}:{speaker}",
                "document": doc_text,
                "metadata": {
                    "game_id": game_id,
                    "parent_id": parent_id,
                    "phase": phase,
                    "speaker": speaker if speaker is not None else -1,
                    "offline_true_role": role or "Unknown",
                    "intents": intents,
                    "doc_type": "utterance",
                    "duration": float(val.get("duration", 0.0)),
                    "rms": float(val.get("rms", 0.0)),
                },
            })

    return utterance_docs


# ---------- 构建 skill_action（固定技能表） ----------

def build_skill_docs(phase_docs: list, data: dict, game_id: str):
    state = data.get("game_state", {}) or {}
    phase_lookup = {pd["phase"]: pd["id"] for pd in phase_docs}

    skill_docs = []
    for phase_key, phase_data in state.items():
        if phase_key in RESERVED_TOP_KEYS:
            continue
        if not isinstance(phase_data, dict):
            continue
        phase = normalize_phase(phase_key)
        parent_id = phase_lookup.get(phase)
        if not parent_id:
            continue

        for key, target in phase_data.items():
            if key not in SKILLS:
                continue
            if isinstance(target, (list, dict, bool)) or target is None:
                continue
            target_str = str(target).strip()
            if not target_str.lstrip("-").isdigit():
                continue

            target_val = int(target_str)
            actor_role, action, is_self = SKILLS[key]

            if is_self:
                if target_val > 0:
                    doc_text = f"阶段：{phase}\n行动：{target_val}号{actor_role}{action}"
                else:
                    doc_text = f"阶段：{phase}\n行动：{action}（未发生）"
            else:
                if target_val > 0:
                    doc_text = (
                        f"阶段：{phase}\n"
                        f"行动者：{actor_role}\n"
                        f"行动：{action} {target_val}号"
                    )
                else:
                    doc_text = (
                        f"阶段：{phase}\n"
                        f"行动者：{actor_role}\n"
                        f"行动：{action}（未执行）"
                    )

            skill_docs.append({
                "id": f"{game_id}:skill:{phase}:{key}:{target_str}",
                "document": doc_text,
                "metadata": {
                    "game_id": game_id,
                    "parent_id": parent_id,
                    "phase": phase,
                    "action_name": key,
                    "action": action,
                    "actor_role": actor_role,
                    "target": target_val,
                    "is_self_action": is_self,
                    "doc_type": "skill_action",
                },
            })

    return skill_docs


# ---------- 处理单文件 ----------

def process_one_file(path: Path):
    game_id = path.stem
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, game_id, f"读取失败: {e}"

    if not isinstance(data, dict):
        return None, game_id, "JSON 顶层不是对象"

    try:
        game_doc = build_game_doc(data, game_id)
        phase_docs = build_phase_docs(data, game_id)
        utterance_docs = build_utterance_docs(phase_docs, data, game_id)
        skill_docs = build_skill_docs(phase_docs, data, game_id)
    except Exception as e:
        return None, game_id, f"构建失败: {e}\n{traceback.format_exc()}"

    return {
        GAME_COLLECTION: [game_doc],
        PHASE_COLLECTION: phase_docs,
        UTTERANCE_COLLECTION: utterance_docs,
        SKILL_COLLECTION: skill_docs,
    }, game_id, None


# ---------- Chroma 客户端 ----------

def get_chroma_client(local_path: Optional[str] = None):
    if local_path:
        Path(local_path).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=local_path)
    if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE:
        raise SystemExit(
            "请在 .env 中配置 CHROMA_API_KEY / CHROMA_TENANT / CHROMA_DATABASE"
            "（或用 --local_path 走本地库）"
        )
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )


# ---------- 写入 ----------

def upsert_collection(client, name: str, docs: list,
                      embed_client: LocalEmbeddingClient, upsert_batch: int = 256):
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(docs)
    if total == 0:
        print(f"  [{name}] 无数据，跳过")
        return collection

    texts = [d["document"] for d in docs]
    print(f"  [{name}] 编码 {total} 条 ...")
    t0 = time.time()
    embeddings = embed_client.encode(texts)
    enc_sec = time.time() - t0

    print(f"  [{name}] 写入 {total} 条 ...")
    t0 = time.time()
    for i in range(0, total, upsert_batch):
        j = min(i + upsert_batch, total)
        for attempt in range(1, 4):
            try:
                collection.upsert(
                    ids=[d["id"] for d in docs[i:j]],
                    documents=texts[i:j],
                    metadatas=[clean_metadata(d["metadata"]) for d in docs[i:j]],
                    embeddings=embeddings[i:j],
                )
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"    [{name}] 写入失败({attempt}/3): {e}，3s 后重试")
                time.sleep(3)
        print(f"  [{name}] {j}/{total}")
    wrt_sec = time.time() - t0
    print(f"  [{name}] 完成：编码 {enc_sec:.1f}s / 写库 {wrt_sec:.1f}s")

    return collection


# ---------- 父子召回 ----------

CHILD_COLLECTIONS = [UTTERANCE_COLLECTION, SKILL_COLLECTION]


def retrieve(client, embed_client: LocalEmbeddingClient, query: str,
             top_k: int = 10, max_phases: int = 3, max_games: int = 2,
             filters: Optional[dict] = None,
             child_collections: Optional[list] = None):
    child_collections = child_collections or CHILD_COLLECTIONS
    q_emb = embed_client.encode([query])

    by_phase: dict[str, list] = {}
    for name in child_collections:
        try:
            col = client.get_collection(name, embedding_function=None)
        except Exception:
            continue
        if col.count() == 0:
            continue

        kwargs = {
            "query_embeddings": q_emb,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        try:
            res = col.query(**kwargs)
        except Exception as e:
            print(f"  [{name}] 查询失败：{e}")
            continue

        for cid, doc, md, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            pid = md.get("parent_id", "")
            if not pid:
                continue
            by_phase.setdefault(pid, []).append({
                "id": cid,
                "doc": doc,
                "md": md,
                "dist": dist,
                "collection": name,
            })

    if not by_phase:
        return []

    sorted_phases = sorted(by_phase.items(), key=lambda kv: min(h["dist"] for h in kv[1]))
    phase_ids = [pid for pid, _ in sorted_phases[:max_phases]]

    phase_col = client.get_collection(PHASE_COLLECTION, embedding_function=None)
    phases = phase_col.get(ids=phase_ids, include=["documents", "metadatas"])

    by_game: dict[str, list] = {}
    phase_info = {}
    for pid, pdoc, pmd in zip(phases["ids"], phases["documents"], phases["metadatas"]):
        gid = pmd.get("parent_id", "")
        by_game.setdefault(gid, []).append(pid)
        phase_info[pid] = {"doc": pdoc, "meta": pmd}

    game_ids = list(by_game.keys())[:max_games]
    if game_ids:
        game_col = client.get_collection(GAME_COLLECTION, embedding_function=None)
        games = game_col.get(ids=game_ids, include=["documents", "metadatas"])
    else:
        games = {"ids": [], "documents": [], "metadatas": []}

    game_info = {
        gid: {"doc": gdoc, "meta": gmd}
        for gid, gdoc, gmd in zip(games["ids"], games["documents"], games["metadatas"])
    }

    output = []
    for gid, pid_list in by_game.items():
        phase_entries = []
        for pid in pid_list:
            pinfo = phase_info.get(pid, {})
            hits = by_phase.get(pid, [])
            grouped = {UTTERANCE_COLLECTION: [], SKILL_COLLECTION: []}
            for h in hits:
                grouped.setdefault(h["collection"], []).append(h)
            for k in grouped:
                grouped[k].sort(key=lambda x: x["dist"])
            phase_entries.append({
                "phase_id": pid,
                "phase_doc": pinfo.get("doc", ""),
                "phase_meta": pinfo.get("meta", {}),
                "hits": grouped,
            })
        output.append({
            "game_id": gid,
            "game_doc": game_info.get(gid, {}).get("doc", ""),
            "game_meta": game_info.get(gid, {}).get("meta", {}),
            "phases": phase_entries,
        })
    return output


# ---------- 输入收集 ----------

def collect_input_files(args):
    files = []
    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            raise SystemExit(f"--input_dir 不是目录: {d}")
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() == ".json":
                files.append(p)
    if args.input:
        p = Path(args.input)
        if not p.is_file():
            raise SystemExit(f"--input 不是文件: {p}")
        files.append(p)

    seen, unique = set(), []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    if not unique:
        raise SystemExit("没找到任何 .json 文件")

    # 随机打乱顺序（可选，配合 --seed 可复现）
    if getattr(args, "random", False):
        seed = getattr(args, "seed", None)
        rng = random.Random(seed)
        rng.shuffle(unique)
        print(f"  随机打乱文件顺序" + (f"（seed={seed}）" if seed is not None else ""))

    # 只取前 N 个文件（N <= 0 表示不限制）
    limit = getattr(args, "limit", None)
    if limit is not None and limit > 0 and len(unique) > limit:
        print(f"  --limit {limit}：共 {len(unique)} 个 json，仅处理前 {limit} 个")
        unique = unique[:limit]

    return unique


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="最多处理 N 个 json 文件（默认按文件名排序取前 N），<=0 或不传表示不限制")
    parser.add_argument("--random", action="store_true",
                        help="随机选取文件（配合 --limit 使用），默认按文件名排序")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，填了则每次抽到的文件一致（可复现）")
    parser.add_argument("--local_path", default=None, help="用本地 Chroma 而不是云端")
    parser.add_argument("--reset", action="store_true", help="删除并重建所有 collection")
    parser.add_argument("--no_demo", action="store_true", help="跳过召回演示")
    parser.add_argument("--upsert_batch", type=int, default=256)
    parser.add_argument("--progress", type=int, default=100)
    args = parser.parse_args()

    if not args.input_dir and not args.input:
        raise SystemExit("请提供 --input_dir 或 --input")

    files = collect_input_files(args)
    print(f"待处理文件 {len(files)} 个")

    embed_client = LocalEmbeddingClient(
        model_name=EMBED_MODEL,
        device=EMBED_DEVICE,
        batch_size=EMBED_BATCH_SIZE,
    )

    print("\n连接 Chroma" + ("（本地）" if args.local_path else " Cloud") + " ...")
    client = get_chroma_client(args.local_path)
    if not args.local_path:
        print(f"  tenant={CHROMA_TENANT}  database={CHROMA_DATABASE}")

    if args.reset:
        for name in (GAME_COLLECTION, PHASE_COLLECTION,
                     UTTERANCE_COLLECTION, SKILL_COLLECTION):
            try:
                client.delete_collection(name)
                print(f"  已删除旧 collection: {name}")
            except Exception:
                pass

    buckets = {name: [] for name in
               (GAME_COLLECTION, PHASE_COLLECTION,
                UTTERANCE_COLLECTION, SKILL_COLLECTION)}

    fail_list = []
    t_start = time.time()

    for i, path in enumerate(files, start=1):
        docs, game_id, err = process_one_file(path)
        if err:
            print(f"  [跳过] {path.name}: {err.splitlines()[0]}")
            fail_list.append((path.name, err))
            continue

        for name, lst in docs.items():
            buckets[name].extend(lst)

        if args.progress and i % args.progress == 0:
            counts = " ".join(f"{n}={len(buckets[n])}" for n in buckets)
            print(f"  ... {i}/{len(files)} 文件 | {counts}")

    print(f"\n解析完成：成功 {len(files) - len(fail_list)} 局，失败 {len(fail_list)} 局")
    for name in buckets:
        print(f"  {name}: {len(buckets[name])} 条")

    for name in (GAME_COLLECTION, PHASE_COLLECTION,
                 UTTERANCE_COLLECTION, SKILL_COLLECTION):
        print(f"\n写入 {name} ...")
        upsert_collection(client, name, buckets[name], embed_client, args.upsert_batch)

    elapsed = time.time() - t_start
    total = sum(len(v) for v in buckets.values())
    print(f"\n完成。总 {total} 条，耗时 {elapsed/60:.1f} 分钟")

    if fail_list:
        bad = Path("ingest_failed.jsonl")
        with bad.open("w", encoding="utf-8") as f:
            for name, err in fail_list:
                f.write(json.dumps({"file": name, "error": err.splitlines()[0]},
                                   ensure_ascii=False) + "\n")
        print(f"失败文件已写 {bad}")

    if not args.no_demo and total:
        demo_query = (
            "阶段：Day 2 Night\n"
            "行动：狼人刀人\n"
            "摘要：夜晚狼人选择击杀目标"
        )
        print("\n===== 父子召回示例 =====")
        results = retrieve(client, embed_client, demo_query,
                           top_k=10, max_phases=3, max_games=2)
        for g in results:
            print(f"\n[game] {g['game_id']}")
            print(g["game_doc"][:200], "...")
            for ph in g["phases"]:
                print(f"\n  [phase] {ph['phase_id']}")
                print("  " + ph["phase_doc"][:200].replace("\n", "\n  "), "...")
                for col_name, hits in ph["hits"].items():
                    if not hits:
                        continue
                    print(f"    命中 {col_name} {len(hits)} 条：")
                    for c in hits[:3]:
                        md = c["md"]
                        print(f"      - {md.get('doc_type')} "
                              f"speaker={md.get('speaker', md.get('actor_role', '?'))} "
                              f"距离={c['dist']:.4f}")
                        print(f"        {c['doc'][:150]}")


if __name__ == "__main__":
    main()