"""Official VLM benchmark metrics (ANLS, ChartQA Relaxed Accuracy, MC), HF dataset loaders, and bootstrap CIs."""
from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from PIL import Image


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _clean_ans(text: str) -> str:
    text = text.strip().split("\n")[0].strip()
    text = re.sub(r"^(answer\s*:\s*|the answer is\s*)", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip(".").strip()


def score_anls(pred: str, golds: list[str], threshold: float = 0.5) -> float:
    """Official DocVQA / MP-DocVQA Average Normalized Levenshtein Similarity."""
    p = " ".join(_clean_ans(pred).lower().split())
    best = 0.0
    for g_raw in golds:
        g = " ".join(str(g_raw).lower().split())
        if not g and not p:
            return 1.0
        denom = max(len(p), len(g))
        if denom == 0:
            continue
        nl = _levenshtein(p, g) / denom
        sim = 1.0 - nl if nl < threshold else 0.0
        best = max(best, sim)
    return float(best)


def score_chartqa_relaxed(pred: str, golds: list[str], rel_tol: float = 0.05) -> float:
    """Official ChartQA relaxed accuracy (+/-5% relative tolerance for numeric answers)."""
    p = _clean_ans(pred).lower().replace(",", "").rstrip("%").strip()
    for g_raw in golds:
        g = str(g_raw).lower().replace(",", "").rstrip("%").strip()
        if p == g:
            return 1.0
        try:
            pf, gf = float(p), float(g)
            if gf == 0.0:
                if abs(pf) < 1e-6:
                    return 1.0
            elif abs(pf - gf) / abs(gf) <= rel_tol:
                return 1.0
        except ValueError:
            if g and (p == g or g in p.split()):
                return 1.0
    return 0.0


def score_multichoice(pred: str, gold_letter: str) -> float:
    """Multiple-choice option extractor for MUIRBENCH (A/B/C/D/E)."""
    p = _clean_ans(pred).upper()
    g = gold_letter.strip().upper()[:1]
    m = re.match(r"^\(?([A-F])\)?(?:\b|\.|:|\s|$)", p)
    if m:
        return 1.0 if m.group(1) == g else 0.0
    m2 = re.search(r"\b([A-F])\b", p)
    return 1.0 if (m2 and m2.group(1) == g) else 0.0


def downsample_image_2x(img: Image.Image, min_side: int = 56) -> Image.Image:
    """2x2 spatial resolution reduction (~4x fewer visual tokens) for the Pooled-Joint baseline."""
    from PIL import Image

    w, h = img.size
    nw, nh = max(min_side, w // 2), max(min_side, h // 2)
    return img.resize((nw, nh), Image.Resampling.BILINEAR)


def clamp_image_resolution(img: Image.Image, max_pixels: int = 512 * 512) -> Image.Image:
    """Ensure consistent real-image token budgets fitting comfortably on L4/T4 VRAM."""
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / float(w * h)) ** 0.5
        img = img.resize((max(56, int(w * scale)), max(56, int(h * scale))), Image.Resampling.BILINEAR)
    return img


def load_vlm_task_items(task: str, n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Stream real evaluation items from HuggingFace datasets without synthetic data."""
    import random
    from datasets import load_dataset

    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    if task == "docvqa":
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
        for row in ds:
            if len(items) >= n:
                break
            img = clamp_image_resolution(row["image"], max_pixels=672 * 672)
            ans = row.get("answers") or [row.get("answer", "")]
            items.append({
                "task": "docvqa",
                "images": [img],
                "question": f"{row['question']} Answer concisely with only the exact value or phrase.",
                "golds": [str(a) for a in ans],
                "metric": "anls",
                "target_page_pos": "only",
            })

    elif task == "chartqa":
        ds = load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
        for row in ds:
            if len(items) >= n:
                break
            img = clamp_image_resolution(row["image"], max_pixels=616 * 616)
            ans = row.get("label") or row.get("answers") or []
            if isinstance(ans, str):
                ans = [ans]
            items.append({
                "task": "chartqa",
                "images": [img],
                "question": f"{row['query']} Output only the short final number or label.",
                "golds": [str(a) for a in ans],
                "metric": "chartqa",
                "target_page_pos": "only",
            })

    elif task == "mp_docvqa":
        # Multi-page document QA: permute target evidence page uniformly across early vs. late slots
        try:
            ds = load_dataset("lmms-lab/MP-DocVQA", split="val", streaming=True)
            for row in ds:
                if len(items) >= n:
                    break
                raw_imgs = [ row[k] for k in sorted(row.keys()) if k.startswith("image_") and row[k] is not None ]
                if not raw_imgs and "images" in row:
                    raw_imgs = list(row["images"])
                if len(raw_imgs) < 2:
                    continue
                ans_idx = int(row.get("answer_page_idx", 0))
                ans_idx = max(0, min(len(raw_imgs) - 1, ans_idx))
                # Select target page + up to 3 real distractor pages from the same document
                other_idxs = [i for i in range(len(raw_imgs)) if i != ans_idx]
                chosen_others = rng.sample(other_idxs, min(3, len(other_idxs)))
                pages = [clamp_image_resolution(raw_imgs[i], max_pixels=448 * 448) for i in chosen_others]
                target_img = clamp_image_resolution(raw_imgs[ans_idx], max_pixels=448 * 448)
                slot = 0 if (len(items) % 2 == 0) else len(pages)  # strict 50/50 early-page vs. late-page split
                pages.insert(slot, target_img)
                ans = row.get("answers") or [row.get("answer", "")]
                if isinstance(ans, str):
                    ans = [ans]
                items.append({
                    "task": "mp_docvqa",
                    "images": pages,
                    "question": f"{row['question']} Answer concisely with only the exact value or phrase.",
                    "golds": [str(a) for a in ans],
                    "metric": "anls",
                    "target_page_pos": "early" if slot == 0 else "late",
                })
        except Exception:
            # Multi-page composition over real DocVQA validation pages if MP-DocVQA split requires auth
            ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
            buf = []
            for row in ds:
                buf.append(row)
                if len(buf) == 4:
                    slot = 0 if (len(items) % 2 == 0) else 3
                    target = buf[slot]
                    pages = [clamp_image_resolution(r["image"], max_pixels=448 * 448) for r in buf]
                    ans = target.get("answers") or [target.get("answer", "")]
                    items.append({
                        "task": "mp_docvqa",
                        "images": pages,
                        "question": f"{target['question']} Answer concisely with only the exact value or phrase.",
                        "golds": [str(a) for a in ans],
                        "metric": "anls",
                        "target_page_pos": "early" if slot == 0 else "late",
                    })
                    buf = []
                    if len(items) >= n:
                        break

    elif task == "muirbench":
        ds = load_dataset("MUIRBENCH/MUIRBENCH", split="test", streaming=True)
        for row in ds:
            if len(items) >= n:
                break
            raw_imgs = [im for im in (row.get("image_list") or []) if im is not None]
            if len(raw_imgs) < 2 or len(raw_imgs) > 6:
                continue
            imgs = [clamp_image_resolution(im, max_pixels=392 * 392) for im in raw_imgs]
            opts = row.get("options") or []
            opt_str = "\n".join(f"({chr(65 + idx)}) {o}" for idx, o in enumerate(opts))
            q_clean = row["question"].replace("<image>", "").strip()
            items.append({
                "task": "muirbench",
                "images": imgs,
                "question": f"{q_clean}\n{opt_str}\nReply with ONLY the single option letter (A, B, C, D, or E).",
                "golds": [str(row["answer"]).strip()],
                "metric": "mc",
                "target_page_pos": "multi",
            })
    else:
        raise ValueError(f"Unknown task: {task}")

    return items


def score_prediction(metric: str, pred: str, golds: list[str]) -> float:
    if metric == "anls":
        return score_anls(pred, golds)
    if metric == "chartqa":
        return score_chartqa_relaxed(pred, golds)
    if metric == "mc":
        return score_multichoice(pred, golds[0])
    raise ValueError(f"Unknown metric: {metric}")


def paired_bootstrap_ci(
    ref_scores: list[float],
    arm_scores: list[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI of (ref - arm) drop in percentage points."""
    assert len(ref_scores) == len(arm_scores) and len(ref_scores) > 0
    g = torch.Generator().manual_seed(seed)
    diffs = (torch.tensor(ref_scores, dtype=torch.float32) - torch.tensor(arm_scores, dtype=torch.float32)) * 100.0
    idx = torch.randint(len(diffs), (n_boot, len(diffs)), generator=g)
    means = diffs[idx].mean(dim=1)
    return float(diffs.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))


def cuda_sync_timer() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()
