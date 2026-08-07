from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def build_unique_pair_table(*frames: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.concat([f[["jd_no", "user_id"]] for f in frames], ignore_index=True)
    return pairs.astype(str).drop_duplicates().sort_values(["jd_no", "user_id"]).reset_index(drop=True)


def merge_pair_scores(df: pd.DataFrame, pairs: pd.DataFrame, scores: np.ndarray, score_col: str = "score") -> pd.DataFrame:
    if len(pairs) != len(scores):
        raise ValueError("pair and score lengths differ")
    scored = pairs.copy()
    scored[score_col] = np.asarray(scores, dtype=np.float32)
    out = df.astype({"jd_no": str, "user_id": str}).merge(scored, on=["jd_no", "user_id"], how="left", validate="many_to_one")
    if out[score_col].isna().any():
        raise ValueError("missing reranker scores after merge")
    return out


def score_unique_pairs(
    pairs: pd.DataFrame,
    job_text: dict[str, str],
    resume_text: dict[str, str],
    scorer: object,
    output_path: str | Path,
    batch_size: int = 32,
    force: bool = False,
) -> np.ndarray:
    output_path = Path(output_path)
    if output_path.exists() and not force:
        scores = np.load(output_path)
        if len(scores) != len(pairs):
            raise ValueError("cached reranker score length mismatch")
        return scores
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), batch_size):
        batch = pairs.iloc[start : start + batch_size]
        text_pairs = [(job_text[str(j)], resume_text[str(u)]) for j, u in zip(batch.jd_no, batch.user_id)]
        values[start : start + len(batch)] = np.asarray(scorer.predict(text_pairs), dtype=np.float32).reshape(-1)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, values)
    temp.replace(output_path)
    return values


@dataclass
class CrossEncoderReranker:
    model_name: str
    batch_size: int = 32
    max_length: int = 512
    model: object | None = None

    def _model(self):
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install the gpu extra to run reranking") from exc
        self.model = CrossEncoder(self.model_name, num_labels=1, max_length=self.max_length, trust_remote_code=True)
        return self.model

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        values = self._model().predict(list(pairs), batch_size=self.batch_size, show_progress_bar=True)
        return np.asarray(values, dtype=np.float32).reshape(-1)


@dataclass
class QwenCausalReranker:
    model_name: str
    batch_size: int = 8
    max_length: int = 512
    instruction: str = "Given a job posting, determine whether the resume is suitable for the role."
    model: object | None = None
    tokenizer: object | None = None

    @staticmethod
    def yes_probability(logits: np.ndarray, yes_index: int, no_index: int) -> np.ndarray:
        selected = np.asarray(logits, dtype=np.float64)[:, [no_index, yes_index]]
        selected -= selected.max(axis=1, keepdims=True)
        probs = np.exp(selected)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1].astype(np.float32)

    def _load(self):
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the gpu extra to run Qwen reranking") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model.eval()
        return self.model, self.tokenizer

    def _prompt(self, job: str, resume: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {job}\n"
            f"<Document>: {resume}\n"
            "<Answer>:"
        )

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        import torch

        model, tokenizer = self._load()
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        no_ids = tokenizer.encode("no", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
            no_ids = tokenizer.encode("No", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise RuntimeError("Qwen yes/no labels must each map to one token")
        outputs: list[np.ndarray] = []
        for start in range(0, len(pairs), self.batch_size):
            prompts = [self._prompt(a, b) for a, b in pairs[start : start + self.batch_size]]
            batch = tokenizer(prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.inference_mode():
                logits = model(**batch).logits[:, -1, :]
            selected = torch.stack([logits[:, no_ids[0]], logits[:, yes_ids[0]]], dim=1)
            outputs.append(torch.softmax(selected.float(), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(outputs).astype(np.float32)
