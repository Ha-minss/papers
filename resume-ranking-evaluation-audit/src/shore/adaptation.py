from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def pairwise_ranknet_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(-(positive_scores - negative_scores)).mean()


def preservation_loss(new_scores: torch.Tensor, reference_scores: torch.Tensor, weight: float) -> torch.Tensor:
    return float(weight) * torch.nn.functional.mse_loss(new_scores, reference_scores.detach())


def preservation_objective(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    teacher_margins: torch.Tensor,
    distill_weight: float,
) -> torch.Tensor:
    margins = positive_scores - negative_scores
    return pairwise_ranknet_loss(positive_scores, negative_scores) + preservation_loss(margins, teacher_margins, distill_weight)


def evaluate_checkpoint_gate(
    hard_accuracy: float,
    random_accuracy: float,
    ndcg10: float,
    baseline: dict[str, float],
    max_general_drop: float = 0.01,
) -> bool:
    return (
        hard_accuracy > baseline["hard_accuracy"]
        and random_accuracy >= baseline["random_accuracy"] - max_general_drop
        and ndcg10 >= baseline["ndcg10"] - max_general_drop
    )


def build_same_job_pairs(frame: pd.DataFrame, label_col: str = "confit_label") -> pd.DataFrame:
    required = {"jd_no", "user_id", "job_text", "resume_text", label_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing pair columns: {sorted(missing)}")
    rows: list[dict] = []
    for job, group in frame.groupby("jd_no", sort=False):
        positives = group[group[label_col].astype(int).eq(1)]
        negatives = group[group[label_col].astype(int).eq(0)]
        for _, pos in positives.iterrows():
            for _, neg in negatives.iterrows():
                rows.append({
                    "jd_no": str(job),
                    "job_text": str(pos.job_text),
                    "positive_user_id": str(pos.user_id),
                    "negative_user_id": str(neg.user_id),
                    "positive_text": str(pos.resume_text),
                    "negative_text": str(neg.resume_text),
                })
    return pd.DataFrame(rows)


class PairwiseTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        base = (str(row.job_text), str(row.positive_text), str(row.negative_text))
        if "teacher_margin" in self.frame.columns:
            return (*base, float(row.teacher_margin))
        return base


def _find_longest_module_list(model: torch.nn.Module) -> torch.nn.ModuleList | None:
    candidates = [module for module in model.modules() if isinstance(module, torch.nn.ModuleList) and len(module) > 0]
    return max(candidates, key=len) if candidates else None


def freeze_for_preservation(model: torch.nn.Module) -> float:
    for parameter in model.parameters():
        parameter.requires_grad = False
    blocks = _find_longest_module_list(model)
    if blocks is None:
        raise ValueError("could not locate transformer block list")
    for parameter in blocks[-1].parameters():
        parameter.requires_grad = True
    head_tokens = ("classifier", "score", "classification_head")
    for name, parameter in model.named_parameters():
        if any(token in name.lower() for token in head_tokens):
            parameter.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise ValueError("preservation setup left no trainable parameters")
    return trainable / total


def _cross_encoder_forward(cross_encoder, jobs: list[str], resumes: list[str], device: torch.device) -> torch.Tensor:
    tokenizer = cross_encoder.tokenizer
    model = cross_encoder.model
    batch = tokenizer(jobs, resumes, padding=True, truncation=True, max_length=cross_encoder.max_length, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(**batch).logits
    return logits.reshape(-1)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 2e-5
    batch_size: int = 8
    gradient_accumulation: int = 4
    warmup_ratio: float = 0.1
    distill_weight: float = 1.0
    max_length: int = 512
    seed: int = 20260730


def train_pairwise_cross_encoder(
    model_name: str,
    pairs: pd.DataFrame,
    output_dir: str | Path,
    config: TrainingConfig,
    *,
    preservation: bool = False,
    teacher_margin_fn: Callable[[list[tuple[str, str]], list[tuple[str, str]]], torch.Tensor] | None = None,
    checkpoint_callback: Callable[[object, int], dict] | None = None,
) -> dict:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("Install the gpu extra to train rerankers") from exc

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cross_encoder = CrossEncoder(model_name, num_labels=1, max_length=config.max_length, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cross_encoder.model.to(device)
    trainable_ratio = 1.0
    if preservation:
        trainable_ratio = freeze_for_preservation(cross_encoder.model)
    optimizer = torch.optim.AdamW((p for p in cross_encoder.model.parameters() if p.requires_grad), lr=config.learning_rate)
    loader = DataLoader(PairwiseTextDataset(pairs), batch_size=config.batch_size, shuffle=True)
    history: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        cross_encoder.model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            if len(batch) == 4:
                jobs, pos_text, neg_text, precomputed_teacher = batch
            else:
                jobs, pos_text, neg_text = batch
                precomputed_teacher = None
            pos_scores = _cross_encoder_forward(cross_encoder, list(jobs), list(pos_text), device)
            neg_scores = _cross_encoder_forward(cross_encoder, list(jobs), list(neg_text), device)
            if preservation:
                if precomputed_teacher is not None:
                    teacher = precomputed_teacher.to(device=device, dtype=torch.float32).reshape(-1)
                elif teacher_margin_fn is not None:
                    positive_pairs = list(zip(jobs, pos_text))
                    negative_pairs = list(zip(jobs, neg_text))
                    teacher = teacher_margin_fn(positive_pairs, negative_pairs).to(device).reshape(-1)
                else:
                    raise ValueError("teacher margins are required for preservation training")
                loss = preservation_objective(pos_scores, neg_scores, teacher, config.distill_weight)
            else:
                loss = pairwise_ranknet_loss(pos_scores, neg_scores)
            (loss / config.gradient_accumulation).backward()
            global_step += 1
            if global_step % config.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        if global_step % config.gradient_accumulation:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
        cross_encoder.save_pretrained(str(checkpoint_dir))
        record = {"epoch": epoch, "train_loss": epoch_loss / max(batches, 1), "checkpoint": str(checkpoint_dir)}
        if checkpoint_callback is not None:
            record.update(checkpoint_callback(cross_encoder, epoch))
        history.append(record)
        (output_dir / "training_state.json").write_text(
            json.dumps({"history": history, "trainable_ratio": trainable_ratio}, indent=2, default=float),
            encoding="utf-8",
        )
    return {"history": history, "trainable_ratio": trainable_ratio, "model_dir": str(output_dir)}


def train_pointwise_cross_encoder(
    model_name: str,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    output_dir: str | Path,
    config: TrainingConfig,
) -> dict:
    try:
        from datasets import Dataset as HFDataset
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    except ImportError as exc:
        raise RuntimeError("Install the gpu extra to train pointwise reranker") from exc
    required = {"job_text", "resume_text", "label"}
    for name, frame in [("train", train_frame), ("valid", valid_frame)]:
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} missing {sorted(required - set(frame.columns))}")
    model = CrossEncoder(model_name, num_labels=1, max_length=config.max_length, trust_remote_code=True)
    train_ds = HFDataset.from_pandas(train_frame[["job_text", "resume_text", "label"]], preserve_index=False)
    valid_ds = HFDataset.from_pandas(valid_frame[["job_text", "resume_text", "label"]], preserve_index=False)
    output_dir = Path(output_dir)
    args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=max(config.batch_size, 16),
        gradient_accumulation_steps=config.gradient_accumulation,
        warmup_ratio=config.warmup_ratio,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=25,
        report_to="none",
        seed=config.seed,
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train(resume_from_checkpoint=True if list(output_dir.glob("checkpoint-*")) else None)
    model.save_pretrained(str(output_dir / "final"))
    return {"model_dir": str(output_dir / "final"), "log_history": trainer.state.log_history}
