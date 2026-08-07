import pytest
torch = pytest.importorskip("torch")
import pandas as pd

from shore.adaptation import (
    PairwiseTextDataset,
    build_same_job_pairs,
    freeze_for_preservation,
    preservation_objective,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
        self.classifier = torch.nn.Linear(2, 1)


def test_same_job_pairs_are_built_from_satisfied_and_rejected():
    frame = pd.DataFrame({
        "jd_no": ["j", "j", "j"],
        "user_id": ["p", "n1", "n2"],
        "confit_label": [1, 0, 0],
        "job_text": ["job"] * 3,
        "resume_text": ["pos", "neg1", "neg2"],
    })
    pairs = build_same_job_pairs(frame)
    assert len(pairs) == 2
    assert set(pairs.positive_user_id) == {"p"}


def test_pairwise_text_dataset_returns_text_triples():
    frame = pd.DataFrame({"job_text": ["job"], "positive_text": ["pos"], "negative_text": ["neg"]})
    item = PairwiseTextDataset(frame)[0]
    assert item == ("job", "pos", "neg")


def test_preservation_freezing_unfreezes_last_block_and_head():
    model = TinyModel()
    ratio = freeze_for_preservation(model)
    assert not any(p.requires_grad for p in model.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in model.encoder.layer[-1].parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert 0 < ratio < 1


def test_preservation_objective_combines_rank_and_teacher_margin():
    pos = torch.tensor([1.0, 0.0])
    neg = torch.tensor([0.0, 1.0])
    teacher = torch.tensor([0.5, -0.5])
    loss = preservation_objective(pos, neg, teacher, distill_weight=2.0)
    assert loss.item() > 0
