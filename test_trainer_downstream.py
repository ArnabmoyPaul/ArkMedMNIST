"""test_trainer_downstream.py — Run: python test_trainer_downstream.py"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trainer import train_downstream_epoch


class _TinyArkLike(nn.Module):
    """Mimics ArkR3D/ArkSwinTransformer's forward(x, head_n) -> (feat, logits)
    contract with a trivial linear encoder, so this test has no GPU/timm
    dependency and works for either modality's model shape."""
    def __init__(self, in_dim=12, feat_dim=4, num_classes=3):
        super().__init__()
        self.enc = nn.Linear(in_dim, feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x, head_n=None):
        feat = self.enc(x.flatten(1))
        return feat, self.head(feat)


def test_train_downstream_epoch_updates_weights_and_returns_finite_loss():
    torch.manual_seed(0)
    model = _TinyArkLike()
    x1 = torch.randn(16, 3, 2, 2)
    x2 = torch.randn(16, 3, 2, 2)  # second (teacher) view -- must be ignored
    y = torch.eye(3)[torch.randint(0, 3, (16,))]
    loader = DataLoader(TensorDataset(x1, x2, y), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.enc.weight.clone()

    avg_loss = train_downstream_epoch(model, 0, "tinyset", loader, torch.device('cpu'),
                                       nn.CrossEntropyLoss(), optimizer, epoch=0, scaler=None)

    assert not torch.allclose(before, model.enc.weight), "weights should have updated"
    assert avg_loss == avg_loss and avg_loss > 0, f"expected a finite positive loss, got {avg_loss}"


def test_train_downstream_epoch_ignores_second_view():
    torch.manual_seed(0)
    model = _TinyArkLike()
    x1 = torch.randn(16, 3, 2, 2)
    x2 = torch.randn(16, 3, 2, 2) * 1000
    y = torch.eye(3)[torch.randint(0, 3, (16,))]
    loader = DataLoader(TensorDataset(x1, x2, y), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    avg_loss = train_downstream_epoch(model, 0, "tinyset", loader, torch.device('cpu'),
                                       nn.CrossEntropyLoss(), optimizer, epoch=0, scaler=None)
    assert avg_loss < 100, f"loss exploded ({avg_loss}) -- second view leaked into training"


if __name__ == "__main__":
    test_train_downstream_epoch_updates_weights_and_returns_finite_loss()
    test_train_downstream_epoch_ignores_second_view()
    print("test_trainer_downstream.py: all checks passed")
