"""test_models_medmnist3d.py -- self-check for ArkR3D (models.py) and the
MedMNIST3DWrapper label-encoding fix (medmnist3d_dataloader.py).
Run: python test_models_medmnist3d.py"""
import torch
from optparse import Values
from models import ArkR3D, build_omni_model_3d, load_kinetics_backbone


def test_arkr3d_forward_survives_native_28cubed_volume():
    # The one real technical risk in adding r3d_18 as the 3D encoder: does its
    # stride pattern collapse a native 28x28x28 MedMNIST volume to a 0-sized
    # spatial map partway through (r3d_18 was designed for Kinetics clips,
    # much larger spatially/temporally)? Confirmed empirically during planning
    # that it doesn't (AdaptiveAvgPool3d(1) tolerates whatever's left), but
    # this is exactly the kind of silent-breakage risk (same class as the Swin
    # window-shrink bug) that deserves a standing check, not just a one-off.
    model = ArkR3D([11, 2], projector_features=None, use_mlp=False)
    x = torch.randn(2, 1, 28, 28, 28)
    feat, logits = model(x, head_n=0)
    assert feat.shape == (2, 512), f"expected pooled (B,512) features, got {feat.shape}"
    assert logits.shape == (2, 11)


def test_load_kinetics_backbone_transfers_encoder_not_heads():
    model = ArkR3D([11, 2], projector_features=None, use_mlp=False)
    head_before = model.omni_heads[0].weight.clone()
    load_kinetics_backbone(model)  # must not raise
    assert torch.allclose(model.omni_heads[0].weight, head_before), \
        "load_kinetics_backbone must not touch omni_heads (strict=False, heads have no matching key)"
    assert model.encoder.stem[0].weight.shape == (64, 1, 3, 7, 7), \
        "stem must stay 1-channel (grayscale volumes) after loading 3-channel Kinetics weights"


def test_build_omni_model_3d_end_to_end():
    args = Values()
    args.projector_features = 128
    args.use_mlp = False
    args.pretrained_weights_3d = None
    model = build_omni_model_3d(args, num_classes_list=[11, 2])
    x = torch.randn(1, 1, 28, 28, 28)
    feat, logits = model(x, head_n=1)
    assert feat.shape == (1, 128)
    assert logits.shape == (1, 2)


if __name__ == "__main__":
    test_arkr3d_forward_survives_native_28cubed_volume()
    test_load_kinetics_backbone_transfers_encoder_not_heads()
    test_build_omni_model_3d_end_to_end()
    print("test_models_medmnist3d.py: all checks passed")
