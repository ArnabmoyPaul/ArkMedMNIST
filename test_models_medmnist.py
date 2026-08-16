"""test_models_medmnist.py — self-check for the MedMNIST-driven additions to
models.py. Run: python test_models_medmnist.py"""
import os
import torch
from models import ArkSwinTransformer, build_omni_model, load_imagenet_backbone, load_finetune_backbone
from optparse import Values


def test_forward_features_pooling_bridges_timm_spatial_output():
    # timm==1.0.28's SwinTransformer.forward_features() returns (B, H, W, C),
    # not the (B, C) vector this class (written against timm==0.5.4) expects
    # before feeding nn.Linear heads -- confirmed via direct smoke test during
    # planning. This must not crash and must return a flat (B, C) vector.
    model = ArkSwinTransformer([9], projector_features=None, use_mlp=False,
                                patch_size=4, window_size=7, embed_dim=96,
                                depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    x = torch.randn(2, 3, 224, 224)
    feat, logits = model(x, head_n=0)
    assert feat.dim() == 2, f"expected pooled (B, C) features, got shape {feat.shape}"
    assert feat.shape == (2, model.num_features)
    assert logits.shape == (2, 9)


def test_generate_embeddings_also_pools():
    model = ArkSwinTransformer([9], projector_features=128, use_mlp=False,
                                patch_size=4, window_size=7, embed_dim=96,
                                depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    x = torch.randn(2, 3, 224, 224)
    emb = model.generate_embeddings(x, after_proj=True)
    assert emb.shape == (2, 128)


def test_build_omni_model_swin_tiny_at_custom_img_size():
    args = Values()
    args.model_name = "swin_tiny"
    args.projector_features = 256
    args.use_mlp = False
    args.crop_size = 112
    args.pretrained_weights = None
    model = build_omni_model(args, num_classes_list=[9, 2])
    x = torch.randn(1, 3, 112, 112)
    feat, logits = model(x, head_n=1)
    assert feat.shape == (1, 256)
    assert logits.shape == (1, 2)
    n_params = sum(p.numel() for p in model.parameters())
    assert 25_000_000 < n_params < 35_000_000, f"unexpected swin_tiny param count: {n_params}"


def test_load_imagenet_backbone_transfers_encoder_not_heads():
    model = ArkSwinTransformer([9], projector_features=None, use_mlp=False,
                                patch_size=4, window_size=7, embed_dim=96,
                                depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    head_before = model.omni_heads[0].weight.clone()
    load_imagenet_backbone(model, "swin_tiny_patch4_window7_224")
    assert torch.allclose(model.omni_heads[0].weight, head_before), \
        "load_imagenet_backbone must not touch omni_heads (strict=False, heads have no matching key)"


def test_load_imagenet_backbone_at_112px_survives_window_shrink():
    # At img_size=112 the last stage's spatial resolution (3x3) is below
    # window_size=7, so timm shrinks that stage's window and its
    # relative_position_bias_table no longer matches the 224px checkpoint's
    # shape -- strict=False alone doesn't skip shape mismatches, only
    # missing/extra keys, and used to raise RuntimeError here.
    model = ArkSwinTransformer([9], projector_features=None, use_mlp=False,
                                img_size=112, patch_size=4, window_size=7, embed_dim=96,
                                depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    load_imagenet_backbone(model, "swin_tiny_patch4_window7_224")  # must not raise


def test_load_finetune_backbone_transfers_encoder_not_heads():
    pretrained = ArkSwinTransformer([9, 2, 5], projector_features=None, use_mlp=False,
                                     patch_size=4, window_size=7, embed_dim=96,
                                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    ckpt_path = "test_fake_checkpoint_heads.pth.tar"
    torch.save({"teacher": pretrained.state_dict()}, ckpt_path)

    try:
        target = ArkSwinTransformer([2], projector_features=None, use_mlp=False,
                                     patch_size=4, window_size=7, embed_dim=96,
                                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
        head_before = target.omni_heads[0].weight.clone()
        load_finetune_backbone(target, ckpt_path, key="teacher")

        assert torch.allclose(target.omni_heads[0].weight, head_before), \
            "omni_heads must stay untouched -- loading them risks splicing in the wrong dataset's head"
        assert torch.allclose(target.patch_embed.proj.weight, pretrained.patch_embed.proj.weight), \
            "encoder weights must transfer from the checkpoint"
    finally:
        os.remove(ckpt_path)


def test_load_finetune_backbone_survives_class_count_matching_no_head():
    pretrained = ArkSwinTransformer([9, 2, 5], projector_features=None, use_mlp=False,
                                     patch_size=4, window_size=7, embed_dim=96,
                                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    ckpt_path = "test_fake_checkpoint_noheadmatch.pth.tar"
    torch.save({"teacher": pretrained.state_dict()}, ckpt_path)
    try:
        target = ArkSwinTransformer([7], projector_features=None, use_mlp=False,
                                     patch_size=4, window_size=7, embed_dim=96,
                                     depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
        load_finetune_backbone(target, ckpt_path, key="teacher")  # must not raise
    finally:
        os.remove(ckpt_path)


if __name__ == "__main__":
    test_forward_features_pooling_bridges_timm_spatial_output()
    test_generate_embeddings_also_pools()
    test_build_omni_model_swin_tiny_at_custom_img_size()
    test_load_imagenet_backbone_transfers_encoder_not_heads()
    test_load_imagenet_backbone_at_112px_survives_window_shrink()
    test_load_finetune_backbone_transfers_encoder_not_heads()
    test_load_finetune_backbone_survives_class_count_matching_no_head()
    print("test_models_medmnist.py: all checks passed")
