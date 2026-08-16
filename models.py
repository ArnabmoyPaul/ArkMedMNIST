import torch
import torch.nn as nn
from functools import partial
from torch.hub import load_state_dict_from_url

import timm.models.vision_transformer as vit
import timm.models.swin_transformer as swin
from torchvision.models.video import r3d_18, R3D_18_Weights
from convnext import ConvNeXt

from timm.models.helpers import load_state_dict

from utils import remap_pretrained_keys_swin


class ArkSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None 
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        x = self._pool(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj = True):
        x = self.forward_features(x)
        x = self._pool(x)
        if after_proj:
            x = self.projector(x)
        return x

    def _pool(self, x):
        """timm>=0.9 changed SwinTransformer.forward_features() to return an
        unpooled spatial map (B, H, W, C) instead of the pooled (B, C) vector
        this class (written against the repo's pinned timm==0.5.4) expects --
        confirmed directly against the installed timm==1.0.28 during planning
        (forward_features returns (B,7,7,1024) for a 224x224 input). Bridge it
        with the encoder's own head pooling (respects whatever `global_pool`
        the encoder was built with) instead of reimplementing pooling."""
        if x.dim() > 2:
            x = self.forward_head(x, pre_logits=True)
        return x

class ArkConvNeXt(ConvNeXt):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, encoder_features=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None 
        if projector_features:
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]
    
    def generate_embeddings(self, x, after_proj = True):
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x

class ArkR3D(nn.Module):
    """3D counterpart to ArkSwinTransformer, for MedMNIST 3D volumes: wraps
    torchvision's r3d_18 as the encoder (fc replaced by Identity so it returns
    pooled (B,512) features) behind the same forward(x, head_n) -> (feat, pred)
    / .projector / .omni_heads contract, so trainer.py's shared train/eval code
    doesn't need to know which encoder it's driving. Random init here --
    pretrained (Kinetics) weights are injected separately via
    load_kinetics_backbone, mirroring load_imagenet_backbone's two-step
    build-then-load pattern for the 2D Swin encoder. Stem is 1-channel
    (grayscale volumes) instead of r3d_18's default 3 (RGB video)."""
    def __init__(self, num_classes_list, projector_features=None, use_mlp=False):
        super().__init__()
        assert num_classes_list is not None

        self.encoder = r3d_18(weights=None)
        encoder_features = self.encoder.fc.in_features  # 512
        self.encoder.fc = nn.Identity()
        old_stem = self.encoder.stem[0]
        self.encoder.stem[0] = nn.Conv3d(1, old_stem.out_channels, kernel_size=old_stem.kernel_size,
                                          stride=old_stem.stride, padding=old_stem.padding, bias=False)

        self.num_features = encoder_features
        self.projector = None
        if projector_features:
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.encoder(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj=True):
        x = self.encoder(x)
        if after_proj and self.projector:
            x = self.projector(x)
        return x

def build_omni_model_from_checkpoint(args, num_classes_list, key):
    if args.model_name == "swin_base": #swin_base_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    elif args.model_name == "swin_large": #swin_large_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_384": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =384, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_768": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =768, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "conv_base":
        model = ArkConvNeXt(num_classes_list, args.projector_features, args.use_mlp, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])

    if args.pretrained_weights is not None:
        checkpoint = torch.load(args.pretrained_weights)
        state_dict = checkpoint[key]
        if any([True if 'module.' in k else False for k in state_dict.keys()]):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}

        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))     
           
    return model

def build_omni_model(args, num_classes_list):
    if args.model_name == "swin_base": #swin_base_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    elif args.model_name == "swin_large": #swin_large_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_384": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =384, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_768": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =768, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_large_1152": #swin_large_patch4_window12_384
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, img_size =1152, patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    elif args.model_name == "swin_tiny": #swin_tiny_patch4_window7_224, sized for 8GB VRAM (rule 4: shrink Swin, don't substitute it)
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp,
                                    img_size=args.crop_size, patch_size=4, window_size=7,
                                    embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
    elif args.model_name == "conv_base":
        model = ArkConvNeXt(num_classes_list, args.projector_features, args.use_mlp, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        # url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth'
    if args.pretrained_weights is not None:
        if args.pretrained_weights.startswith('https'):
            state_dict = load_state_dict_from_url(url=args.pretrained_weights, map_location='cpu')
        else:
            state_dict = load_state_dict(args.pretrained_weights)
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        k_del = []
        for k in state_dict.keys():
            if "attn_mask" in k:
                k_del.append(k)
        print(f"Removing key {k_del} from pretrained checkpoint")
        for k in k_del:
            del state_dict[k]
            
        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))

    return model

def build_omni_model_3d(args, num_classes_list):
    """3D counterpart to build_omni_model. Random-init ArkR3D; pretrained
    (Kinetics) weights come from load_kinetics_backbone, same two-step
    build-then-load split build_omni_model uses via load_imagenet_backbone.
    args.pretrained_weights_3d (mirrors args.pretrained_weights) is a saved
    checkpoint file, e.g. student/teacher resuming from a prior run -- most
    callers won't set it, hence the getattr default."""
    model = ArkR3D(num_classes_list, args.projector_features, args.use_mlp)
    pretrained_weights_3d = getattr(args, 'pretrained_weights_3d', None)
    if pretrained_weights_3d is not None:
        state_dict = load_state_dict(pretrained_weights_3d)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))
    return model

def save_checkpoint(state,filename='model'):

    torch.save(state, filename + '.pth.tar')

def load_kinetics_backbone(model):
    """Kinetics init for a from-scratch ArkR3D, mirroring load_imagenet_backbone's
    build-then-load pattern for the 2D Swin encoder. r3d_18's stem conv is
    3-channel (Kinetics is RGB video); ArkR3D's stem is 1-channel (grayscale
    volumes), so the stem weight is averaged across the channel dim before
    loading instead of dropped outright -- dropping it (like the Swin
    shape-mismatch filter does for its window-shrunk keys) would throw away
    the pretrained low-level filters on the very first layer, which the
    averaging trick preserves."""
    src = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    state_dict = {f'encoder.{k}': v for k, v in src.state_dict().items() if not k.startswith('fc.')}
    stem_key = 'encoder.stem.0.weight'
    state_dict[stem_key] = state_dict[stem_key].mean(dim=1, keepdim=True)
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded Kinetics backbone (r3d_18) with msg: {}'.format(msg))
    return model

def load_imagenet_backbone(model, timm_model_name):
    """ImageNet init for a from-scratch ArkSwinTransformer. `args.init` is
    parsed in main_ark.py's help text (Random|ImageNet_1k|...) but never
    actually read anywhere in this codebase (confirmed via grep: no other
    reference to args.init in main_ark.py/engine.py/models.py/trainer.py) --
    ImageNet init has to be wired in explicitly. Mirrors build_omni_model's
    existing pretrained-weight loading pattern (attn_mask key removal,
    strict=False, printed load message) but sources the state dict from
    timm's own hub instead of args.pretrained_weights."""
    import timm
    src = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
    state_dict = src.state_dict()
    k_del = [k for k in state_dict if "attn_mask" in k]
    for k in k_del:
        del state_dict[k]
    # strict=False only tolerates missing/extra keys, not shape mismatches -- at
    # crop_size < 224 the last stage's window auto-shrinks (timm SwinTransformer
    # clamps window_size to the stage's spatial resolution), so its
    # relative_position_bias_table shape no longer matches the 224px checkpoint.
    model_sd = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items()
                  if k not in model_sd or v.shape == model_sd[k].shape}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded ImageNet backbone ({}) with msg: {}'.format(timm_model_name, msg))
    return model

def load_finetune_backbone(model, checkpoint_path, key='teacher'):
    """Loads a single-task-head model's encoder from a multi-head omni-pretraining
    checkpoint. omni_heads keys are always dropped before loading: the
    checkpoint's omni_heads.0 is whichever dataset was first in the ORIGINAL
    pretraining dataset_list, not this model's target dataset -- loading it
    unfiltered would crash on a shape mismatch, or worse, silently splice in
    the wrong dataset's head whenever class counts happen to match
    (strict=False only tolerates missing/extra keys, not shape mismatches, so
    a genuine mismatch would still raise -- the head must be filtered out
    before load_state_dict ever sees it)."""
    # weights_only=False: PyTorch 2.6+ defaults torch.load to weights_only=True,
    # which rejects our own checkpoints (omni_engine's 'lossMIN' entry is a list
    # of numpy float64s from evaluate(), not on the default allowlist). Safe
    # here since these checkpoints are self-generated, not from an untrusted source.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint[key]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('omni_heads.')}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded {} backbone from {} with msg: {}'.format(key, checkpoint_path, msg))
    return msg
