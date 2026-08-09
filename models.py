import torch
import torch.nn as nn
from functools import partial
from torch.hub import load_state_dict_from_url

import timm.models.vision_transformer as vit
import timm.models.swin_transformer as swin
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

def save_checkpoint(state,filename='model'):

    torch.save(state, filename + '.pth.tar')

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
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded ImageNet backbone ({}) with msg: {}'.format(timm_model_name, msg))
    return model
