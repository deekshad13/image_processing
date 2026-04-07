"""
Multi-engine model registry.
Each engine wraps a different vision model and provides a uniform interface:
  load() -> model
  embed(model, image_tensor) -> numpy embedding
  get_transform() -> torchvision transform
"""
import torch
import numpy as np
from torchvision import transforms

# ─── Engine definitions ──────────────────────────────────────────────────────

ENGINES = {}


def register_engine(engine_id, info):
    ENGINES[engine_id] = info


def _imagenet_transform(resize=256, crop=224):
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─── 1. DINOv2 ViT-S/14 (current baseline) ─────────────────────────────────

def _load_dinov2_vits14():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model

register_engine('dinov2_vits14', {
    'name': 'DINOv2 ViT-S/14',
    'family': 'DINOv2',
    'params': '21M',
    'embed_dim': 384,
    'description': 'Meta self-supervised ViT (Small). Current default. Strong on texture/shape similarity.',
    'pros': 'Best general visual similarity; lightweight; no labels needed',
    'cons': 'Not domain-specialized; needs fine-tuning for crop diseases',
    'load': _load_dinov2_vits14,
    'transform': _imagenet_transform,
    'embed': lambda model, tensor: model(tensor).squeeze().detach().numpy(),
})


# ─── 2. DINOv2 ViT-B/14 (bigger, more accurate) ────────────────────────────

def _load_dinov2_vitb14():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()
    return model

register_engine('dinov2_vitb14', {
    'name': 'DINOv2 ViT-B/14',
    'family': 'DINOv2',
    'params': '86M',
    'embed_dim': 768,
    'description': 'Meta self-supervised ViT (Base). 4x larger, richer embeddings for fine-grained tasks.',
    'pros': 'Best fine-grained similarity; more capacity for subtle disease features',
    'cons': 'Heavier (4x params); still needs fine-tuning for best results',
    'load': _load_dinov2_vitb14,
    'transform': _imagenet_transform,
    'embed': lambda model, tensor: model(tensor).squeeze().detach().numpy(),
})


# ─── 3. CLIP ViT-B/32 (OpenAI, vision-language) ────────────────────────────

def _load_clip_vitb32():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    return (model, preprocess)

def _embed_clip(model_tuple, tensor):
    model = model_tuple[0]
    return model.encode_image(tensor).squeeze().detach().numpy().astype(np.float32)

def _clip_transform_b32():
    import open_clip
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    return preprocess

register_engine('clip_vitb32', {
    'name': 'CLIP ViT-B/32',
    'family': 'CLIP',
    'params': '151M',
    'embed_dim': 512,
    'description': 'OpenAI vision-language model trained on 2B image-text pairs. Understands semantic concepts.',
    'pros': 'Understands "what" things are semantically; zero-shot capable',
    'cons': 'Weaker on fine-grained visual similarity than DINOv2; optimized for text-image matching',
    'load': _load_clip_vitb32,
    'transform': _clip_transform_b32,
    'embed': _embed_clip,
})


# ─── 4. CLIP ViT-L/14 (larger CLIP) ────────────────────────────────────────

def _load_clip_vitl14():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    model.eval()
    return (model, preprocess)

def _clip_transform_l14():
    import open_clip
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    return preprocess

register_engine('clip_vitl14', {
    'name': 'CLIP ViT-L/14',
    'family': 'CLIP',
    'params': '428M',
    'embed_dim': 768,
    'description': 'Large CLIP model. Richer semantic understanding, heavier compute.',
    'pros': 'Best semantic understanding; large model capacity',
    'cons': 'Slowest inference; large model; still weaker than DINOv2 on visual similarity',
    'load': _load_clip_vitl14,
    'transform': _clip_transform_l14,
    'embed': _embed_clip,
})


# ─── 5. ConvNeXt Base (modern CNN) ──────────────────────────────────────────

def _load_convnext():
    import timm
    model = timm.create_model('convnext_base', pretrained=True, num_classes=0)
    model.eval()
    return model

register_engine('convnext_base', {
    'name': 'ConvNeXt Base',
    'family': 'CNN',
    'params': '89M',
    'embed_dim': 1024,
    'description': 'Modern CNN from Meta that rivals Vision Transformers. Strong on local texture patterns.',
    'pros': 'Excellent local feature extraction; good for texture-based diseases; fast inference',
    'cons': 'Weaker on global shape/structure than ViTs; no self-supervised pretraining benefit',
    'load': _load_convnext,
    'transform': _imagenet_transform,
    'embed': lambda model, tensor: model(tensor).squeeze().detach().numpy(),
})


# ─── 6. ResNet50 (classic baseline) ─────────────────────────────────────────

def _load_resnet50():
    import timm
    model = timm.create_model('resnet50', pretrained=True, num_classes=0)
    model.eval()
    return model

register_engine('resnet50', {
    'name': 'ResNet-50',
    'family': 'CNN',
    'params': '25M',
    'embed_dim': 2048,
    'description': 'Classic CNN baseline. Industry workhorse for 10 years. Good reference point.',
    'pros': 'Fastest inference; smallest model; well-understood; easy to fine-tune',
    'cons': 'Weakest features; cannot capture global context; outdated architecture',
    'load': _load_resnet50,
    'transform': _imagenet_transform,
    'embed': lambda model, tensor: model(tensor).squeeze().detach().numpy(),
})


# ─── Engine manager ─────────────────────────────────────────────────────────

_loaded_models = {}

def load_engine(engine_id):
    """Load a model engine. Caches after first load."""
    if engine_id not in ENGINES:
        return None
    if engine_id not in _loaded_models:
        info = ENGINES[engine_id]
        print(f"Loading engine: {info['name']}...")
        _loaded_models[engine_id] = {
            'model': info['load'](),
            'transform': info['transform'](),
        }
        print(f"  Engine {info['name']} ready (dim={info['embed_dim']})")
    return _loaded_models[engine_id]


def get_embedding_with_engine(engine_id, pil_image):
    """Get embedding for a PIL image using specified engine."""
    engine_data = load_engine(engine_id)
    if engine_data is None:
        return None
    info = ENGINES[engine_id]
    transform = engine_data['transform']
    model = engine_data['model']
    tensor = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        emb = info['embed'](model, tensor)
    return emb


def list_engines():
    """Return list of available engine info (without callables)."""
    result = []
    for eid, info in ENGINES.items():
        result.append({
            'engine_id': eid,
            'name': info['name'],
            'family': info['family'],
            'params': info['params'],
            'embed_dim': info['embed_dim'],
            'description': info['description'],
            'pros': info['pros'],
            'cons': info['cons'],
            'loaded': eid in _loaded_models,
        })
    return result
