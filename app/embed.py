"""CLIP wrapper: puts product photos, user screenshots and text descriptions in one vector space."""
import numpy as np
import torch
import open_clip
from PIL import Image

from .config import CLIP_MODEL, CLIP_FALLBACK


class Embedder:
    def __init__(self, model_name: str = CLIP_MODEL):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.name = model_name
        except Exception as e:  # hub unreachable, model card changed, etc.
            name, pretrained = CLIP_FALLBACK
            print(f"[embed] {model_name} failed ({e.__class__.__name__}); falling back to {name}/{pretrained}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
            self.tokenizer = open_clip.get_tokenizer(name)
            self.name = f"{name}/{pretrained}"
        self.model.eval().to(self.device)
        self.dim = int(self.texts(["a photo of clothing"]).shape[-1])  # also warms the model up

    @torch.no_grad()
    def images(self, pil_images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(pil_images), batch_size):
            batch = torch.stack([self.preprocess(im.convert("RGB")) for im in pil_images[i:i + batch_size]])
            feats = self.model.encode_image(batch.to(self.device))
            out.append(_unit(feats))
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    @torch.no_grad()
    def texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        return _unit(self.model.encode_text(tokens))


def _unit(feats: torch.Tensor) -> np.ndarray:
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()
