"""CLIP wrapper: puts product photos, user screenshots and text descriptions in one vector space.

Two backends with the same interface (texts(), images(), dim, name, device):
- OnnxEmbedder (app/onnx_embed.py): torch-free, 8-bit weights, used by the server whenever
  data/clip_text.onnx + data/clip_image.onnx exist. ~275 MB instead of ~2 GB of torch.
- Embedder (below): the original torch/open_clip model, used to build the index and export ONNX.
"""
import numpy as np
from PIL import Image

from .config import CLIP_FALLBACK, CLIP_MODEL, DATA


def get_embedder():
    """Prefer the torch-free ONNX encoder; fall back to torch (local dev before app.export_onnx)."""
    if (DATA / "clip_text.onnx").exists() and (DATA / "clip_image.onnx").exists():
        from .onnx_embed import OnnxEmbedder
        return OnnxEmbedder()
    return Embedder()


class Embedder:
    def __init__(self, model_name: str = CLIP_MODEL):
        import open_clip
        import torch
        self.torch = torch
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

    def images(self, pil_images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = self.torch.stack([self.preprocess(im.convert("RGB")) for im in pil_images[i:i + batch_size]])
                out.append(self._unit(self.model.encode_image(batch.to(self.device))))
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            return self._unit(self.model.encode_text(tokens))

    @staticmethod
    def _unit(feats) -> np.ndarray:
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()
