"""Torch-free CLIP encoder: ONNX Runtime + a vendored tokenizer + numpy preprocessing.
Same interface as embed.Embedder (texts(), images(), dim, name)."""
import numpy as np
import onnxruntime as ort
from PIL import Image

from .clip_tokenizer import clip_tokens
from .config import DATA

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SIZE = 224


def preprocess(im: Image.Image) -> np.ndarray:
    """Resize shortest side to 224 (bicubic), centre-crop 224, normalise - as open_clip does."""
    im = im.convert("RGB")
    w, h = im.size
    s = SIZE / min(w, h)
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    im = im.crop((left, top, left + SIZE, top + SIZE))
    x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


class OnnxEmbedder:
    name = "Marqo/marqo-fashionCLIP (onnx, 8-bit text / 4-bit image)"
    device = "cpu"

    def __init__(self):
        self.opts = ort.SessionOptions()
        self.opts.intra_op_num_threads = 2
        self.opts.enable_cpu_mem_arena = False      # return activation memory after each call
        self.text = ort.InferenceSession(str(DATA / "clip_text.onnx"), self.opts, providers=["CPUExecutionProvider"])
        self._image = None                          # loaded on first upload - most visitors never need it
        self.dim = int(self.texts(["a photo of clothing"]).shape[-1])

    @property
    def image(self):
        if self._image is None:
            self._image = ort.InferenceSession(str(DATA / "clip_image.onnx"), self.opts, providers=["CPUExecutionProvider"])
        return self._image

    def texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = [self.text.run(None, {"tokens": clip_tokens(texts[i:i + batch_size])})[0]
               for i in range(0, len(texts), batch_size)]
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), dtype=np.float32)

    def images(self, pil_images: list[Image.Image], batch_size: int = 8) -> np.ndarray:
        out = []
        for i in range(0, len(pil_images), batch_size):
            batch = np.stack([preprocess(im) for im in pil_images[i:i + batch_size]])
            out.append(self.image.run(None, {"pixels": batch})[0])
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), dtype=np.float32)
