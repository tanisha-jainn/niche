"""Export the CLIP encoders to small ONNX models so the server runs without torch.

- text encoder:  8-bit weight-only MatMuls (lossless in practice) + fp16 token table   (~100 MB)
- image encoder: 4-bit weight-only MatMuls - only reads uploaded screenshots for taste  (~60 MB)
Both saved in external-data format, which roughly halves ONNX Runtime's memory at load.

Writes data/clip_text.onnx(+.data) and data/clip_image.onnx(+.data), then reports parity with torch.
Run once locally (needs torch): python -m app.export_onnx
"""
import json

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from .build import cache_path
from .config import DATA
from .embed import Embedder

TEXT, IMAGE = DATA / "clip_text.onnx", DATA / "clip_image.onnx"
TEXT_BITS, IMAGE_BITS = 8, 4


class _Unit(torch.nn.Module):
    """Registers the CLIP model as a submodule (so its weights export as initializers) and
    L2-normalises the chosen encoder's output."""
    def __init__(self, model, which: str):
        super().__init__()
        self.m = model
        self.which = which

    def forward(self, x):
        f = getattr(self.m, self.which)(x)
        return f / f.norm(dim=-1, keepdim=True)


def _weight_only(path, bits: int) -> onnx.ModelProto:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    q = MatMulNBitsQuantizer(onnx.load(str(path)), bits=bits, block_size=32, is_symmetric=True)
    q.process()
    return q.model.model


def _fp16_token_table(m: onnx.ModelProto) -> None:
    """Gather is type-agnostic: store the 49408x512 token table in fp16 and cast the rows back."""
    name = next(n.input[0] for n in m.graph.node if n.op_type == "Gather" and "token_embedding" in n.input[0])
    init = next(t for t in m.graph.initializer if t.name == name)
    w16 = numpy_helper.to_array(init).astype(np.float16)
    m.graph.initializer.remove(init)
    m.graph.initializer.append(numpy_helper.from_array(w16, name))
    for i, n in enumerate(list(m.graph.node)):
        if n.op_type == "Gather" and name in n.input:
            out = n.output[0]
            n.output[0] = out + "_fp16"
            m.graph.node.insert(i + 1, helper.make_node("Cast", [out + "_fp16"], [out], to=TensorProto.FLOAT))
            break


def _save(m: onnx.ModelProto, path) -> None:
    for p in (path, path.with_name(path.name + ".data")):
        p.unlink(missing_ok=True)
    onnx.save(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=path.name + ".data", size_threshold=1024)


def main() -> None:
    from .clip_tokenizer import clip_tokens
    from .onnx_embed import OnnxEmbedder

    emb = Embedder()
    emb.model.to("cpu").eval()
    emb.device = "cpu"
    torch.backends.mha.set_fastpath_enabled(False)     # the fused attention kernel has no ONNX export
    tmp_t, tmp_i = DATA / "clip_text.fp32.onnx", DATA / "clip_image.fp32.onnx"
    for which, dummy, name, path in (("encode_text", torch.from_numpy(clip_tokens(["a photo of clothing", "x"])), "tokens", tmp_t),
                                     ("encode_image", torch.randn(2, 3, 224, 224), "pixels", tmp_i)):
        torch.onnx.export(_Unit(emb.model, which), (dummy,), str(path), input_names=[name], output_names=["embedding"],
                          dynamic_axes={name: {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, dynamo=False)

    text = _weight_only(tmp_t, TEXT_BITS)
    _fp16_token_table(text)
    _save(text, TEXT)
    _save(_weight_only(tmp_i, IMAGE_BITS), IMAGE)
    tmp_t.unlink()
    tmp_i.unlink()
    for p in (TEXT, IMAGE):
        print(f"{p.name}: {(p.stat().st_size + p.with_name(p.name + '.data').stat().st_size) / 1e6:.0f} MB")

    # parity with torch on real queries and catalog photos
    o = OnnxEmbedder()
    qs = ["low-rise black pants", "fall jacket", "lace trim denim", "satin slip dress", "chunky knit cardigan",
          "oversized grey wool blazer", "floral midi dress", "leather pants", "white linen shirt", "pleated tennis skirt"]
    t_ref, t_new = emb.texts(qs), o.texts(qs)
    items = [json.loads(l) for l in open(DATA / "index.jsonl")][::900]
    imgs = [Image.open(cache_path(it["image"])).convert("RGB") for it in items if cache_path(it["image"]).exists()][:40]
    i_ref, i_new = emb.images(imgs), o.images(imgs)
    E = np.load(DATA / "embeddings.npy").astype(np.float32)
    overlap = np.mean([len(set(np.argsort(-(E @ a))[:50]) & set(np.argsort(-(E @ b))[:50])) / 50 for a, b in zip(t_ref, t_new)])
    ci = (i_ref * i_new).sum(1)
    print(f"text cosine vs torch {((t_ref * t_new).sum(1)).mean():.4f} | search top-50 overlap {overlap:.2f}")
    print(f"image cosine vs torch mean {ci.mean():.4f} min {ci.min():.4f}")


if __name__ == "__main__":
    main()
