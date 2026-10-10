"""Semantic human parsing (ATR 18-class) for the local edit masks.

The P2.6 mask builder had no segmentation model, so its "hair" region was an
ellipse around the head and its "clothing" region a rectangle under the chin
(see :mod:`app.domains.portrait.edit_mask`). Those stand-ins could not promise
a hairline, and a circle around the head is not a hairstyle silhouette.

This module owns the one place that knows how to turn a photograph into the
semantic *label map* the mask builder consumes:

    GoGoDuck912/Self-Correction-Human-Parsing  -- MIT, the upstream project
    pirocheto/schp-atr-18                      -- MIT, ONNX export used here

The export is the ATR (``exp-schp-201908301523-atr.pth``) ResNet-101 model at
18 classes, ~82.3 mIoU on ATR. It is a *pretrained* network: nothing in this
repository trains it, fine-tunes it or downloads it as a side effect of a
request. The weight files are not committed to git; provenance and checksums
are pinned below and an operator fetches them once (see ``tools.run_human_parsing``).

    file                        sha256                                                             bytes
    schp-atr-18.onnx            c9f7c261d2049fb0ea694749b71df455ac8f3d34117c9f8b5157cdf685a437bd   1_477_886
    schp-atr-18.onnx.data       53a86d9ed61f276f0a3ddfbf2da03434f2e32b7e8f6a7aa8782b49c37842e467 266_665_984
    int8-static/schp-atr-18.onnx fbc1ed7ca1bd3f8e30a95a6b6e2ef1d0dfb36a4a49b9ee2d6d4cd3e7ff4d5c1f  69_387_404

``onnxruntime`` is imported lazily and only inside :meth:`SchpOnnxSegmenter.segment`,
so a deployment that has not installed it still imports the API and gets an
explicit :class:`HumanParsingUnavailable` instead of a crash at import time.
The mask builder never falls back to the geometric ellipse when this raises:
a missing parser is reported as a gap, not hidden behind a circle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ------------------------------------------------------------------ taxonomy

#: ATR class ids as the SCHP ATR model emits them (``logits[0]`` channel order).
ATR_CLASSES: dict[int, str] = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

CLASS_IDS: dict[str, int] = {name: index for index, name in ATR_CLASSES.items()}

BACKGROUND = CLASS_IDS["Background"]
HAT = CLASS_IDS["Hat"]
HAIR = CLASS_IDS["Hair"]
SUNGLASSES = CLASS_IDS["Sunglasses"]
UPPER_CLOTHES = CLASS_IDS["Upper-clothes"]
SKIRT = CLASS_IDS["Skirt"]
PANTS = CLASS_IDS["Pants"]
DRESS = CLASS_IDS["Dress"]
BELT = CLASS_IDS["Belt"]
FACE = CLASS_IDS["Face"]
SCARF = CLASS_IDS["Scarf"]

#: ATR has no separate "neck" class -- the neck comes back as ``Face``.
FACE_CLASSES: tuple[int, ...] = (FACE,)

#: Everything that is a garment in this taxonomy. ``Hat``/``Sunglasses``/``Scarf``
#: are deliberately absent: they sit on the body and must not be repainted as
#: clothing.
CLOTHING_CLASSES: tuple[int, ...] = (UPPER_CLOTHES, SKIRT, PANTS, DRESS)

CLASS_COUNT = len(ATR_CLASSES)

# ------------------------------------------------------------------- effects

HUMAN_PARSING_MODEL_FILENAME = "schp-atr-18.onnx"

#: MIT-licensed ONNX export of the SCHP ATR model (Hugging Face hub).
HUMAN_PARSING_HF_REPO = "pirocheto/schp-atr-18"
HUMAN_PARSING_HF_FILE = "onnx/schp-atr-18.onnx"
HUMAN_PARSING_LICENSE = "MIT"
HUMAN_PARSING_UPSTREAM = "https://github.com/GoGoDuck912/Self-Correction-Human-Parsing"
HUMAN_PARSING_SHA256 = "c9f7c261d2049fb0ea694749b71df455ac8f3d34117c9f8b5157cdf685a437bd"
HUMAN_PARSING_DATA_SHA256 = "53a86d9ed61f276f0a3ddfbf2da03434f2e32b7e8f6a7aa8782b49c37842e467"

#: The network is fully convolutional but the export is pinned to 512x512.
INPUT_SIZE: tuple[int, int] = (512, 512)

#: ImageNet statistics, in RGB order -- the same order ``SCHPImageProcessor``
#: feeds the network after ``PIL.Image.convert("RGB")``.
PIXEL_MEAN = (0.406, 0.456, 0.485)
PIXEL_STD = (0.225, 0.224, 0.229)

INPUT_NAME = "pixel_values"
OUTPUT_NAME = "logits"


class HumanParsingUnavailable(RuntimeError):
    """Raised when the semantic parser cannot run in this environment."""


def default_model_path(models_dir: Path | str | None = None) -> Path:
    """Where the ATR ONNX export is expected to live."""
    base = Path(models_dir) if models_dir is not None else Path("data/models")
    return base / HUMAN_PARSING_MODEL_FILENAME


# ------------------------------------------------------------ pre/post-process


def preprocess(image_rgb: Any) -> np.ndarray:
    """Photo -> the ``(1, 3, 512, 512)`` float32 tensor the export expects.

    A faithful, dependency-light copy of the upstream ``SCHPImageProcessor``:
    bilinear resize to 512x512, scale to 0-1, ImageNet normalisation, HWC->CHW.
    """
    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("human parsing expects an HxWx3 RGB image")

    resized = _resize_bilinear(array, INPUT_SIZE).astype(np.float32) / 255.0
    normalised = (resized - np.asarray(PIXEL_MEAN, dtype=np.float32)) / np.asarray(
        PIXEL_STD, dtype=np.float32
    )
    return np.ascontiguousarray(np.transpose(normalised, (2, 0, 1))[None])


def label_map_from_logits(logits: Any, target_size: tuple[int, int]) -> np.ndarray:
    """``(1, C, h, w)`` logits -> ``uint8`` label map at ``target_size`` (h, w).

    The argmax runs at the network's own resolution and only then is the class
    map expanded with nearest-neighbour, so a boundary pixel is a whole class
    and never a blended colour between two labels.
    """
    array = np.asarray(logits)
    if array.ndim != 4:
        raise ValueError("expected (N, C, H, W) logits")
    if array.shape[1] != CLASS_COUNT:
        raise ValueError(
            f"expected {CLASS_COUNT} ATR classes, got {array.shape[1]} channels"
        )

    labels = np.argmax(array[0], axis=0).astype(np.uint8)
    height, width = int(target_size[0]), int(target_size[1])
    return _resize_nearest(labels, (height, width))


def _resize_bilinear(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize in 8-bit, exactly what ``PIL``/``SCHPImageProcessor`` does."""
    from PIL import Image

    height, width = int(size[0]), int(size[1])
    pil = Image.fromarray(np.ascontiguousarray(image))
    if (pil.height, pil.width) != (height, width):
        pil = pil.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(pil)


def _resize_nearest(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    height, width = int(size[0]), int(size[1])
    pil = Image.fromarray(labels)
    if (pil.height, pil.width) != (height, width):
        pil = pil.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(pil, dtype=np.uint8)


# ------------------------------------------------------------------- runtime


class SchpOnnxSegmenter:
    """CPU human parsing with the pinned SCHP ATR ONNX export.

    Measured on this machine (Apple silicon, CPUExecutionProvider, one thread
    pool): ~2.4 s per 904x1190 photo, ~590 MiB peak RSS including the 257 MB
    weight file. The session is created once and reused.
    """

    def __init__(self, model_path: Path | str, *, providers: tuple[str, ...] | None = None) -> None:
        self._model_path = Path(model_path)
        self._providers = providers
        self._session: Any | None = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    def _build_session(self) -> Any:
        if not self._model_path.is_file():
            raise HumanParsingUnavailable(
                f"Human-parsing model not found at {self._model_path}. Fetch "
                f"{HUMAN_PARSING_HF_REPO}:{HUMAN_PARSING_HF_FILE} (MIT) once; "
                "no download is attempted automatically."
            )
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - depends on the environment
            raise HumanParsingUnavailable(
                "onnxruntime is not installed in this environment, so the "
                "semantic ATR parser cannot run. Install it in an isolated "
                "environment (see experiments/p27-human-parsing) and run "
                "tools.run_human_parsing there; the mask builder will not "
                f"substitute the geometric ellipse. ({exc})"
            ) from exc

        providers = self._providers or ("CPUExecutionProvider",)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            return ort.InferenceSession(
                str(self._model_path), sess_options=options, providers=list(providers)
            )
        except Exception as exc:  # pragma: no cover - depends on the environment
            raise HumanParsingUnavailable(
                f"onnxruntime could not load {self._model_path}: {exc}"
            ) from exc

    def segment(self, image_rgb: Any) -> np.ndarray:
        """Return the ``uint8`` ATR label map for *image_rgb* (HxW, original size)."""
        array = np.asarray(image_rgb)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("human parsing expects an HxWx3 RGB image")

        session = self._session or self._build_session()
        self._session = session
        tensor = preprocess(array)
        outputs = session.run([OUTPUT_NAME], {INPUT_NAME: tensor})
        return label_map_from_logits(outputs[0], (array.shape[0], array.shape[1]))

    def close(self) -> None:  # pragma: no cover - best effort cleanup
        self._session = None
