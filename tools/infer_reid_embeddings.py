# encoding: utf-8
"""
Utility script to run FastReID inference on either:
1. A flat collection of images, or
2. A directory of tracklet folders that each contain image frames.

The output is a single .pt file with frame-level embeddings and, when relevant,
tracklet-level embeddings built by mean-pooling frame features followed by
L2-normalization.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

sys.path.append(".")

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run FastReID embedding inference")
    parser.add_argument(
        "input",
        help="Path to an image, a directory of images, or a directory of tracklet folders",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="FastReID config file",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Checkpoint .pth file",
    )
    parser.add_argument(
        "--output",
        default="outputs/reid_embeddings.pt",
        help="Output .pt path",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of frames per forward pass",
    )
    parser.add_argument(
        "--input-mode",
        default="auto",
        choices=["auto", "images", "tracklets"],
        help="How to interpret the input directory",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images in image mode",
    )
    parser.add_argument(
        "--save-frame-embeddings",
        action="store_true",
        help="Keep per-frame embeddings in the output file",
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Additional config overrides",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(directory: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    images = [p for p in directory.glob(pattern) if is_image_file(p)]
    return sorted(images)


def infer_input_mode(input_path: Path) -> str:
    if input_path.is_file():
        return "images"

    child_dirs = sorted([p for p in input_path.iterdir() if p.is_dir()])
    if child_dirs:
        for child in child_dirs:
            if list_images(child, recursive=True):
                return "tracklets"

    return "images"


def collect_samples(input_path: Path, input_mode: str, recursive: bool) -> List[Dict[str, object]]:
    if input_mode == "images":
        if input_path.is_file():
            images = [input_path]
        else:
            images = list_images(input_path, recursive=recursive)
        if not images:
            raise ValueError(f"No images found under {input_path}")
        return [
            {
                "frame_path": str(image.resolve()),
                "tracklet_id": image.stem,
                "tracklet_path": str(image.resolve()),
            }
            for image in images
        ]

    if not input_path.is_dir():
        raise ValueError("Tracklet mode expects a directory input")

    samples: List[Dict[str, object]] = []
    tracklet_dirs = sorted([p for p in input_path.iterdir() if p.is_dir()])
    if not tracklet_dirs:
        raise ValueError(f"No tracklet directories found under {input_path}")

    for tracklet_dir in tracklet_dirs:
        frames = list_images(tracklet_dir, recursive=True)
        if not frames:
            continue
        for frame in frames:
            samples.append(
                {
                    "frame_path": str(frame.resolve()),
                    "tracklet_id": tracklet_dir.name,
                    "tracklet_path": str(tracklet_dir.resolve()),
                }
            )

    if not samples:
        raise ValueError(f"No images found inside tracklet directories under {input_path}")

    return samples


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(
        [
            "MODEL.WEIGHTS",
            args.weights,
            "MODEL.DEVICE",
            args.device,
        ]
    )
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def build_predictor(cfg) -> DefaultPredictor:
    if cfg.MODEL.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return DefaultPredictor(cfg)


def preprocess_image(image_path: str, size_hw: Sequence[int]) -> torch.Tensor:
    if cv2 is not None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = image[:, :, ::-1]
        image = cv2.resize(image, tuple(size_hw[::-1]), interpolation=cv2.INTER_CUBIC)
    else:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((size_hw[1], size_hw[0]), resample=Image.BICUBIC)
        image = np.asarray(image)

    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    return image


def batched(sequence: Sequence[Dict[str, object]], batch_size: int):
    for idx in range(0, len(sequence), batch_size):
        yield sequence[idx: idx + batch_size]


def extract_embeddings(
    predictor: DefaultPredictor,
    cfg,
    samples: Sequence[Dict[str, object]],
    batch_size: int,
) -> Tuple[torch.Tensor, List[str], List[str], List[str]]:
    all_embeddings: List[torch.Tensor] = []
    frame_paths: List[str] = []
    tracklet_ids: List[str] = []
    tracklet_paths: List[str] = []

    total_batches = (len(samples) + batch_size - 1) // batch_size
    for batch in tqdm.tqdm(batched(samples, batch_size), total=total_batches, desc="Extracting embeddings"):
        tensors = [preprocess_image(sample["frame_path"], cfg.INPUT.SIZE_TEST) for sample in batch]
        batch_tensor = torch.stack(tensors, dim=0)
        embeddings = predictor(batch_tensor)
        embeddings = F.normalize(embeddings, dim=1)

        all_embeddings.append(embeddings)
        frame_paths.extend([sample["frame_path"] for sample in batch])
        tracklet_ids.extend([sample["tracklet_id"] for sample in batch])
        tracklet_paths.extend([sample["tracklet_path"] for sample in batch])

    return torch.cat(all_embeddings, dim=0), frame_paths, tracklet_ids, tracklet_paths


def aggregate_tracklets(
    frame_embeddings: torch.Tensor,
    tracklet_ids: Sequence[str],
    tracklet_paths: Sequence[str],
) -> Dict[str, object]:
    grouped: Dict[str, Dict[str, object]] = {}

    for idx, (tracklet_id, tracklet_path) in enumerate(zip(tracklet_ids, tracklet_paths)):
        entry = grouped.setdefault(
            tracklet_id,
            {"indices": [], "tracklet_path": tracklet_path},
        )
        entry["indices"].append(idx)

    ordered_ids = sorted(grouped.keys())
    aggregated_embeddings: List[torch.Tensor] = []
    ordered_paths: List[str] = []
    frame_counts: List[int] = []

    for tracklet_id in ordered_ids:
        indices = grouped[tracklet_id]["indices"]
        pooled = frame_embeddings[indices].mean(dim=0, keepdim=True)
        pooled = F.normalize(pooled, dim=1).squeeze(0)
        aggregated_embeddings.append(pooled)
        ordered_paths.append(grouped[tracklet_id]["tracklet_path"])
        frame_counts.append(len(indices))

    tracklet_embeddings = torch.stack(aggregated_embeddings, dim=0)
    similarity = torch.mm(tracklet_embeddings, tracklet_embeddings.t())

    return {
        "tracklet_ids": ordered_ids,
        "tracklet_paths": ordered_paths,
        "tracklet_frame_counts": frame_counts,
        "tracklet_embeddings": tracklet_embeddings,
        "tracklet_similarity": similarity,
    }


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    input_mode = args.input_mode
    if input_mode == "auto":
        input_mode = infer_input_mode(input_path)

    cfg = setup_cfg(args)
    predictor = build_predictor(cfg)
    samples = collect_samples(input_path, input_mode=input_mode, recursive=args.recursive)
    frame_embeddings, frame_paths, tracklet_ids, tracklet_paths = extract_embeddings(
        predictor, cfg, samples, args.batch_size
    )

    result = {
        "config_file": str(Path(args.config_file).expanduser().resolve()),
        "weights": str(Path(args.weights).expanduser().resolve()),
        "device": cfg.MODEL.DEVICE,
        "input": str(input_path),
        "input_mode": input_mode,
        "feature_dim": int(frame_embeddings.shape[1]),
        "normalized": True,
    }

    if args.save_frame_embeddings or input_mode == "images":
        result["frame_paths"] = frame_paths
        result["frame_tracklet_ids"] = tracklet_ids
        result["frame_tracklet_paths"] = tracklet_paths
        result["frame_embeddings"] = frame_embeddings

    result.update(aggregate_tracklets(frame_embeddings, tracklet_ids, tracklet_paths))

    torch.save(result, output_path)

    print("Saved inference output to:", output_path)
    print("Input mode:", input_mode)
    print("Frames processed:", len(frame_paths))
    print("Tracklets aggregated:", len(result["tracklet_ids"]))
    print("Feature dimension:", result["feature_dim"])


if __name__ == "__main__":
    main()
