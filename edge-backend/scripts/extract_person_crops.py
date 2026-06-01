#!/usr/bin/env python3
"""
Extrae crops de personas desde RTSP o archivo de video usando YOLO.
"""
import argparse
import hashlib
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="RTSP URL o ruta de video")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--config", default=0.5, type=float)
    parser.add_argument("--sample-every", default=12, type=int, help="Procesar 1 de cada N frames")
    parser.add_argument("--min-width", default=50, type=int)
    parser.add_argument("--min-height", default=90, type=int)
    parser.add_argument("--max-images", default=3000, type=int)
    parser.add_argument("--dedup-distance", default=6, type=int, help="Umbral hash visual (0-64)")
    parser.add_argument("--prefix", default="person")
    return parser.parse_args()


def dhash(image_bgr, hash=8):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if v else "0" for v in diff.flatten())
    return int(bits, 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {args.source}")

    frame_idx = 0
    saved = 0
    recent_hashes: list[int] = []

    try:
        while saved < args.max_images:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1
            if frame_idx % args.sample_every != 0:
                continue

            results = model(
                frame,
                classes=[0],  # person
                conf=args.conf,
                verbose=False,
            )
            if not results:
                continue

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            frame_h, frame_w = frame.shape[:2]
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].cpu().numpy().tolist()]
                x1 = max(0, min(frame_w - 1, x1))
                x2 = max(0, min(frame_w, x2))
                y1 = max(0, min(frame_h - 1, y1))
                y2 = max(0, min(frame_h, y2))

                w = x2 - x1
                h = y2 - y1
                if w < args.min_width or h < args.min_height:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Deduplicación visual simple.
                hv = dhash(crop)
                is_dup = any(hamming(hv, prev) <= args.dedup_distance for prev in recent_hashes[-150:])
                if is_dup:
                    continue

                recent_hashes.append(hv)
                conf = float(boxes.conf[i].cpu().numpy())
                stamp = f"{frame_idx:08d}_{i:02d}_{int(conf * 1000):03d}"
                digest = hashlib.sha1(crop.tobytes()).hexdigest()[:8]
                filename = f"{args.prefix}_{stamp}_{digest}.jpg"
                out_path = args.output_dir / filename
                cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                saved += 1

                if saved % 100 == 0:
                    print(f"Guardados: {saved} crops")
                if saved >= args.max_images:
                    break
    finally:
        cap.release()

    print(f"Finalizado. Total crops guardados: {saved}")
    print(f"Salida: {args.output_dir}")


if __name__ == "__main__":
    main()
