#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import cv2


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
LEFT_EYE = {
    "upper": 37,
    "lower": 41,
    "outer": 36,
    "inner": 39,
}
RIGHT_EYE = {
    "upper": 43,
    "lower": 47,
    "outer": 45,
    "inner": 42,
}
NOSE_TIP = 30


def iter_images(input_dir: Path):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def estimate_eye_state(landmarks, eye_idx, open_threshold):
    upper = landmarks[eye_idx["upper"]]
    lower = landmarks[eye_idx["lower"]]
    outer = landmarks[eye_idx["outer"]]
    inner = landmarks[eye_idx["inner"]]

    lid_dist = float(abs(upper["y_px"] - lower["y_px"]))
    eye_width = float(max(1, abs(outer["x_px"] - inner["x_px"])))
    ratio = float(lid_dist / eye_width)
    state = "open" if ratio >= open_threshold else "closed"

    return {
        "state": state,
        "lid_distance_px": lid_dist,
        "eye_width_px": eye_width,
        "ratio": ratio,
    }


def estimate_head_tilt(landmarks, tilt_threshold_deg):
    left_outer = landmarks[LEFT_EYE["outer"]]
    right_outer = landmarks[RIGHT_EYE["outer"]]
    dx = float(right_outer["x_px"] - left_outer["x_px"])
    dy = float(right_outer["y_px"] - left_outer["y_px"])
    angle_deg = float(math.degrees(math.atan2(dy, max(1.0, dx))))

    if abs(angle_deg) < tilt_threshold_deg:
        tilt = "level"
    elif angle_deg > 0:
        tilt = "right"
    else:
        tilt = "left"

    return {
        "roll_deg": angle_deg,
        "tilt": tilt,
    }


def load_face_detector(face_cascade_path):
    if face_cascade_path == "opencv_default":
        face_cascade_path = str(
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise SystemExit(f"Failed to load face cascade: {face_cascade_path}")
    return face_cascade


def load_facemark(model_path):
    if not hasattr(cv2, "face"):
        raise SystemExit(
            "OpenCV facemark not available. Install opencv-contrib-python."
        )
    if hasattr(cv2.face, "createFacemarkLBF"):
        facemark = cv2.face.createFacemarkLBF()
    else:
        facemark = cv2.face.FacemarkLBF_create()
    facemark.loadModel(str(model_path))
    return facemark


def main():
    parser = argparse.ArgumentParser(
        description="Run lightweight face landmark detection on images."
    )
    parser.add_argument(
        "--input-dir",
        default="images/input",
        help="Directory containing images (default: images/input).",
    )
    parser.add_argument(
        "--output-dir",
        default="images/output",
        help="Directory to write annotated images and JSON (default: images/output).",
    )
    parser.add_argument("--max-faces", type=int, default=1)
    parser.add_argument(
        "--face-cascade",
        default="opencv_default",
        help=(
            "Path to Haar cascade XML. Use 'opencv_default' for bundled default."
        ),
    )
    parser.add_argument(
        "--lbf-model",
        default="models/lbfmodel.yaml",
        help="Path to LBF facemark model (default: models/lbfmodel.yaml).",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--eye-open-threshold",
        type=float,
        default=0.18,
        help="Eye open threshold as ratio of eye width (default: 0.18).",
    )
    parser.add_argument(
        "--tilt-threshold-deg",
        type=float,
        default=5.0,
        help="Head tilt threshold in degrees (default: 5.0).",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draw landmarks on output images.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"Input dir does not exist: {input_dir}")

    face_detector = load_face_detector(args.face_cascade)
    lbf_model_path = Path(args.lbf_model)
    if not lbf_model_path.exists():
        raise SystemExit(
            "LBF model not found. Download lbfmodel.yaml and pass --lbf-model."
        )
    facemark = load_facemark(lbf_model_path)

    image_paths = list(iter_images(input_dir))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    processed = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_rects = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        faces_rects = faces_rects[: args.max_faces]

        faces = []
        if len(faces_rects) > 0:
            _, landmarks_all = facemark.fit(gray, faces_rects)
            for face_idx, (rect, lm) in enumerate(
                zip(faces_rects, landmarks_all)
            ):
                x, y, w, h = rect
                landmarks = []
                for point in lm[0]:
                    landmarks.append(
                        {
                            "x": float(point[0] / width),
                            "y": float(point[1] / height),
                            "z": 0.0,
                            "x_px": int(point[0]),
                            "y_px": int(point[1]),
                        }
                    )
                left_eye_state = estimate_eye_state(
                    landmarks, LEFT_EYE, args.eye_open_threshold
                )
                right_eye_state = estimate_eye_state(
                    landmarks, RIGHT_EYE, args.eye_open_threshold
                )
                head_tilt = estimate_head_tilt(
                    landmarks, args.tilt_threshold_deg
                )
                faces.append(
                    {
                        "face_index": face_idx,
                        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                        "landmarks": landmarks,
                        "eyes": {
                            "left": left_eye_state,
                            "right": right_eye_state,
                        },
                        "head_pose": head_tilt,
                    }
                )

            output = {
                "image": {
                    "path": str(image_path),
                    "width": width,
                    "height": height,
                },
                "faces": faces,
            }

            json_path = output_dir / f"{image_path.stem}_landmarks.json"
            json_path.write_text(json.dumps(output, indent=2))

        if args.draw:
            annotated = image.copy()
            for face in faces:
                for point in face["landmarks"]:
                    cv2.circle(
                        annotated,
                        (point["x_px"], point["y_px"]),
                        1,
                        (0, 255, 0),
                        1,
                    )
                left_status = face["eyes"]["left"]["state"]
                right_status = face["eyes"]["right"]["state"]
                roll = face["head_pose"]["roll_deg"]
                tilt = face["head_pose"]["tilt"]
                left_pt = (
                    face["landmarks"][LEFT_EYE["outer"]]["x_px"],
                    face["landmarks"][LEFT_EYE["outer"]]["y_px"] - 6,
                )
                right_pt = (
                    face["landmarks"][RIGHT_EYE["outer"]]["x_px"],
                    face["landmarks"][RIGHT_EYE["outer"]]["y_px"] - 6,
                )
                cv2.putText(
                    annotated,
                    f"L:{left_status}",
                    left_pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"R:{right_status}",
                    right_pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                nose_pt = (
                    face["landmarks"][NOSE_TIP]["x_px"],
                    face["landmarks"][NOSE_TIP]["y_px"] - 8,
                )
                cv2.putText(
                    annotated,
                    f"roll:{roll:.1f} {tilt}",
                    nose_pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            out_img = output_dir / f"{image_path.stem}_landmarks.jpg"
            cv2.imwrite(str(out_img), annotated)

        processed += 1
        print(f"Processed {image_path.name} -> {json_path.name}")

    print(f"Done. Processed {processed} image(s). Output in {output_dir}")


if __name__ == "__main__":
    main()
