"""
pipeline.py - VideoMAE + YOLOv8 사람 검출 + 투기자 강조 (트래킹/ID 없음)
"""
import argparse, json, os
import cv2, numpy as np
import torch
from decord import VideoReader, cpu
from ultralytics import YOLO
from model import GarbageDumpingVideoMAE

VIDEOMAE_NUM_FRAMES = 16
VIDEOMAE_IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
CHECKPOINT   = "/data/leecg1219/KHUDA_173/checkpoints/best_model.pt"
DUMP_MODEL   = "/data/leecg1219/KHUDA_173/yolo_runs/trash_dump_ep5/weights/best.pt"
PERSON_MODEL = "yolov8m.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./pipeline_outputs")
    p.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    p.add_argument("--dump_model", type=str, default=DUMP_MODEL)
    p.add_argument("--person_model", type=str, default=PERSON_MODEL)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--clip_frames", type=int, default=48)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--illegal_thresh", type=float, default=0.85)
    p.add_argument("--dump_conf", type=float, default=0.65)
    p.add_argument("--person_conf", type=float, default=0.4)
    p.add_argument("--iou_thresh", type=float, default=0.1)
    return p.parse_args()


def load_videomae(checkpoint, device):
    model = GarbageDumpingVideoMAE().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def frames_to_tensor(frames_rgb):
    resized = np.stack([cv2.resize(f, (VIDEOMAE_IMAGE_SIZE, VIDEOMAE_IMAGE_SIZE)) for f in frames_rgb])
    normed = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normed).permute(0, 3, 1, 2).float()


@torch.no_grad()
def sliding_window_inference(vr, videomae, device, clip_frames, stride, thresh):
    total = len(vr)
    frame_scores = np.zeros(total)
    frame_counts = np.zeros(total)
    start = 0
    while start + clip_frames <= total:
        indices = np.linspace(start, start + clip_frames - 1, num=VIDEOMAE_NUM_FRAMES).astype(int)
        frames = vr.get_batch(indices).asnumpy()
        tensor = frames_to_tensor(frames).unsqueeze(0).to(device)
        outputs = videomae(tensor)
        prob = torch.softmax(outputs["clip_logits"], dim=1)[0, 1].item()
        for idx in indices:
            frame_scores[idx] += prob
            frame_counts[idx] += 1
        start += stride
    avg = np.where(frame_counts > 0, frame_scores / frame_counts, 0)

    # frame_counts=0인(한 번도 평가 안 된) 프레임은 0이 아니라 결측치이므로
    # 양쪽의 평가된 프레임 값으로 선형 보간해서 채운다.
    evaluated = frame_counts > 0
    if evaluated.sum() >= 2:
        idx_all = np.arange(len(avg))
        avg = np.interp(idx_all, idx_all[evaluated], avg[evaluated])

    high = [i for i, s in enumerate(avg) if s >= thresh]
    segs = []
    MAX_MERGE_GAP = 4  # 이 프레임 수 이내의 끊김만 이어붙임 (10 -> 4로 강화)
    if high:
        seg = [high[0]]
        for f in high[1:]:
            if f - seg[-1] <= MAX_MERGE_GAP:
                seg.append(f)
            else:
                if len(seg) >= 3:
                    segs.append((seg[0], seg[-1], float(avg[seg[0]:seg[-1]+1].mean())))
                seg = [f]
        if len(seg) >= 3:
            segs.append((seg[0], seg[-1], float(avg[seg[0]:seg[-1]+1].mean())))
    return segs, avg


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*50, flush=True)
    print("쓰레기 무단투기 탐지 파이프라인 (사람 검출 + 투기자 강조)", flush=True)
    print(f"영상: {args.video_path}", flush=True)
    print("="*50, flush=True)

    print("\n[1/3] 모델 로드...", flush=True)
    videomae     = load_videomae(args.checkpoint, args.device)
    dump_model   = YOLO(args.dump_model)
    person_model = YOLO(args.person_model)

    cap = cv2.VideoCapture(args.video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    vr = VideoReader(args.video_path, ctx=cpu(0))
    print(f"  ✅ {total}프레임, {fps:.1f}fps, {W}x{H}", flush=True)

    print("\n[2/3] VideoMAE 분석...", flush=True)
    segs, avg = sliding_window_inference(vr, videomae, args.device,
                                          args.clip_frames, args.stride, args.illegal_thresh)
    print(f"  ✅ illegal 구간 {len(segs)}개:", flush=True)
    for s, e, p in segs:
        print(f"    frame {s}~{e} (prob={p:.3f})", flush=True)

    print("\n[3/3] 결과 영상 생성 (트래킹 없음, 투기자만 강조)...", flush=True)

    vname = os.path.splitext(os.path.basename(args.video_path))[0]
    out_path = os.path.join(args.output_dir, f"{vname}_result.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    dumping_frame_count = 0

    cap = cv2.VideoCapture(args.video_path)
    for fidx in range(total):
        ret, frame = cap.read()
        if not ret: break

        score  = avg[fidx] if fidx < len(avg) else 0
        in_ill = any(s <= fidx <= e for s, e, _ in segs)

        dumping_box = None
        if in_ill:
            cv2.rectangle(frame, (0, 0), (W, H), (0, 0, 255), 8)
            dr = dump_model(frame, conf=args.dump_conf, verbose=False)
            if dr[0].boxes is not None and len(dr[0].boxes) > 0:
                best = max(dr[0].boxes, key=lambda b: float(b.conf[0]))
                dumping_box = tuple(best.xyxy[0].tolist())
                dumping_frame_count += 1

        pr = person_model(frame, conf=args.person_conf, classes=[0], verbose=False)
        if pr[0].boxes is not None:
            for box in pr[0].boxes:
                bbox = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, bbox)

                is_dumper = dumping_box is not None and iou(bbox, dumping_box) > args.iou_thresh

                if is_dumper:
                    color, thickness, label = (0, 0, 255), 4, "DUMPING"
                else:
                    color, thickness, label = (255, 100, 0), 2, "person"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        status = "ILLEGAL" if in_ill else "normal"
        cv2.putText(frame, f"Frame:{fidx} | {status} | score:{score:.2f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if in_ill else (0, 255, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    result = {
        "illegal_segments": [{"start": s, "end": e, "prob": p} for s, e, p in segs],
        "dumping_frames_detected": dumping_frame_count,
    }
    with open(os.path.join(args.output_dir, f"{vname}_events.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}", flush=True)
    print(f"✅ 완료! illegal:{len(segs)}개, 투기bbox탐지된 프레임:{dumping_frame_count}개", flush=True)
    print(f"결과: {out_path}", flush=True)


if __name__ == "__main__":
    main()
