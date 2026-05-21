import numpy as np


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denom = areaA + areaB - interArea
    if denom <= 0:
        return 0.0

    return interArea / denom


class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_missed=10):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_idx):
        summaries = []

        used = set()

        for tid, tr in list(self.tracks.items()):
            tr["missed"] += 1

        for det in detections:
            best_id = None
            best_iou = self.iou_thresh

            for tid, tr in self.tracks.items():
                iou_val = iou(det, tr["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = tid

            if best_id is not None:
                tr = self.tracks[best_id]
                old_cx = (tr["bbox"][0] + tr["bbox"][2]) / 2
                old_cy = (tr["bbox"][1] + tr["bbox"][3]) / 2
                new_cx = (det[0] + det[2]) / 2
                new_cy = (det[1] + det[3]) / 2

                step = np.sqrt((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2)

                tr["bbox"] = det
                tr["missed"] = 0
                tr["total_distance"] += step
                tr["last_step"] = step

                summaries.append(tr)
                used.add(best_id)
            else:
                self.tracks[self.next_id] = {
                    "id": self.next_id,
                    "bbox": det,
                    "missed": 0,
                    "total_distance": 0.0,
                    "last_step": 0.0,
                }
                summaries.append(self.tracks[self.next_id])
                used.add(self.next_id)
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if tid not in used:
                self.tracks[tid]["last_step"] = 0.0
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        summaries.sort(key=lambda x: x["id"])
        return summaries
