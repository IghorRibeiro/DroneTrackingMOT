"""
Roda o pipeline completo (detector + tracker) sobre um vídeo original
do dataset, gerando um vídeo anotado com caixas + IDs persistentes.
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path("/Users/ighorribeiro/PycharmProjects/DroneTrackingMOT")
WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "dataset" / "MultiUAV_Train" / "TrainVideos" / "MultiUAV-002.mp4"
OUT_DIR = PROJECT_ROOT / "runs"

TRACKER_CFG = "botsort.yaml"  # troque para "bytetrack.yaml" para comparar depois

model = YOLO(str(WEIGHTS))

results = model.track(
    source=str(VIDEO_PATH),
    conf=0.25,
    tracker=TRACKER_CFG,
    save=True,
    project=str(OUT_DIR),
    name=f"track_test_{TRACKER_CFG.split('.')[0]}",
    persist=True,
    device="mps",
)

print(f"Vídeo anotado salvo em: {OUT_DIR / f'track_test_{TRACKER_CFG.split(chr(39))[0]}'}")