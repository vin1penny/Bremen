"""Identity of the selected pitch model; person-pose models remain independent."""
from pathlib import Path


ROBOFLOW_SPORTS_REVISION = "42c80c06b6b65a7f89455b89fe31cdf4c38ba227"
ROBOFLOW_PITCH_URL = "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"
ROBOFLOW_PITCH_SHA256 = "28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36"
ROBOFLOW_PITCH_CHECKPOINT = Path(
    "/mnt/storage2/vincent/football-pose/models/pitch-reference/roboflow-football-pitch.pt"
)
