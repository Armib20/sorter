"""Block 3, part 1: train an anomaly detector on one MVTec AD category,
measure it, and export it to OpenVINO.

Run from your project folder, with the .venv active:
    python train_detector.py

Written for Anomalib 2.x. If an import fails, paste the error.
"""

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

CATEGORY = "bottle"  # one product type; MVTec AD has 15


def main():
    # 1. Data. Training uses only good images; testing uses good and defective.
    #    Expects ./datasets/MVTecAD/bottle with train, test and ground_truth.
    datamodule = MVTecAD(root="./datasets/MVTecAD", category=CATEGORY)

    # 2. Model. PatchCore stores features of normal image patches in a
    #    memory bank; there is no gradient training.
    model = Patchcore()

    # 3. Engine. Runs fit, test and export, and writes everything to ./results.
    #    CPU avoids gaps in Apple GPU support for some operations.
    engine = Engine(accelerator="cpu", default_root_dir="./results")

    engine.fit(model=model, datamodule=datamodule)    # build the memory bank
    engine.test(model=model, datamodule=datamodule)   # score held-out images
    engine.export(model=model, export_type=ExportType.OPENVINO)  # .xml + .bin


# Only run when launched directly. Data-loading helper processes import this
# file, and without this guard each of them would start training again.
if __name__ == "__main__":
    main()