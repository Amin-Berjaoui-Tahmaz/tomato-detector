"""Fine-tune RF-DETR Nano (Apache-2.0, COCO-pretrained) on the tomato splits, then export ONNX."""
import sys

from rfdetr import RFDETRNano

# 576 instead of Nano's default 384: tomatoes are ~40 px in 1280x720 frames and shrink to
# ~12 px at 384. Costs CPU speed; measured in evaluate.py.
RES = 576
OUT = "runs/nano"

if __name__ == "__main__":
    if "--export-only" not in sys.argv:
        RFDETRNano(resolution=RES).train(
            dataset_dir="data/coco", output_dir=OUT, epochs=60, batch_size=4, grad_accum_steps=4, lr=1e-4,
            # 0 workers: on Windows each worker is a full Python+torch process (~1.5 GB RAM);
            # with 2 the 16 GB laptop ran out of RAM while saving a checkpoint (MemoryError).
            num_workers=0)
    # Best validation epoch (EMA weights), not the last one: val mAP peaks early on 252 images.
    # Same weights exported at 3 input sizes: smaller input = faster on CPU, less accurate (trained multi-scale).
    for r in (576, 448, 384):
        RFDETRNano(resolution=r, pretrain_weights=f"{OUT}/checkpoint_best_total.pth").export(
            output_dir=f"{OUT}/r{r}", verbose=False)
