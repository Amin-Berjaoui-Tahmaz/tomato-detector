Test set: 152 frames from a separate recording pass (never seen in training); 1772 unripe, 38 turning, 32 ripe boxes.

| input | mAP50 | mAP50-95 | AP50 unripe | AP50 turning | AP50 ripe | CPU ms/frame | CPU FPS |
|---|---|---|---|---|---|---|---|
| 576 px | 0.546 | 0.217 | 0.678 | 0.307 | 0.654 | 307 | 3.3 |
| 448 px | 0.494 | 0.202 | 0.620 | 0.222 | 0.638 | 185 | 5.4 |
| 384 px | 0.450 | 0.178 | 0.564 | 0.181 | 0.605 | 139 | 7.2 |

CPU: model only, onnxruntime 1.32.0 CPUExecutionProvider, all threads, mean of 200 runs after 20 warm-up, laptop Intel Core i9-13900H, FP32.
