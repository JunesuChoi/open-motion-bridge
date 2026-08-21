# MMPose WholeBody provider

`mmpose-rtmpose-l-wholebody` is the opt-in high-precision provider for face, hands, feet, and body anchors. It is intended for graphics that must follow a wrist, palm, face, or individual hand landmark. The default `mediapipe` provider remains available for lightweight local analysis.

## Runtime and model policy

1. Prepare a PyTorch build appropriate for the target CPU/CUDA environment.
2. Follow the [MMPose installation compatibility guidance](https://mmpose.readthedocs.io/en/latest/installation.html) for the matching MMCV and MMDetection runtime.
3. Install this project's optional packages only after that environment is ready:

   ```powershell
   pip install -e ".\packages\analyzer-python[mmpose]"
   ```

4. Store the RTMPose-L COCO-WholeBody config/checkpoint and a compatible local person-detector config/checkpoint outside this repository.

The command refuses absent local files, an unavailable MMPose runtime, malformed predictions, and unsupported provider output. It does not download aliases or weights and does not fall back to MediaPipe.

MMPose's inferencer supports supplying local pose and detector configs/checkpoints; its model-zoo documents RTMPose WholeBody variants. [Inference guide](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html) · [WholeBody model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/wholebody_2d_keypoint.html)

## Run

```powershell
python -m open_motion_bridge analyze "D:\input\clip.mp4" --output "D:\analysis" `
  --pose-provider mmpose-rtmpose-l-wholebody `
  --mmpose-pose-config "D:\models\rtmpose-l-wholebody.py" `
  --mmpose-pose-weights "D:\models\rtmpose-l-wholebody.pth" `
  --mmpose-detector-config "D:\models\person-detector.py" `
  --mmpose-detector-weights "D:\models\person-detector.pth" `
  --mmpose-device cuda:0 --force
```

Use `--mmpose-device cpu` only where GPU execution is unavailable; it is expected to be substantially slower for full-video analysis.

## Output and review

The resulting immutable Tracking IR records the MMPose version, only asset file names, device, selection policy, 133 stable landmark names, and source hash. WholeBody scores become normalized visibility values; MMPose 2D output does not imply metric depth.

This initial provider exports one primary person. It selects the most confident/largest candidate at the start and then follows the highest bbox-IoU candidate. A weak match is written to `lifecycle.idChanges`, receives a drift warning, and requires review. Do not use it as proof of persistent multi-person identity tracking.

MMPose is Apache-2.0, but the repository's MIT license does not automatically license model weights or detector checkpoints. Verify the license and redistribution terms for every chosen local checkpoint before publishing a generated project or bundle.
