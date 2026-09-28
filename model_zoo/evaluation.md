## Evaluation

To evaluate a pre-trained LVMT model, first prepare the datasets by following the instructions
in [datasets/README.md](../datasets/README.md), then download a checkpoint from the
[DINOv2 model zoo](dinov2.md) or the [DINOv3 model zoo](dinov3.md).

All commands below are run from the repository root.

### YouTube-VIS 2019, YouTube-VIS 2021, OVIS

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /path/to/output
```

🔧 Replace `/path/to/config.yaml` with the path to the config file.  
🔧 Replace `/path/to/weight.pth` with the path to the checkpoint to evaluate.  
🔧 Replace `/path/to/output` with the path to the output folder.  
🔧 Change the value of `--num-gpus` to the number of GPUs available to you.

YouTube-VIS 2019 and OVIS report the metrics directly. YouTube-VIS 2021 writes a
`results.json` for the [CodaLab server](https://codalab.lisn.upsaclay.fr/competitions/7680).

### YouTube-VIS 2022

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /path/to/output
```

After generating the predictions with the above command, compute the AP over the long videos:

```bash
python utils/yt2022_evaluate.py /path/to/dataset/ytvis_2022 /path/to/output/inference
```

🔧 Replace `/path/to/dataset` with the path to the dataset folder.  
🔧 Replace `/path/to/output` with the path to the output folder.

### VIPSeg

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /path/to/output
```

After generating the predictions with the above command, compute VPQ and STQ:

```bash
DATAROOT='/path/to/dataset/VIPSeg_720P/panomasksRGB'
IMGSAVEROOT='/path/to/output/inference'
GT_JSONFILE='/path/to/dataset/VIPSeg_720P/panoptic_gt_VIPSeg_val.json'

# VPQ
python utils/eval_vpq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE
# STQ
python utils/eval_stq_vspw.py --submit_dir $IMGSAVEROOT --truth_dir $DATAROOT --pan_gt_json_file $GT_JSONFILE
```

🔧 Replace `/path/to/dataset` with the path to the dataset folder.  
🔧 Replace `/path/to/output` with the path to the output folder.

### VSPW

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \
  OUTPUT_DIR /path/to/output
```

After generating the predictions with the above command, compute mIoU and mVC<sub>16</sub>:

```bash
DATAROOT='/path/to/dataset/VSPW_480p'
PREDROOT='/path/to/output/inference'

python utils/eval_miou_vspw.py $DATAROOT $PREDROOT
python utils/eval_vc_vspw.py $DATAROOT $PREDROOT
```

🔧 Replace `/path/to/dataset` with the path to the dataset folder.  
🔧 Replace `/path/to/output` with the path to the output folder.
