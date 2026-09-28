# LVMT Model Zoo - DINOv3

> FPS measured on NVIDIA H100 with default torch.compile.

The `Init Weights` column holds the **stage-1 (segmenter)** checkpoint used to initialize
training of the online model. The `Download` column holds the final **stage-2 (online)**
LVMT checkpoint. Scores are those of the released checkpoint; the paper reports the mean
over five runs, so numbers may differ slightly.

## Video Instance Segmentation

### YouTube-VIS 2019

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">AP</th>
<th valign="bottom">AR<sub>10</sub></th>
<th valign="bottom">FPS</th>
<th valign="bottom">Init Weights</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: LVMT-L -->
<tr><td align="left"><a href="../configs/ytvis19/lvmt/vit-large/lvmt_online_ViTL_dinov3.yaml">LVMT-L</a></td>
<td align="center">69.6</td>
<td align="center">74.7</td>
<td align="center">122</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/yt-2019_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov3_online/yt_2019_dinov3_online_69.6.pth?download=true">Online Weights</a></td>
</tr>
</tbody></table>

### YouTube-VIS 2021

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">AP</th>
<th valign="bottom">AR<sub>10</sub></th>
<th valign="bottom">FPS</th>
<th valign="bottom">Init Weights</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: LVMT-L -->
<tr><td align="left"><a href="../configs/ytvis21/lvmt/vit-large/lvmt_online_ViTL_dinov3.yaml">LVMT-L</a></td>
<td align="center">65.7</td>
<td align="center">69.6</td>
<td align="center">122</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/yt-2021_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov3_online/yt_2021_dinov3_online_65.7.pth?download=true">Online Weights</a></td>
</tr>
</tbody></table>

### YouTube-VIS 2022

Evaluated with the YouTube-VIS 2021 model; the 2022 validation set adds the long videos.
Scores are reported on the long-video subset.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">AP<sup>L</sup></th>
<th valign="bottom">AR<sup>L</sup><sub>10</sub></th>
<th valign="bottom">FPS</th>
<th valign="bottom">Init Weights</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: LVMT-L -->
<tr><td align="left"><a href="../configs/ytvis22/lvmt/vit-large/lvmt_online_ViTL_dinov3.yaml">LVMT-L</a></td>
<td align="center">51.5</td>
<td align="center">55.4</td>
<td align="center">121</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/yt-2021_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov3_online/yt_2022_dinov3_online_51.5.pth?download=true">Online Weights</a></td>
</tr>
</tbody></table>

### OVIS

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">AP</th>
<th valign="bottom">AR<sub>10</sub></th>
<th valign="bottom">FPS</th>
<th valign="bottom">Init Weights</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: LVMT-S -->
<tr><td align="left"><a href="../configs/ovis/lvmt/vit-small/lvmt_online_ViTS_dinov3.yaml">LVMT-S</a></td>
<td align="center">39.1</td>
<td align="center">45.8</td>
<td align="center">182</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/ovis_small_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov3_online/ovis_small_dinov3_online_39.1.pth?download=true">Online Weights</a></td>
</tr>
<!-- ROW: LVMT-B -->
<tr><td align="left"><a href="../configs/ovis/lvmt/vit-base/lvmt_online_ViTB_dinov3.yaml">LVMT-B</a></td>
<td align="center">48.1</td>
<td align="center">53.6</td>
<td align="center">155</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/ovis_base_dinov3_sgemnter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov3_online/ovis_base_dinov3_online_48.1.pth?download=true">Online Weights</a></td>
</tr>
<!-- ROW: LVMT-L -->
<tr><td align="left"><a href="../configs/ovis/lvmt/vit-large/lvmt_online_ViTL_dinov3.yaml">LVMT-L</a></td>
<td align="center">56.7</td>
<td align="center">61.6</td>
<td align="center">95</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/ovis_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov3_online/ovis_dinov3_online_56.5.pth?download=true">Online Weights</a></td>
</tr>
</tbody></table>

## Video Panoptic Segmentation

### VIPSeg

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">VPQ</th>
<th valign="bottom">STQ</th>
<th valign="bottom">FPS</th>
<th valign="bottom">Init Weights</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: LVMT-L -->
<tr><td align="left"><a href="../configs/VIPSeg/lvmt/vit-large/lvmt_Online_ViTL_dinov3.yaml">LVMT-L</a></td>
<td align="center">60.3</td>
<td align="center">53.4</td>
<td align="center">57</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/vipseg_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov3_online/vipseg_dinov3_online_60.3_53.4.pth?download=true">Online Weights</a></td>
</tr>
</tbody></table>

## Video Semantic Segmentation

### VSPW

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">mVC<sub>16</sub></th>
<th valign="bottom">mIoU</th>
<th valign="bottom">FPS</th>
<th valign="bottom">Init Weights</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: LVMT-L -->
<tr><td align="left"><a href="../configs/VSPW/lvmt/vit-large/lvmt_online_ViTL_dinov3.yaml">LVMT-L</a></td>
<td align="center">95.3</td>
<td align="center">66.4</td>
<td align="center">57</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov3_segmenter/vspw_dinov3_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov3_online/vspw_dinov3_online_95.3_66.4.pth?download=true">Online Weights</a></td>
</tr>
</tbody></table>
