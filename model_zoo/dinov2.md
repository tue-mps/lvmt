# LVMT Model Zoo - DINOv2

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
<tr><td align="left"><a href="../configs/ytvis19/lvmt/vit-large/lvmt_online_ViTL_dinov2.yaml">LVMT-L</a></td>
<td align="center">70.1</td>
<td align="center">74.4</td>
<td align="center">127</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_segmenter/yt-2019_dinov2_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov2_online/yt_2019_dinov2_online_70.1.pth?download=true">Online Weights</a></td>
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
<tr><td align="left"><a href="../configs/ytvis21/lvmt/vit-large/lvmt_online_ViTL_dinov2.yaml">LVMT-L</a></td>
<td align="center">66.4</td>
<td align="center">70.4</td>
<td align="center">127</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_segmenter/yt-2021_dinov2_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov2_online/yt_2021_dinov2_online_66.4.pth?download=true">Online Weights</a></td>
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
<tr><td align="left"><a href="../configs/ytvis22/lvmt/vit-large/lvmt_online_ViTL_dinov2.yaml">LVMT-L</a></td>
<td align="center">48.4</td>
<td align="center">52.1</td>
<td align="center">127</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_segmenter/yt-2021_dinov2_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov2_online/yt_2022_dinov2_online_48.4.pth?download=true">Online Weights</a></td>
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
<!-- ROW: LVMT-L -->
<tr><td align="left"><a href="../configs/ovis/lvmt/vit-large/lvmt_online_ViTL_dinov2.yaml">LVMT-L</a></td>
<td align="center">55.6</td>
<td align="center">60.5</td>
<td align="center">97</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/blob/main/Dinov2_segmenter/ovis_dinov2_segmenter.pth">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov2_online/ovis_dinov2_online_55.6.pth?download=true">Online Weights</a></td>
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
<tr><td align="left"><a href="../configs/VIPSeg/lvmt/vit-large/lvmt_Online_ViTL_dinov2.yaml">LVMT-L</a></td>
<td align="center">56.6</td>
<td align="center">51.3</td>
<td align="center">59</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_segmenter/vipseg_dinov2_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov2_online/vipseg_dinov2_online_56.6_51.3.pth?download=true">Online Weights</a></td>
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
<tr><td align="left"><a href="../configs/VSPW/lvmt/vit-large/lvmt_online_ViTL_dinov2.yaml">LVMT-L</a></td>
<td align="center">95.2</td>
<td align="center">65.4</td>
<td align="center">59</td>
<td align="center"><a href="https://huggingface.co/tue-mps/PMT_Video/resolve/main/Dinov2_segmenter/vspw_dinov2_segmenter.pth?download=true">Segmenter Weights</a></td>
<td align="center"><a href="https://huggingface.co/tue-mps/LVMT/resolve/main/Dinov2_online/vspw_dinov2_online_95.2_65.4.pth?download=true">Online Weights</a></td>
</tr>
</tbody></table>
