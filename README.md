# Centerline Boundary Dice Loss for Vascular Segmentation
- 📃 [**Paper**](https://arxiv.org/abs/2407.01517)

## :bulb: News
* **(July 2, 2024):** Accepted by MICCAI 2024, updated full codebase.
* **(October 13, 2023):** :tada: Our solution, powered by cbDice, won second place 🥈 in FinalTest-CTA-MultiClass and fourth place in FinalTest-MRA-MultiClass at the MICCAI 2023 [TopCoW 🐮](https://topcow23.grand-challenge.org/evaluation/finaltest-cta-multiclass/leaderboard) Challenge.
* **(October 12, 2023):** Released part of the centerline boundary loss codes for [nnU-Net V2](https://github.com/MIC-DKFZ/nnUNet/releases/tag/v2.2).

## Overview
cbDice consists of several main components. The following links will take you directly to the core parts of the codebase:

- cbDice Calculation Demo: The demo is available in the [cbDice_cal_demo](https://github.com/PengchengShi1220/cbDice/tree/main/cbDice_cal_demo) folder. It includes three scenarios: translation, deformation, and diameter imbalance.
- Network Training: The [nnUNetTrainer_variants](https://github.com/PengchengShi1220/cbDice/tree/main/nnUNetTrainer_variants) folder contains the files responsible for network training.
- Loss Function: The loss functions, including [cbDice](https://github.com/PengchengShi1220/cbDice/blob/main/loss/cbdice_loss.py), [clDice](https://github.com/PengchengShi1220/cbDice/blob/main/loss/cldice_loss.py), and [B-DoU](https://github.com/PengchengShi1220/cbDice/blob/main/loss/b_dou_loss.py), are located in the [loss](https://github.com/PengchengShi1220/cbDice/tree/main/loss) folder.

## Installation Guide

Install [cuCIM](https://github.com/rapidsai/cucim) and [cupy](https://github.com/cupy/cupy) for GPU-accelerated distance transform in [MONAI](https://github.com/Project-MONAI/MONAI/blob/64ea76d83a92b7cf7f13c8f93498d50037c3324c/monai/transforms/utils.py#L2193):

```bash
pip install monai

# For CUDA 12.x
pip install cucim-cu12
pip install cupy==12.3

# For CUDA 11.x
pip install cucim-cu11
pip install cupy-cuda11x
```

## Differentiable Binarization

To obtain a differentiable binarized predicted probability map of the foreground, follow these steps:

```python
y_pred_fore = y_pred[:, 1:]
y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0]  # C foreground channels -> 1 channel
y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
y_prob_binary = torch.softmax(y_pred_binary, 1)
y_pred_prob = y_prob_binary[:, 1]  # predicted probability map of foreground
```

## Differentiable Skeletonization

We provide two options for skeletonization:

1. **[Topology-preserving skeletonization](https://github.com/martinmenten/skeletonization-for-gradient-based-optimization)**: This method ensures high topological accuracy but operates at a slower speed. Refer to [skeletonize.py](https://github.com/PengchengShi1220/cbDice/blob/main/loss/skeletonize.py) for implementation details. This method is based on the paper "A Skeletonization Algorithm for Gradient-based Optimization" (ICCV, 2023).

2. **[Morphological skeletonization](https://github.com/jocpae/clDice)**: This method runs faster but offers lower topological accuracy. Refer to [soft_skeleton.py](https://github.com/PengchengShi1220/cbDice/blob/main/loss/soft_skeleton.py) for implementation details. This method is also discussed in the paper mentioned above.

## Weighted Mask and Skeleton Processing

The [get_weights](https://github.com/PengchengShi1220/cbDice/blob/84390a18d2393bfab6f4b3da011cfa1c1d2ec2a1/loss/cbdice_loss.py#L104) function is used to apply weights to the mask and skeleton. If using ground truth (`y_true`), probabilities are not considered. However, for predictions (`pred`), probabilities must be taken into account.

1. **Distance Transform Computation**:
    - `mask_input` and `skel_input` are processed using the [distance_transform_edt](https://github.com/PengchengShi1220/cbDice/blob/84390a18d2393bfab6f4b3da011cfa1c1d2ec2a1/loss/cbdice_loss.py#L115) function to obtain `dist_map_norm`, `skel_R_norm`, and `I_norm`.

2. **Probability Multiplication**:
    - The distance maps are then multiplied by their respective probabilities:
        - `dist_map_norm` (denoted as Q_vp) is multiplied by `mask_prob`.
        - `skel_R_norm` (denoted as Q_spvp) is multiplied by `mask_prob`.
        - `I_norm` (denoted as Q_sp) is multiplied by `skel_prob`.

For detailed implementation, see the [get_weights](https://github.com/PengchengShi1220/cbDice/blob/84390a18d2393bfab6f4b3da011cfa1c1d2ec2a1/loss/cbdice_loss.py#L104) function.

If you have any issues or need further assistance, feel free to open an issue on our GitHub repository.

## Citation
If you use cbDice in your research, please cite:

```
@article{shi2024centerline,
  title={Centerline Boundary Dice Loss for Vascular Segmentation},
  author={Shi, Pengcheng and Hu, Jiesi and Yang, Yanwu and Gao, Zilve and Liu, Wei and Ma, Ting},
  journal={arXiv preprint arXiv:2407.01517},
  year={2024}
}
```
