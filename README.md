# Centerline Boundary Dice Loss for Vascular Segmentation

<div align="center">
  | 📃 [**Paper**](https://arxiv.org/abs/2407.01517) |
</div>

## :bulb: News
* **(July 2, 2024):** Accepted by MICCAI 2024, updated full codebase.
* **(October 13, 2023):** :tada: Our solution, powered by cbDice, won second place 🥈 in FinalTest-CTA-MultiClass and fourth place in FinalTest-MRA-MultiClass at the MICCAI 2023 [TopCoW 🐮](https://topcow23.grand-challenge.org/evaluation/finaltest-cta-multiclass/leaderboard) Challenge.
* **(October 12, 2023):** Released part of the centerline boundary loss codes for [nnU-Net V2](https://github.com/MIC-DKFZ/nnUNet/releases/tag/v2.2).


cbDice consists of several main components. The following links will take you directly to the core parts of the codebase:

- cbDice Calculation Demo: The network architecture is available in the [cbDice_cal_demo](https://github.com/PengchengShi1220/cbDice/tree/main/cbDice_cal_demo) folder. This includes three scenarios: translation, deformation, and diameter imbalance.
- Network Training: The [nnUNetTrainer_variants](https://github.com/PengchengShi1220/cbDice/tree/main/nnUNetTrainer_variants) folder contains the files responsible for network training.
- Loss Function: The loss functions, including cbDice, clDice, and B-DoU, are located in the [loss](https://github.com/PengchengShi1220/cbDice/tree/main/loss) folder.
