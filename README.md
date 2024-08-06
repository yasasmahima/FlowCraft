FlowCraft: Unveiling Adversarial Robustness of LiDAR Scene Flow Estimation
---

[//]: # ([![arXiv]&#40;https://img.shields.io/badge/arXiv-2401.16122-b31b1b?logo=arxiv&logoColor=white&#41;]&#40;https://arxiv.org/abs/2401.16122&#41; )

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deflow-decoder-of-scene-flow-network-in/scene-flow-estimation-on-argoverse-2&#41;]&#40;https://paperswithcode.com/sota/scene-flow-estimation-on-argoverse-2?p=deflow-decoder-of-scene-flow-network-in&#41; )

[//]: # ([![poster]&#40;https://img.shields.io/badge/Poster-6495ed?style=flat&logo=Shotcut&logoColor=wihte&#41;]&#40;https://hkustconnect-my.sharepoint.com/:b:/g/personal/qzhangcb_connect_ust_hk/EXP_uXYmm_tItTWc8MafXHoB-1dVrMnvF1-lCzU1PXAvqQ?e=2FPfBS&#41; )

[//]: # ([![video]&#40;https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white&#41;]&#40;https://youtu.be/bZ4uUv0nDa0&#41;)

Preprint 

Task: Adversarial Attacks against LiDAR Scene Flow Estimation 

Networks: DeFlow, FastFlow3D, NSFP

Pre-trained weights for models are available in [Onedrive link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qzhangcb_connect_ust_hk/Et85xv7IGMRKgqrVeJEVkMoB_vxlcXk6OZUyiPjd4AArIg?e=lqRGhx). 


**Scripts** to run attacks:

- `un_attack.py`: Perform FlowCraft, PGD or CosPGD attacks using local loss (Note: Update scripts->networks->models->loss_func.py-> line 67 to `return error_back, error`)

- `un_attack_global)loss.py`: Perform FlowCraft, PGD or CosPGD attacks using global loss

- `transfer_attack_nsfp.py` : Perform FlowCraft, PGD or CosPGD transfer attacks on NSFP (Test time optimization)

- `transfer_fast_flow.py` : Perform FlowCraft, PGD or CosPGD transfer attacks on FastFlow3D

## Setup

**Environment**: Clone the repo and build the environment, check [detail installation](assets/README.md) for more information. [Conda](https://docs.conda.io/projects/miniconda/en/latest/)/[Mamba](https://github.com/mamba-org/mamba) is recommended.
```
git clone --recursive https://github.com/yasasmahima/SceneFLow-Attack.git
cd SceneFlow-Attack
mamba env create -f environment.yaml
```

mmcv:
```bash
mamba activate deflow
cd ~/SceneFlow-Attack/mmcv && export MMCV_WITH_OPS=1 && export FORCE_CUDA=1 && pip install -e .
```

## Prepare Data

Normally need 10-45 mins finished run following commands totally (my computer 15 mins, our cluster 40 mins).
```bash
python dataprocess/extract_av2.py --av2_type sensor --data_mode train --argo_dir <argo path> --output_dir <out path>
python dataprocess/extract_av2.py --av2_type sensor --data_mode val --mask_dir <mask_path>
python dataprocess/extract_av2.py --av2_type sensor --data_mode test --mask_dir <mask_path>
```


We use the weights provided by official DeFlow code base [Onedrive link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qzhangcb_connect_ust_hk/Et85xv7IGMRKgqrVeJEVkMoB_vxlcXk6OZUyiPjd4AArIg?e=lqRGhx). These checkpoints also include parameters and status of that epoch inside it.


Note: This code base is based on https://github.com/KTH-RPL/DeFlow

## Cite & Acknowledgements

```
@article{zhang2024deflow,
  author={Mahima, K.T. Yasas; Perera, Asanka; Anavatti, Sreenatha; Garratt, Matt},
  title={FlowCraft: Unveiling Adversarial Robustness of LiDAR Scene Flow Estimation},
  journal={},
  year={2024}
}
```
