# FOM-SAM3 Policy

Code for **Towards Fine-Grained Object Manipulation: SAM3-Guided Visuomotor Policy with Persistent Memory Learning and Focused Visual Conditioning**.

[Paper](https://arxiv.org/abs/2609.21621) · [Project page](https://hlmeng-casia.github.io/FOM-SAM3-Policy/)

Includes FO Memory learning, frozen SAM3 inference, and FSAE conditioning for Diffusion Policy and ACT. Data, checkpoints, and robot deployment code are not included.

Install with `pip install -r requirements.txt`, install [SAM3](https://github.com/facebookresearch/sam3) separately, and set your paths in `configs/`. Training and inference entry points are in `train/` and `inference/`.

MIT license. See [third-party notices](THIRD_PARTY_NOTICES.md) for dependencies.
