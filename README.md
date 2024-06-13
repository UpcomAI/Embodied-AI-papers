# Embodied-AI-papers
[![Awesome](https://awesome.re/badge.svg)](https://github.com/UpcomAI/Embodied-AI-papers/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/UpcomAI/Embodied-AI-papers/blob/main/LICENSE)
![](https://img.shields.io/github/last-commit/UpcomAI/Embodied-AI-papers?color=green) 
![](https://img.shields.io/badge/PRs-Welcome-red) 

> What can LLMs do for ROBOTs? 

🙌 This repository collects papers integrating **Embodied AI** and **large language models (LLMs)**.

😎 Welcome to recommend missing papers through **`Adding Issues`** or **`Pull Requests`**. 

🥽 The papers are ranked according to our **subjective** opinions.

## 📜 Table of Content

- [Embodied-AI-papers](#embodied-ai-papers)
  - [📜 Table of Content](#-table-of-content)
  - [✨︎ Outstanding Papers](#︎-outstanding-papers)
  - [📥 Paper Inbox](#-paper-inbox)
    - [Survey](#survey)
    - [Datasets \& Simulator](#datasets--simulator)
    - [Algorithms](#algorithms)
    - [Applications](#applications)
      - [Perception](#perception)
      - [Policy](#policy)
      - [Action](#action)
      - [Control](#control)
    - [System Implementation](#system-implementation)

## ✨︎ Outstanding Papers

- \[[arXiv 2024](https://arxiv.org/pdf/2401.02117)\] Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation \[[Project](https://mobile-aloha.github.io)\]
- \[[arXiv 2024](https://arxiv.org/pdf/2402.05741)\] Real-World Robot Applications of Foundation Models: A Review
- \[[arXiv 2023](https://arxiv.org/abs/2210.01911)\] Grounding Language with Visual Affordances over Unstructured Data (**HULC++**)

## 📥 Paper Inbox

### Survey

- \[[arXiv 2024](https://arxiv.org/pdf/2312.10807)] Language-conditioned Learning for Robotic Manipulation: A Survey

### Datasets \& Simulator

- \[[CoRL 2023 Workshop TGR](https://openreview.net/forum?id=zraBtFgxT0&noteId=kNJ60a3jR5)\] Open X-Embodiment: Robotic Learning Datasets and RT-X Models \[[Project](https://robotics-transformer-x.github.io)\]
- \[[IEEE RA-L 2023](https://ieeexplore.ieee.org/document/10107764)\] Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments \[[Project](https://isaac-orbit.github.io)\]
- \[[arXiv 2023](https://arxiv.org/pdf/2308.00055)\] Towards Building AI-CPS with NVIDIA Isaac Sim: An Industrial Benchmark and Case Study for Robotics Manipulation \[[Project](https://sites.google.com/view/ai-cps-robotics-manipulation/home)\]
- \[[IROS 2023](https://ieeexplore.ieee.org/abstract/document/10341672)\] HANDAL: A Dataset of Real-World Manipulable Object Categories with Pose Annotations, Affordances, and Reconstructions \[[Project](https://nvlabs.github.io/HANDAL/)\]
- Physically grounded vision-language models for robotic manipulation
- Affordance detection of tool parts from geometric features
- Learning affordance grounding from exocentric images
- Bridgedata v2: A dataset for robot learning at scale
- Rh20t: A robotic dataset for learning diverse skills in one-shot
- Droid: A large-scale in-the-wild robot manipulation dataset
- Ar2-d2: Training a robot without a robot
- OAKINK2: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion `Dual-Arm`

### Algorithms

- Rt-1: Robotics transformer for real-world control at scale
- Palm-e: An embodied multimodal language model
- Rt-2: Vision-language-action models transfer web knowledge to robotic control
- Steve-1: A generative model for text-to-behavior in minecraft
- Q-transformer: Scalable offline reinforcement learning via autoregressive q-functions
- Diffusion model is an effective planner and data synthesizer for multi-task reinforcement learning
- Liv: Language-image representations and rewards for robotic control
- Supervised pretraining can learn in-context reinforcement learning
- Robocat: A self-improving foundation agent for robotic manipulation
- Vint: A foundation model for visual navigation
- Foundation reinforcement learning: towards embodied generalist agents with foundation prior assistance

### Applications

#### Perception

- Affordancellm: Grounding affordance from vision language models
- \[[arXiv 2024](https://arxiv.org/pdf/2309.02561)\] Physically Grounded Vision-Language Models for Robotic Manipulation
- \[[CoRL 2023](https://openreview.net/forum?id=8yTS_nAILxt)\] REFLECT: Summarizing Robot Experiences for FaiLure Explanation and CorrecTion \[[Project](https://robot-reflect.github.io/)\]
- Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding
- \[[ICRA 2022](https://ieeexplore.ieee.org/abstract/document/9811889)\] Affordance Learning from Play for Sample-Efficient Policy Learning \[[Project](http://vapo.cs.uni-freiburg.de)\] (**VAPO**) `[w/o LLM]`
- \[[IEEE RA-L 2022](https://ieeexplore.ieee.org/abstract/document/9849097)\] What Matters in Language Conditioned Robotic Imitation Learning over Unstructured Data \[[Project](http://hulc.cs.uni-freiburg.de)\]

#### Policy

- \[[ICRA 2024 Workshop VLMNM](https://openreview.net/forum?id=jGrtIvJBpS)\] Octo: An Open-Source Generalist Robot Policy
- Language models as zero-shot trajectory generators
- Mutex: Learning unified policies from multimodal task specifications
- \[[arXiv 2024](https://arxiv.org/pdf/2403.12761)\] BTGenBot: Behavior Tree Generation for Robotic Tasks with Lightweight LLMs
- \[[arXiv 2024](https://arxiv.org/abs/2403.17124)\] Grounding Language Plans in Demonstrations Through Counterfactual Perturbations
- \[[arXiv 2024](https://arxiv.org/pdf/2403.11289)\] ManipVQA: Injecting Robotic Affordance and Physically Grounded Information into Multi-Modal Large Language Models \[[Code](https://github.com/SiyuanHuang95/ManipVQA)\]
- \[[ICRA 2023](https://ieeexplore.ieee.org/abstract/document/10160591)\] Code as Policies: Language Model Programs for Embodied Control \[[Project](https://code-as-policies.github.io)\]
- \[[CoRL 2023](https://openreview.net/forum?id=4ZK8ODNyFXx)\] Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners \[[Project](https://robot-help.github.io)\]
- Roboclip: One demonstration is enough to learn robot policies
- Skill transformer: A monolithic policy for mobile manipulation
- Generalizable long-horizon manipulations with large language models
- LLM-MARS: Large Language Model for Behavior Tree Generation and NLP-enhanced Dialogue in Multi-Agent Robot Systems

#### Action

- \[[arXiv 2024](https://arxiv.org/pdf/2403.03949)\] Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation \[[Project](https://real-to-sim-to-real.github.io/RialTo/)\]
- \[[arXiv 2024](https://arxiv.org/abs/2305.19075)\] Language-Conditioned Imitation Learning with Base Skill Priors under Unstructured Data
- Zero-shot robotic manipulation with pretrained image-editing diffusion models
- \[[NeurIPS 2023 Poster](https://openreview.net/forum?id=KtvPdGb31Z&referrer=%5Bthe%20profile%20of%20Anji%20Liu%5D(%2Fprofile%3Fid%3D~Anji_Liu1))\] Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents \[[Project](https://github.com/CraftJarvis/MC-Planner.)\] `[Minecraft]`
- Open-world object manipulation using pre-trained vision-language models
- Pave the way to grasp anything: Transferring foundation models for universal pick-place robots
- Waypoint-based imitation learning for robotic manipulation
- MOMA-Force: Visual-Force Imitation for Real-World Mobile Manipulation
- Object-centric instruction augmentation for robotic manipulation


#### Control

- \[[arXiv 2024](https://arxiv.org/pdf/2403.05304)\] Spatiotemporal Predictive Pre-training for Robotic Motor Control
- \[[PMLR 2023](https://proceedings.mlr.press/v229/radosavovic23a.html)\] Robot learning with sensorimotor pre-training
- \[[arXiv 2023](https://arxiv.org/abs/2305.10912)\] A generalist dynamics model for control
- \[[ICLR 2023 RRL Poster](https://openreview.net/forum?id=TIV7eEY8qY)\] Chain-of-thought predictive control
- \[[ICML 2023](https://dl.acm.org/doi/abs/10.5555/3618408.3618914)\] On pre-training for visuo-motor control: Revisiting a learning-from-scratch baseline
- \[[arXiv 2023](https://arxiv.org/pdf/2309.14236)\] MoDem-V2: Visuo-Motor World Models for Real-World Robot Manipulation

### System Implementation

- \[[SIGCHI 2024](https://dl.acm.org/doi/proceedings/10.1145/3610978?tocHeading=heading6)] Language, Camera, Autonomy! Prompt-engineered Robot Control for Rapidly Evolving Deployment (**CLEAR**) `[Software]`
- \[[Autonomous Robots 2023](https://link.springer.com/article/10.1007/s10514-023-10139-z)\] TidyBot: personalized robot assistance with large language models \[[Project](https://tidybot.cs.princeton.edu)\]
- \[[RSS 2024](https://roboticsconference.org/program/papers/016/)\]Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (**ALOHA**)
- \[[arXiv 2024](https://arxiv.org/pdf/2404.03570)\] Embodied AI with Two Arms: Zero-shot Learning, Safety and Modularity `Dual-Arm`
  - \[[arXiv 2023](https://arxiv.org/pdf/2305.10403)\] Palm 2 technical report
  - \[[arXiv 2022](https://arxiv.org/pdf/2205.06230)\] Simple Open-Vocabulary Object Detection with Vision Transformers
  - \[[ICRA 2023](https://ieeexplore.ieee.org/abstract/document/10161283/)\] Robotic Table Wiping via Reinforcement Learning and Whole-body Trajectory Optimization