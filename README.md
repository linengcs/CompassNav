<div align="center">
<h3>CompassNav: Steering from Path Imitation to Decision Understanding in Navigation</h3>

LinFeng Li <sup>1,2</sup>&nbsp;[Jian Zhao](https://scholar.google.com.sg/citations?hl=en&user=zdhRJCkAAAAJ&view_op=list_works&gmla=AJsN-F4PURIx5GMQHVpprJJBjTsNC62YCHjxGsKOwVhrkZ1aJsLgBiuKPBbAgbdcE5_KNw3OnLQgOVSjlqmS6gc-6ti0M2K5o-klHgoOywFCbdaaGnpis130zvgoZFJkVfmoNKpo8Krp) <sup>2</sup>&nbsp;[Yuan Xie](https://scholar.google.com/citations?user=RN1QMPgAAAAJ&hl=en) <sup>1</sup>&nbsp;[Xin Tan](https://scholar.google.com/citations?hl=zh-CN&user=UY4NCdcAAAAJ&view_op=list_works) <sup>1</sup>&nbsp;[Xuelong Li](https://scholar.google.com/citations?user=ahUibskAAAAJ&hl=en) <sup>2</sup>&nbsp;

<sup>1 </sup>East China Normal University; <sup>2 </sup>The Institute of Artificial Intelligence (TeleAI), China Telecom&nbsp;

[![ArXiv](https://img.shields.io/badge/ArXiv-<2510.10154>-<COLOR>.svg)](https://arxiv.org/abs/2510.10154) [![Webpage](https://img.shields.io/badge/Webpage-CompassNav-<COLOR>.svg)](https://linengcs.github.io/CompassNav/)[![Dataset](https://img.shields.io/badge/Dataset-CompassData-<COLOR>.svg)](https://huggingface.co/datasets/Lineng/CompassData)[![Model](https://img.shields.io/badge/Model-CompassNav7B-<COLOR>.svg)](https://huggingface.co/datasets/Lineng/CompassData)

<p align="center">
  <img src="assets/crop_compass.svg" width="600">>
</p>

</div>

The top panel contrasts our End-to-End Goal Navigation paradigm with traditional approaches. Unlike Vision-Language Navigation (VLN), which relies on dense, step-by-step instructions, and complex Modular Navigation pipelines, CompassNav directly maps a high-level goal (e.g., "find the plant") to an action through integrated spatial logical reasoning.

The bottom panel details our core contribution-how to stimulate model reasoning ability: a paradigm shift from "Path Imitation" to "Decision Understanding." While traditional methods train agents to replicate a single expert trajectory and penalize any deviation, our agent learns to evaluate the relative quality of all feasible paths at each decision point. This approach cultivates a true "internal compass," enabling the agent to make more intelligent and flexible decisions in unseen environments.

## TODO

- [ ] Release Compass-Data-22k
- [ ] Release CompassNav-7B
- [ ] Release CompassNav training code
- [ ] Release CompassNav Object Goal Nav/Instance Image-Goal Nav test code
