# diffusion-ReTrac

This is the official implementation of "_Data Attribution for Diffusion Models: Timestep-induced Bias in Influence Estimation_" (TMLR 2024)

[[Paper]](https://arxiv.org/abs/2401.09031)  [[OpenReview]](https://openreview.net/forum?id=P3Lyun7CZs)

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#tldr">TL;DR</a>
    <li>
      <a href="#requirements">Requirements</a>
  <ol>
</details>




<a name="tldr"></a>
<!-- GETTING STARTED -->
### TL;DR

> Influence estimations for diffusion models can be highly dependent on training timesteps, introducing bias
and arbitrariness in attribution results. We identify the **dominating norm effect** where this bias causes top
influential samples to be the same across diverse test images (ie. generally influential). To address this,
we present diffusion-ReTrac with re-normalization technique to provide fair and targeted attribution.


<a name="requirements"></a>
### Requirements
