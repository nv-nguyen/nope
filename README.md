## NOPE: Novel Object Pose Estimation from a Single Image <br><sub>Official PyTorch implementation </sub>

![Teaser image](./media/framework.png)

**NOPE: Novel Object Pose Estimation from a Single Image**<br>
[Van Nguyen Nguyen](https://nv-nguyen.github.io/)
, [Thibault Groueix](http://imagine.enpc.fr/~groueixt/)
, [Yinlin Hu](https://yinlinhu.github.io/),
[Mathieu Salzmann](https://people.epfl.ch/mathieu.salzmann), 
[Vincent Lepetit](https://vincentlepetit.github.io/) <br>
**[Paper](https://arxiv.org/pdf/2303.13612.pdf)
, [Project Page](https://nv-nguyen.github.io/nope/)**

Abstract: *The practicality of 3D object pose estimation remains limited for many applications due to the need for prior knowledge of a 3D model and a training period for new objects. To address this limitation, we propose an approach that takes a single image of a new object as input and predicts the relative pose of this object in new images without prior knowledge of the object’s 3D model and without requiring training time for new objects and categories. We achieve this by training a model to directly predict discriminative embeddings for viewpoints surrounding the object. This prediction is done using a simple U-Net architecture with attention and conditioned on the desired pose, which yields extremely fast inference. We compare our approach to state-of-the-art methods and show it outperforms them both in terms of accuracy and robustness.*


<p align="center">
  <img src=./media/result.gif width="50%"/>
</p>

<font size="-1">\* We use the canonical pose of the 3D model to visualize this distribution, but not as input to our method.</font>

<font size="-1">\+ We visualize the predicted pose by rendering the object from this pose, but the 3D model is only used for visualization purposes, not as input to our method.</font>





## <sub>We will make the code available soon. </sub>


## Citation

```latex
@article{nguyen2023nope,
title={NOPE: Novel Object Pose Estimation from a Single Image},
author={Nguyen, Van Nguyen and Groueix, Thibault and Hu, Yinlin and Salzmann, Mathieu and Lepetit, Vincent},
journal={arXiv preprint arXiv:2303.13612},
year={2023}}
```