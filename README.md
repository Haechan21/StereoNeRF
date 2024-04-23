# StereoNeRF<br><sub>Official PyTorch Implementation</sub>

**Generalizable Novel-View Synthesis using a Stereo Camera**<br>
Haechan Lee*, Wonjoon Jin*, Seung-Hwan Baek, Sunghyun Cho<br>

## News
- [x] Our paper is accepted to CVPR 2024!
- [x] Check out our [Project page](https://jinwonjoon.github.io/stereonerf/)!
- [x] The StereoNVS dataset and the BlendedMVS-Stereo dataset are released. Check out our [Google Drive Link](https://drive.google.com/drive/folders/1PI-_ESKw8fX_2YMD2v5DLR3FizikYxHO?usp=sharing)!
- [ ] Code will be released soon.

## Abstact
In this paper, we propose the first generalizable view synthesis approach that specifically targets multi-view stereo-camera images. Since recent stereo matching has demonstrated accurate geometry prediction, we introduce stereo matching into novel-view synthesis for high-quality geometry reconstruction. To this end, this paper proposes a novel framework, dubbed StereoNeRF, which integrates stereo matching into a NeRF-based generalizable view synthesis approach. StereoNeRF is equipped with three key components to effectively exploit stereo matching in novel-view synthesis: a stereo feature extractor, a depth-guided plane-sweeping, and a stereo depth loss. Moreover, we propose the StereoNVS dataset, the first multi-view dataset of stereo-camera images, encompassing a wide variety of both real and synthetic scenes. Our experimental results demonstrate that StereoNeRF surpasses previous approaches in generalizable view synthesis.
