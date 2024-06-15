# StereoNeRF<br><sub>Official PyTorch Implementation</sub>

**Generalizable Novel-View Synthesis using a Stereo Camera**<br>
Haechan Lee*, Wonjoon Jin*, Seung-Hwan Baek, Sunghyun Cho<br>

## News
- [x] Our paper is accepted to CVPR 2024!
- [x] Check out our [Project page](https://jinwonjoon.github.io/stereonerf/)!
- [x] The StereoNVS dataset and the BlendedMVS-Stereo dataset are released. Check out our [Google Drive Link](https://drive.google.com/drive/folders/1PI-_ESKw8fX_2YMD2v5DLR3FizikYxHO?usp=sharing)!
- [x] Code id released!

## Abstact
In this paper, we propose the first generalizable view synthesis approach that specifically targets multi-view stereo-camera images. Since recent stereo matching has demonstrated accurate geometry prediction, we introduce stereo matching into novel-view synthesis for high-quality geometry reconstruction. To this end, this paper proposes a novel framework, dubbed StereoNeRF, which integrates stereo matching into a NeRF-based generalizable view synthesis approach. StereoNeRF is equipped with three key components to effectively exploit stereo matching in novel-view synthesis: a stereo feature extractor, a depth-guided plane-sweeping, and a stereo depth loss. Moreover, we propose the StereoNVS dataset, the first multi-view dataset of stereo-camera images, encompassing a wide variety of both real and synthetic scenes. Our experimental results demonstrate that StereoNeRF surpasses previous approaches in generalizable view synthesis.

## Installation & Dependency
Tested with Docker image [(pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel)](https://hub.docker.com/layers/pytorch/pytorch/1.9.1-cuda11.1-cudnn8-devel/images/sha256-fd8fcd6e1196d8965657b04e7dfb666046063904b767c1fd75df8039fe0ada17?context=explore).

To install python library dependencies, run the code below:

```
pip install -r requirements.txt
```

If you want to reproduce the results of StereoNeRF:
- Adjust data paths in the config files located in the 'configs' folder.
- Download the pretrained [UniMatch](https://github.com/autonomousvision/unimatch/tree/master) weights to the 'pretrained_weights' folder.
- Prepare disparity maps of StereoNVS-Real predicted by UniMatch.

You can download the pretrained weights and rendered StereoNVS-Real images of StereoNeRF from our [Google Drive Link](https://drive.google.com/drive/folders/1fqcYa5a4sR4JEi5ulyChX3hibv5rgmcr).

(Our code is heavily based on [GeoNeRF](https://github.com/idiap/GeoNeRF)).

## Train
To train the model, run:

```
python run_stereo_nerf.py --config configs/config_general_stereo_real.txt
```

## Inference
Set the `ckpt_file` path properly at line 667 in 'run_stereo_nerf.py', then run:

```
python run_stereo_nerf.py --config configs/config_general_stereo_real.txt --eval
```

## Metric
To calculate the scores, run:

```
python calc_score.py --pred /path/to/evaluation/folder/in/prediction
```

## License of StereoNVS Dataset
- StereoNVS-Real: [CC BY-NC 3.0 DEED](https://creativecommons.org/licenses/by-nc/3.0/deed.en) (Non-commercial use)
- BlendedMVS-stereo: [CC BY-NC 3.0 DEED](https://creativecommons.org/licenses/by-nc/3.0/deed.en) (Non-commercial use)

### Contact
If you want to contact the author, email to: gocks8@gmail.com