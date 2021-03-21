# Seeing in Extra Darkness Using a Deep-red Flash
This is the PyTorch code for 2021 CVPR paper "Seeing in Extra Darkness Using a Deep-red Flash"
### [Project Page](https://vccimaging.org/Publications/Xiong2021Seeing/) | [Video]() | [Paper](https://vccimaging.org/Publications/Xiong2021Seeing/Xiong2021Seeing.pdf)

[Seeing in Extra Darkness Using a Deep-red Flash](https://vccimaging.org/Publications/Xiong2021Seeing/Xiong2021Seeing.pdf)  
 [Jinhui Xiong](https://jhxiong.github.io/)<sup>1</sup>\*,
 [Jian Wang](https://jianwang-cmu.github.io/)<sup>2</sup>\*,
 [Wolfgang Heidrich](https://vccimaging.org/People/heidriw/bio)<sup>1</sup>,
 [Shree Nayar](http://www.cs.columbia.edu/~nayar/)<sup>2</sup> <br>
 <sup>1</sup>KAUST, <sup>2</sup>Snap Research
  \*denotes equal contribution  
CVPR 2021 (Oral)

<img src='img/teaser.jpg'>
Top left: Human vision uses cones and rods for the perception of light. Photopic vision is associated with cones, occurring at bright-light conditions (over 3 cd/m^2). Scotopic vision is associated with rods, occurring at dim-light conditions (below 10^{-3} cd/m^2). At intermediate light levels, both rods and cones are active, which is called mesopic vision. <br>
Bottom left: We propose to use deep-red (e.g. 660 nm) light as flash for low-light imaging in mesopic light levels. This new flash can be introduced into smartphones with a minor hardware adjustment. <br>
Middle: The eye spectral sensitivity in a dimly lit environment (0.01 cd/m^2) and the relative responses of R, G and B color channels of the camera we used, as well as the emissions spectrum of the red LED. Under dim lighting, rod vision dominates, yet the rods are nearly insensible to deep-red light. Meanwhile, our LED flash can be sensed by the camera especially in the red and green channels. <br>
Right: Inputs to our videography pipeline are a sequence of no-flash and flash frames, and the outputs are denoised and would yield temporally stable videos with no frame rate loss.

## Image Filtering
To test image filtering on our data, we prepared a notebook
```
evaluate.ipynb
```
in the [image_filtering](https://github.com/vccimaging/Deep-Red-Flash/tree/main/image_filtering) folder.

## Video Filtering
To test video filtering, you need to first install PWC-Net(https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) for flow computation and Temporal Consistency Network(https://github.com/phoenix104104/fast_blind_video_consistency) for enhancing tempotal consistency.

After installation, you need to correctly import them (sample code is commented) and run
```
python video_filtering.py
```
