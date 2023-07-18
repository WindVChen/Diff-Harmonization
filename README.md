<div align="center">

<h1><a href="https://arxiv.org/abs/2307.08182">Zero-Shot Image Harmonization with Generative Model Prior</a></h1>

**[Jianqi Chen](https://windvchen.github.io/), [Zhengxia Zou](https://scholar.google.com.hk/citations?hl=en&user=DzwoyZsAAAAJ), [Yilan Zhang](https://scholar.google.com.hk/citations?hl=en&user=wZ4M4ecAAAAJ), [Keyan Chen](https://scholar.google.com.hk/citations?hl=en&user=5RF4ia8AAAAJ), and [Zhenwei Shi](https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ)**

![](https://komarev.com/ghpvc/?username=windvchenDiff-Harmonization&label=visitors)
![GitHub stars](https://badgen.net/github/stars/windvchen/Diff-Harmonization)
[![](https://img.shields.io/badge/license-Apache--2.0-blue)](#License)
[![](https://img.shields.io/badge/arXiv-2307.08182-b31b1b.svg)](https://arxiv.org/abs/2307.08182)

</div>

### Share us a :star: if this repo does help

This is the official repository of ***Diff-Harmonization***. We are working on further improvements to this method (see **Appendix D** of the paper) to provide a better user experience, so stay tuned for more updates.

If you encounter any question about the paper, please feel free to contact us. You can create an issue or just send email to me windvchen@gmail.com. Also welcome for any idea exchange and discussion.

BTW:
In the process of waiting for the final code, you may wish to pay attention to our [***INR-Harmonization***](https://github.com/WindVChen/INR-Harmonization) work that we recently released the final code. It is **the first dense pixel-to-pixel method applicable to high-resolution (*~6K*) images** without any hand-crafted filter design, based on *Implicit Neural Representation*,.

## Updates

[**07/18/2023**] Repository init.

## TODO
- [ ] Code release
- [ ] Gradio release

## Table of Contents

- [Abstract](#Abstract)
- [Results](#Results)
- [Citation & Acknowledgments](#Citation-&-Acknowledgments)
- [License](#License)


## Abstract

![DiffHarmon's framework](assets/network.png)

Recent image harmonization methods have demonstrated promising results. However, due to their heavy reliance on a large number of composite images, these works are expensive in the training phase and often fail to generalize to unseen images. In this paper, we draw lessons from **human behavior** and come up with a **zero-shot image harmonization** method. Specifically, in the harmonization process, a human mainly utilizes his long-term prior on harmonious images and makes a composite image close to that prior. To imitate that, we resort to pretrained generative models for the prior of natural images. For the guidance of the harmonization direction, we propose an Attention-Constraint Text which is optimized to well illustrate the image environments. Some further designs are introduced for preserving the foreground content structure. The resulting framework, highly consistent with human behavior, can achieve harmonious results without burdensome training. Extensive experiments have demonstrated the effectiveness of our approach, and we have also explored some interesting applications.

## Results

![Visual comparisons](assets/visualizations.png#pic_center)
![Visual comparisons2](assets/visualizations2.png#pic_center)
![Visual comparisons3](assets/visualizations3.png#pic_center)

## Citation & Acknowledgments
If you find this paper useful in your research, please consider citing:
```

```

## License
This project is licensed under the Apache-2.0 license. See [LICENSE](LICENSE) for details.