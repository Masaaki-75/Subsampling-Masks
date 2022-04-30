# Subsampling-Masks
Python implementation of different types of masks for image subsampling.

## :star: If you find this helpful, click a star! :star: ##

# 1. Display
The following pictures show different subsampling masks generated by the functions in `subsampling.py`, including annular mask, disk-shaped mask, radial mask, ripple-like mask and Archimedes spiral mask. Each of them represents a certain subsampling pattern in medical image reconstruction and can be obtained by running `demo.py` (just remember to set your local path correctly to save the pictures).

For detailed information, please see the comments in `subsampling.py`.

<img src="https://github.com/Masaaki-75/Subsampling-Masks/blob/main/figs/masks_annular.png">
<img src="https://github.com/Masaaki-75/Subsampling-Masks/blob/main/figs/masks_disk.png">
<img src="https://github.com/Masaaki-75/Subsampling-Masks/blob/main/figs/masks_radial.png">
<img src="https://github.com/Masaaki-75/Subsampling-Masks/blob/main/figs/masks_ripple.png">
<img src="https://github.com/Masaaki-75/Subsampling-Masks/blob/main/figs/masks_spiral.png">

# 2. Dependencies
- python==3.6.5<br>
- skimage==0.19.2<br>
- numpy==1.21.6


# 3. TODO
- Code optimization
- Implementations of other subsampling masks generation (e.g. poisson disk, uniform, ...)
