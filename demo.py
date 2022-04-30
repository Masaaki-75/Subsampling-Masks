# coding:utf-8
import os
import matplotlib.pyplot as plt
from matplotlib import rc
from subsampling import *
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use: rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def show_figures(masks, suptitle, titles, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    axes = axes.flatten()
    plt.suptitle(suptitle, size=18)
    for i, (mask, title) in enumerate(zip(masks, titles)):
        axes[i].imshow(mask, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(title, size=10)
        for d in ['top', 'right', 'bottom', 'left']:
            axes[i].spines[d].set_visible(False)
    fig.tight_layout()
    # plt.show(block=True)
    if save_path is not None:
        fig.savefig(save_path, dpi=100)


if __name__ == '__main__':
    size = (512, 512)
    save_dir = r'C:\Users\75169\Desktop\tmp\masks'
    masks_annular = [0] * 4
    masks_annular[0] = get_annular_mask(size, radii=(50, 100), center=None, refine=False,
                                        save_path=os.path.join(save_dir, 'annular_r50r100_c1_f0.png'))
    masks_annular[1] = get_annular_mask(size, radii=(50, 100), center=(50, 100), refine=False,
                                        save_path=os.path.join(save_dir, 'annular_r50r100_c0_f0.png'))
    masks_annular[2] = get_annular_mask(size, radii=100, center=None, eps=0.5, refine=True,
                                        save_path=os.path.join(save_dir, 'annular_r100_c1_f1.png'))
    masks_annular[3] = get_annular_mask(size, radii=100, center=None, eps=2, refine=False,
                                        save_path=os.path.join(save_dir, 'annular_r100_c1_f0.png'))
    titles_annular = [r'$r=[50,100], [x_c,y_c]=[256,256]$', r'$r=[50,100], [x_c,y_c]=[50,100]$',
                      r'$r=[100], [x_c,y_c]=[256,256]{\rm(refined)}$', r'$r=[100\pm 2], [x_c,y_c]=[256,256]$']

    masks_disk = [0] * 4
    masks_disk[0] = get_disk_mask(size, radius=100, center=None, save_path=os.path.join(save_dir, 'disk_r100_c1.png'))
    masks_disk[1] = get_disk_mask(size, radius=100, center=(50, 100), save_path=os.path.join(save_dir, 'disk_r100_c0.png'))
    masks_disk[2] = get_disk_mask(size, radius=50, center=None, save_path=os.path.join(save_dir, 'disk_r50_c1.png'))
    masks_disk[3] = get_disk_mask(size, radius=50, center=(50, 100), save_path=os.path.join(save_dir, 'disk_r50_c0.png'))
    titles_disk = [r'$r=100, [x_c,y_c]=[256,256]$', r'$r=100, [x_c,y_c]=[50,100]$',
                   r'$r=50, [x_c,y_c]=[256,256]$', r'$r=50, [x_c,y_c]=[50,100]$']

    masks_radial = [0] * 4
    masks_radial[0] = get_radial_mask(size, center=None, dtheta=5, is_degree=True,
                                      save_path=os.path.join(save_dir, 'radial_dt5_dr1_c1.png'))
    masks_radial[1] = get_radial_mask(size, center=None, dtheta=20, is_degree=True,
                                      save_path=os.path.join(save_dir, 'radial_dt20_dr1_c1.png'))
    masks_radial[2] = get_radial_mask(size, center=(50, 100), dtheta=5, is_degree=True,
                                      save_path=os.path.join(save_dir, 'radial_dt5_dr1_c0.png'))
    masks_radial[3] = get_radial_mask(size, center=None, dtheta=5, is_degree=True, dr=3,
                                      save_path=os.path.join(save_dir, 'radial_dt5_dr3_c1.png'))
    titles_radial = [r'$\Delta \theta=5^{\rm o}, \Delta r=1, [x_c,y_c]=[256,256]$',
                     r'$\Delta \theta=20^{\rm o}, \Delta r=1, [x_c,y_c]=[256,256]$',
                     r'$\Delta \theta=5^{\rm o}, \Delta r=1, [x_c,y_c]=[50,100]$',
                     r'$\Delta \theta=5^{\rm o}, \Delta r=3, [x_c,y_c]=[256,256]$']

    masks_ripple = [0] * 4
    masks_ripple[0] = get_ripple_mask(size, center=None, dr=10, eps=1, refine=True,
                                      save_path=os.path.join(save_dir, 'ripple_dr10_c1_f1_d0.png'))
    masks_ripple[1] = get_ripple_mask(size, center=None, dr=20, eps=2, refine=True,
                                      save_path=os.path.join(save_dir, 'ripple_dr20_c1_f1_d0.png'))
    masks_ripple[2] = get_ripple_mask(size, center=None, dr=10, eps=0.5, refine=False, density_function='center',
                                      save_path=os.path.join(save_dir, 'ripple_dr10_c1_f0_d1.png'))
    masks_ripple[3] = get_ripple_mask(size, center=None, dr=10, eps=0.5, refine=False, density_function='rim',
                                      save_path=os.path.join(save_dir, 'ripple_dr10_c1_f0_d2.png'))
    titles_ripple = [r'$\Delta r=10, [x_c,y_c]=[256,256] {\rm (equid)}$',
                     r'$\Delta r=20, [x_c,y_c]=[256,256] {\rm (equid)}$',
                     r'$\Delta r=10, [x_c,y_c]=[256,256] {\rm (center)}$',
                     r'$\Delta r=10, [x_c,y_c]=[256,256] {\rm (rim)}$']

    masks_spiral = [0] * 4
    masks_spiral[0] = get_spiral_mask(size, center=None, a=0, b=3, refine=True, dtheta=0.1, is_degree=True,
                                      save_path=os.path.join(save_dir, 'spiral_a0b3_c1_f1.png'))
    masks_spiral[1] = get_spiral_mask(size, center=(100, 200), a=0, b=3, refine=True, dtheta=0.1, is_degree=True,
                                      save_path=os.path.join(save_dir, 'spiral_a0b3_c0_f1.png'))
    masks_spiral[2] = get_spiral_mask(size, center=None, a=0, b=10, refine=True, dtheta=0.1, is_degree=True,
                                      save_path=os.path.join(save_dir, 'spiral_a0b10_c1_f1.png'))
    masks_spiral[3] = get_spiral_mask(size, center=None, a=50, b=3, refine=True, dtheta=0.1, is_degree=True,
                                      save_path=os.path.join(save_dir, 'spiral_a50b3_c1_f1.png'))
    titles_spiral = [r'$\Delta \theta=0.1^{\rm o}, a=0, b=3, [x_c,y_c]=[256,256]$',
                     r'$\Delta \theta=0.1^{\rm o}, a=0, b=3, [x_c,y_c]=[100,200]$',
                     r'$\Delta \theta=0.1^{\rm o}, a=0, b=10, [x_c,y_c]=[50,100]$',
                     r'$\Delta \theta=0.1^{\rm o}, a=50, b=3, [x_c,y_c]=[256,256]$']

    show_figures(masks_annular, suptitle='Annular Masks', titles=titles_annular, save_path=os.path.join(save_dir, 'masks_annular.png'))
    show_figures(masks_disk, suptitle='Disk Masks', titles=titles_disk, save_path=os.path.join(save_dir, 'masks_disk.png'))
    show_figures(masks_radial, suptitle='Radial Masks', titles=titles_radial, save_path=os.path.join(save_dir, 'masks_radial.png'))
    show_figures(masks_ripple, suptitle='Ripple Masks', titles=titles_ripple, save_path=os.path.join(save_dir, 'masks_ripple.png'))
    show_figures(masks_spiral, suptitle='Spiral Masks', titles=titles_spiral, save_path=os.path.join(save_dir, 'masks_spiral.png'))
