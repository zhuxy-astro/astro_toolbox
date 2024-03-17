import numpy as np

"""Example:
```
    from astro_toolbox import bpt

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, wspace=0)
    ax_nii, ax_sii, ax_oi = gs.subplots(sharey=True)

    ax_nii.set_ylim(*bpt.oiii_lim)
    ax_nii.set_ylabel(bpt.oiii_label)

    ax_nii.set_xlim(*bpt.nii_lim)
    ax_nii.set_xlabel(bpt.nii_label)
    ax_sii.set_xlim(*bpt.sii_lim)
    ax_sii.set_xlabel(bpt.sii_label)
    ax_oi.set_xlim(*bpt.oi_lim)
    ax_oi.set_xlabel(bpt.oi_label)

    for ax, x in zip([ax_nii, ax_sii, ax_oi], [niiha, siiha, oiha]):
        select = select_z
        ax.scatter(x[select], oiiihb[select],
                   s=1, c='silver', rasterized=True)

    ax_nii.plot(*bpt.nii_sf_comp_line_points(), 'k', linewidth=2)
    ax_nii.plot(*bpt.nii_agn_comp_line_points(), 'k--', linewidth=2)
    ax_sii.plot(*bpt.sii_sf_agn_line_points(), 'k', linewidth=2)
    ax_sii.plot(*bpt.sii_liner_seyfert_line_points(), 'k--', linewidth=2)
    ax_oi.plot(*bpt.oi_sf_agn_line_points(), 'k', linewidth=2)
    ax_oi.plot(*bpt.oi_liner_seyfert_line_points(), 'k--', linewidth=2)

    fig.show()
    fig.savefig('figures/bpt-cen.pdf')
```
"""


# %% init
def _line_calculator(k, x0, y0):
    """returns a hyperbolic function of the form k / (x - x0) + y0
    """
    def return_func(x, k=k, x0=x0, y0=y0):
        return k / (x - x0) + y0
    return return_func


def _straight_line_calculator(k, b):
    """returns a straight line function of the form k * x + b
    """
    def return_func(x, k=k, b=b):
        return k * x + b
    return return_func


def _line_points_generator(func, x_left, x_right, n_points=50):
    """usage:
        # in calc.py
        nii_agn_comp_line_points = _line_points_generator(nii_agn_comp, -2, 0.3, 50)
        # in other files
        plt.plot(*bpt.nii_agn_comp_line_points(n_points=200))
    """
    def return_func(func=func, x_left=x_left, x_right=x_right, n_points=n_points):
        xs = np.linspace(x_left, x_right, n_points)
        ys = func(xs)
        return xs, ys
    return return_func


# %% oiii as y axis
oiii_lim = (-1.2, 1.5)
oiii_label = r'$\log (\mathrm{[O_{III}]/H\beta})$'


# %% nii
nii_lim = (-2., 1.)
nii_label = r'$\log (\mathrm{[N_{II}]/H\alpha})$'

nii_agn_comp = _line_calculator(0.61, 0.47, 1.19)
nii_sf_comp = _line_calculator(0.61, 0.05, 1.3)

nii_agn_comp_line_points = _line_points_generator(nii_agn_comp, -2, 0.3)
nii_sf_comp_line_points = _line_points_generator(nii_sf_comp, -2, -0.15)


def nii(nii_ha, oiii_hb):
    """
    return: array of ints. 0: sf, 1: composite, 2: agn, nan: nan.
    """
    result = np.empty_like(nii_ha)
    result[:] = np.nan

    sf_comp = nii_sf_comp(nii_ha)
    agn_comp = nii_agn_comp(nii_ha)

    select_sf = oiii_hb < sf_comp
    result[select_sf] = 0
    select_agn = oiii_hb > agn_comp
    result[select_agn] = 2
    select_comp = oiii_hb >= sf_comp
    select_comp &= oiii_hb <= agn_comp
    result[select_comp] = 1
    return result


# %% sii
sii_lim = (-1.2, 0.8)
sii_label = r'$\log (\mathrm{[S_{II}]/H\alpha})$'

sii_sf_agn = _line_calculator(0.72, 0.32, 1.30)
sii_liner_seyfert = _straight_line_calculator(1.89, 0.76)

sii_sf_agn_line_points = _line_points_generator(sii_sf_agn, -1.2, 0.1)
sii_liner_seyfert_line_points = _line_points_generator(sii_liner_seyfert, -0.3, 0.1)


def sii(sii_ha, oiii_hb):
    """
    return: array of ints. 0: sf, 1: liner, 2: seyfert, nan: nan.
    """
    result = np.empty_like(sii_ha)
    result[:] = np.nan

    sf_agn = sii_sf_agn(sii_ha)
    liner_seyfert = sii_liner_seyfert(sii_ha)

    select_sf = oiii_hb < sf_agn
    result[select_sf] = 0
    select_seyfert = oiii_hb >= sf_agn
    select_seyfert &= oiii_hb >= liner_seyfert
    result[select_seyfert] = 2
    select_liner = oiii_hb >= sf_agn
    select_liner &= oiii_hb < liner_seyfert
    result[select_liner] = 1
    return result


# %% oi
oi_lim = (-2.2, 0)
oi_label = r'$\log (\mathrm{[O_{I}]/H\alpha})$'

oi_sf_agn = _line_calculator(0.73, -0.59, 1.33)
oi_liner_seyfert = _straight_line_calculator(1.18, 1.30)

oi_sf_agn_line_points = _line_points_generator(oi_sf_agn, -2.2, -0.8)
oi_liner_seyfert_line_points = _line_points_generator(oi_liner_seyfert, -1.1, -0.5)


def oi(oi_ha, oiii_hb):
    """
    return: array of ints. 0: sf, 1: liner, 2: seyfert, nan: nan.
    """
    result = np.empty_like(oi_ha)
    result[:] = np.nan

    sf_agn = oi_sf_agn(oi_ha)
    liner_seyfert = oi_liner_seyfert(oi_ha)

    select_sf = oiii_hb < sf_agn
    result[select_sf] = 0
    select_seyfert = oiii_hb >= sf_agn
    select_seyfert &= oiii_hb >= liner_seyfert
    result[select_seyfert] = 2
    select_liner = oiii_hb >= sf_agn
    select_liner &= oiii_hb < liner_seyfert
    result[select_liner] = 1
    return result
