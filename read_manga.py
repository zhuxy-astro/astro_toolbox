# %%
# ## Importing packages
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import astropy.units as u


# %%
# ## plotmap function
# plotmap函数，快速画彩色map
# map_data就是进来的图，如果想只显示好数据，麻烦调用的时候套一层selectdata函数
# title和几个label不再赘述，记得字符串前带r
# setcmap是表示值大小的颜色表，默认彩虹
# setmin和setmax是设定的最大最小值。如果map_data里面有超出此值的，就只画set范围内的图，超出的格点画作color map的顶点颜色
#
def plotmap(
    map_data,
    title,
    xlabel=r"$\Delta x$ (arcsec)",
    ylabel=r"$\Delta y$ (arcsec)",
    barlabel="",
    setcmap=plt.cm.jet,
    setmin=None,
    setmax=None,
    filename=None,
):

    figg = plt.figure()
    axx = figg.add_subplot()

    # 如果实际的数据在设置的setmin和setmax范围内，那就用实际的范围作图；如果实际数据超出了set的范围，那就只画set范围内的图
    if (
        setmin and np.nanmin(map_data) > setmin
    ):  # 如果设置了setmin，而且实际的min比setmin要大，那么就用实际的min替代setmin
        setmin = np.nanmin(map_data)
    if setmax and np.nanmax(map_data) < setmax:
        setmax = np.nanmax(map_data)

    im = axx.imshow(
        map_data, vmin=setmin, vmax=setmax, cmap=setcmap  # 其中vmin和vmax天生是None
    )

    cb = figg.colorbar(im, ax=axx)
    cb.ax.tick_params(direction="in", length=5, width=1.0)
    cb.set_label(barlabel)
    axx.set_title(title)
    axx.set_xlabel(xlabel)
    axx.set_ylabel(ylabel)


# %% [markdown] tags=[]
# # Read Data

# %%
directory = "/Users/ZhuXY/Projects/Data/SDSS/MaNGA/"

# %%
dapallf = fits.open(directory + "dapall-mpl-6.fits")
dapall = dapallf[1].data
# dapallf.close()
# 这是manga的一个总的文件

# %% tags=[]
dapallf[1].header

# %%
data = fits.open(
    "/Users/ZhuXY/Documents/astronomy/courses/General Astronomy/Experiment/20181216/m42_110.fits"
)

# %%
mapdata = data[0].data

# %%
plotmap(np.log10(mapdata), "", setcmap=plt.cm.gray)

# %% [markdown] tags=[]
# # MaNGA data

# %% [markdown]
# ## Read data

# %% cell_style="center"
# for galaxyi in range (5):

# 下面两句是通过名字来匹配一个源
# plateifu='7977-9101'
# galaxyi=np.where(dapall['plateifu']==plateifu)[0][0]

galaxyi = 4

print(galaxyi)

dapalli = dapall[galaxyi]
# 从很多个DAP里面抽取我要的这一个，后面基本上都用的这一个
plate = str(dapalli["plate"])
ifudesign = str(dapalli["IFUDESIGN"])
plateifu = plate + "-" + ifudesign
# 把名字先取出来，后面好用

# 打开两个源文件
dir_1 = directory + "MPL6-MAPS/" + plate + "/" + ifudesign + "/"
tar_dir = "manga" + "-" + plateifu + "-MAPS-SPX-GAU-MILESHC.fits.gz"
cube = fits.open(dir_1 + tar_dir)

dir_ssp = directory + "mpl6-pipe3d/" + plate + "/"
tar_ssp = "manga" + "-" + plateifu + ".Pipe3D.cube.fits.gz"
cube_ssp = fits.open(dir_ssp + tar_ssp)

# %% tags=[]
cube_ssp[1].header  # pipe3d

# %% tags=[]
cube[0].header

# %% cell_style="center"
ssp_stellar = cube_ssp[1].data
# 这是一个大集合，里面能取出很多pipe3d的东西
# ######stellar mass density dust corrected  (log10) in units of  m_Sun/spaxels^2 ############
stellar_sur = ssp_stellar[19]

# ######ha velocity in units of km/s ############
v_ha = cube[36].data[18]  # Line-of-sight emission line velocity
v_ha_mask = cube[38].data[
    18
]  # Data quality mask for Gaussian-fitted velocity measurements

# %% [markdown] tags=[]
# ## Select data

# %%
snr = cube[5].data
# 直接读取的：Mean g-band weighted signal-to-noise ratio per pixel
indplot = (v_ha_mask == 0) & (snr > 3)
# 初步筛选：用一个bool标记所有数据里没说不能用的、Ha信噪比大于3的点8
# snr本来是5，但是数据可能比较少，改到3差不多


# %%
def selectdata(readdata, use=indplot, useless=np.nan):
    # 按某原则挑选值，默认按照indplot
    # 返回的是：不好的点都变成了nan（或自定义的useless），好的点保留原先值
    # use是有用的index，useless是没用的那些点填上啥
    if type(useless) is int:
        map_data = np.full(readdata.shape, float(useless))
        # 如果不用float，那么如果useless为整数，则会把扔进来的data都变成整数……佛了
    else:
        map_data = np.full(readdata.shape, useless)
    # 先用useless把新数组填满。如果用full_like，在readdata为bool数组的时候会出问题。
    map_data[use] = readdata[use]  # 然后把好数据塞进去
    return map_data


# %% [markdown] tags=[]
# ## z, distance and arcsec per spaxel

# %%
arcsec_per_spaxel = 0.5

z = dapalli["Z"]
print("z =", z)

Hubble = 0.69
# Hubble is in 100 km/s/Mpc
d_ang = dapalli["ADIST_Z"] / Hubble
d_lum = dapalli["LDIST_Z"] / Hubble
# d_lum and d_ang are in Mpc

d_lum_cm = d_lum * u.Mpc.to("cm")

spaxel = arcsec_per_spaxel * u.arcsec
kpc_per_spaxel = (spaxel.to("rad").value) * d_ang * 1e3
# 每个像素对应0.5 arcsec，把它们转换成长度的kpc和面积的pc^2，后面好用

# %% [markdown] tags=[]
# ## Axis, radius, PA and inclination

# %%
# 坐标轴，arcsec
x0 = 0
y0 = 0
# 中心点的漂移，没有认真检查后续代码，会让后续变得一团糟，慎用
# x0&y, xpos, ypos are in arcsec

nx = (
    np.arange(stellar_sur.shape[0]) - stellar_sur.shape[0] / 2
) * arcsec_per_spaxel + x0
ny = (
    np.arange(stellar_sur.shape[1]) - stellar_sur.shape[1] / 2
) * arcsec_per_spaxel + y0
xpos, ypos = np.meshgrid(nx, ny, sparse=False, indexing="xy")
# 用ste_v的形状来定义本星系使用的所有map的形状
# nx和ny分别是，横纵坐标中各个格点距离星系中心的一维数列，以arcsec为单位（不完全是赤经赤纬）
# 其中的arcsec_per_spaxel用到了MaNGA数据中每个像素点的大小是0.5 arcsec
# xpos和ypos分别是把nx和ny弄成二维的map，每个格点上是本位置的横或纵坐标

# %%
# 倾角inc，弧度制
ell = cube[0].header["ECOOELL"]
# 这个代表1-b/a
alpha = 0.2
# alpha代表厚度，经验地取0.2
sini = np.sqrt(ell * (2 - ell) / (1 - alpha**2))
inc = np.arcsin(sini)
sin2i = sini**2
cosi = np.cos(inc)
cos2i = 1 - sin2i
# plotmap(1-roty**2/(radius**2-rotx**2),'sin2i')
# 检查sin^2i是否均匀
# plotmap(np.sqrt(rotx**2+roty**2/(1-sini**2))/radius,'',setmin=0.995)
# radius和用x、y计算得的结果比较，检查倾角，用radius推得的结果，两者在~2E-3精度上一致
print("inclination =", inc / np.pi * 180)

# %%
# 主轴角PA，弧度制
PAbase = 0
# PA继续逆时针转多少度
PA = (dapalli["NSA_ELPETRO_PHI"] + PAbase) / 180 * np.pi
# 从数据中读出的角度改为弧度
print("PA =", PA / np.pi * 180)
cosPA = np.cos(PA)
sinPA = np.sin(PA)
rotx = -sinPA * xpos + cosPA * ypos
roty = -sinPA * ypos - cosPA * xpos
# 依照PA定义一个旋转后的坐标map，rotx沿主轴方向
# 两个rot坐标是天空平面的坐标而非盘面的

# %%
# radius，单位arcsec
# 代表各个点与星系中心的盘面距离
# 如果刚才没有手动定义坐标原点和PA的偏移，就用原数据
if (np.abs(x0) < 0.1) & (np.abs(y0) < 0.1) & (np.abs(PAbase) < 0.1):
    radius = cube[2].data[0]
else:
    radius = np.sqrt(rotx**2 + roty**2 / cos2i)
# 由于是盘面距离，而rotx和roty是天空平面，所以上式有一个/cos2i
maxr = np.nanmax(selectdata(radius))
# indplot里面可用数据的最大的radius是多少
radius_in_kpc = radius / arcsec_per_spaxel * kpc_per_spaxel
radius_in_m = radius / arcsec_per_spaxel * kpc_per_spaxel * u.kpc.to("m")

# %%
Reff = dapall["NSA_ELPETRO_TH50_R"][4]
Reff_in_kpc = Reff / arcsec_per_spaxel * kpc_per_spaxel
print("Reff =", Reff_in_kpc, "kpc")
# 读取Reff并做单位换算

# %%
# 盘面平面上的theta，从主轴开始记角度制0-360

theta = cube[2].data[2] - np.arcsin(np.sin(PAbase / 180 * np.pi) * cosi) / np.pi * 180
# 如果有PA的手动调整的话就搞一下

angle_from_PA = 90 - np.abs(np.abs(theta - 180) - 90)
# 各点偏离主轴的盘面角度的绝对值

cos_theta = np.cos(theta / 180 * np.pi)
sin_theta = np.sin(theta / 180 * np.pi)
# 注意此处sin和cos是从theta来的，有正负

# %%
cube.close()
cube_ssp.close()
