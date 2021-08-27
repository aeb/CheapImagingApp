import ngeht_array
import data

import matplotlib.pyplot as plt
import numpy as np


# statdict = ngeht_array.read_array('./arrays/eht2022_230_ehtim.txt')
statdict = ngeht_array.read_array('./arrays/ngeht_ref1_230_ehtim.txt')

data.generate_data_from_file('./source_images/toy_story_aliens.png',statdict)
#data.generate_data_from_file('./source_images/GRRT_IMAGE_data1400_freq230.npy',statdict)

quit()

# data_orig = data.read_data('./data/V_M87_eht2022_230_perfect_scanavg_tygtd.dat')
# data_orig = data.read_data('./data/V_M87_ngeht_ref1_230_perfect_scanavg_tygtd.dat')
# ra = 12 + 30/60. + 49.42338/3600.
# dec = 12 + 23/60. + 28.0439/3600.
 
# data_orig = data.read_data('./data/V_SGRA_eht2022_230_perfect_scanavg_tygtd.dat')
data_orig = data.read_data('./data/V_SGRA_ngeht_ref1_230_perfect_scanavg_tygtd.dat')
ra = 17 + 45/60. + 40.0409/3600.
dec = - (29 + 0/60. + 28.118/3600.)



################################################
################################################
### Read in images, uses ehtim temporarily while we migrate to image format
print("================== Reading image =========================")
import ehtim as eh
# img = eh.image.load_fits("/Users/abroderick/Downloads/ngEHT_Challenge_1/models/M87/GRRT_IMAGE_data1400_freq230.fits")
img = eh.image.load_fits("/Users/abroderick/Downloads/ngEHT_Challenge_1/models/SGRA/fromm230_scat.fits")
x,y = np.meshgrid(np.arange(0,img.xdim),np.arange(0,img.ydim),indexing='xy')
# I = img.ivec.reshape([img.xdim,img.ydim])
I = np.flipud(img.ivec.reshape([img.xdim,img.ydim]))
x = -(x-0.5*(img.xdim-1))*img.psize * 180.*3600e6/np.pi # dRA
y = (y-0.5*(img.ydim-1))*img.psize * 180.*3600e6/np.pi  # dDec


print("Sum/integral:",np.sum(I[:]),np.sum(I[:])*(x[1,1]-x[0,0])*(y[1,1]-y[0,0]))

print("================== Plotting image ========================")
plt.figure()
xc = x[1792:2304,1792:2304]
yc = y[1792:2304,1792:2304]
Ic = I[1792:2304,1792:2304]
       
plt.pcolor(xc,yc,np.log10(Ic),vmax=np.max(np.log10(Ic)),vmin=np.max(np.log10(Ic))-3)
plt.xlim((xc[0,0],xc[-1,-1]))
# plt.show()

################################################
################################################

print("================== Generating data =======================")

data_new = data.generate_data(230,ra,dec,x,y,I,statdict,day=90,min_elev=10,scan_time=1200,bandwidth=8)


import pickle
with open("datagen.pkl","wb") as f :
    pickle.dump(data_new, f)


print("================== Plotting data =========================")

########### u-v baseline map
plt.figure()
plt.plot(data_orig['u'],data_orig['v'],'.b',alpha=1)
plt.plot(data_new['u'],data_new['v'],'.r',alpha=0.5,ms=3)
plt.xlim((10,-10))
plt.ylim((-10,10))


########### t vs u for AA-LM
plt.figure()
keepo = (data_orig['s2']=='AA')*(data_orig['s1']=='LM')
keepn = (data_new['s2']=='AA')*(data_new['s1']=='LM')
plt.plot(data_orig['t'][keepo],data_orig['u'][keepo],'.b',alpha=1)
plt.plot(data_new['t'][keepn],data_new['u'][keepn],'.r',alpha=0.5,ms=3)

########### |u| vs V with errors
plt.figure()
plt.errorbar(np.sqrt(data_orig['u']**2+data_orig['v']**2),np.abs(data_orig['V']),yerr=data_orig['err'],fmt='.b',alpha=0.5)
plt.errorbar(np.sqrt(data_new['u']**2+data_new['v']**2),np.abs(data_new['V']),yerr=data_new['err'],fmt='.r',alpha=0.25)
plt.yscale('log')
plt.ylim((1e-4,3))

########### |u| vs real(V) with errors
plt.figure()
plt.errorbar(np.sqrt(data_orig['u']**2+data_orig['v']**2),np.real(data_orig['V']),yerr=data_orig['err'],fmt='.b',alpha=0.5)
plt.errorbar(np.sqrt(data_new['u']**2+data_new['v']**2),np.real(data_new['V']),yerr=data_new['err'],fmt='.r',alpha=0.25)

########### |u| vs imag(V) with errors
plt.figure()
plt.errorbar(np.sqrt(data_orig['u']**2+data_orig['v']**2),np.imag(data_orig['V']),yerr=data_orig['err'],fmt='.b',alpha=0.5)
plt.errorbar(np.sqrt(data_new['u']**2+data_new['v']**2),np.imag(data_new['V']),yerr=data_new['err'],fmt='.r',alpha=0.25)

########### |u| vs errors (original data has bw=8 GHz)
plt.figure()
plt.plot(np.sqrt(data_orig['u']**2+data_orig['v']**2),data_orig['err'],'.b',alpha=0.5)
plt.plot(np.sqrt(data_new['u']**2+data_new['v']**2),data_new['err'],'.r',alpha=0.25)
plt.yscale('log')
plt.ylim((1e-4,3))



plt.show()
