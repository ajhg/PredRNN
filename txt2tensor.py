
import numpy as np

grid = np.loadtxt('fullgrid_7984_atl_s.txt')

#tsteps = int(np.shape(grid)[0]/np.shape(grid)[1]/3)
tsteps = 8768 # time steps
nex = tsteps - 20 # training examples, 21 samples each

size = 17 # number of lat/lon points # 6020
chnls = 3 # number of channels (meteo info)
#chnls = 4 # number of channels (meteo info)

hgt=np.zeros((tsteps,size,size)) # geopotencial
u=np.zeros((tsteps,size,size)) # zonal wind
v=np.zeros((tsteps,size,size)) # meridional wind
#vimfc=np.zeros((tsteps,size,size)) # vimfc

for t in range(tsteps): # time stepts
  for lat in range(size): # latitude
    hgt[t,(size-1)-lat,:]=grid[t*(chnls*size)+lat,:]
    u[t,(size-1)-lat,:]=grid[t*(chnls*size)+lat+(1*size),:]
    v[t,(size-1)-lat,:]=grid[t*(chnls*size)+lat+(2*size),:]
    #vimfc[t,(size-1)-lat,:]=grid[t*(chnls*size)+lat+(3*size),:]

full=np.zeros((tsteps,size,size,chnls)) # full dataset

for t in range(tsteps):
  full[t,:,:,0] = hgt[t,:,:]
  full[t,:,:,1] = u[t,:,:]
  full[t,:,:,2] = v[t,:,:]
  #full[t,:,:,3] = vimfc[t,:,:]

sliced=np.zeros((nex,21,size,size,chnls)) # train set with 21 time steps examples

for gt in range(nex): # global time
  for lt in range(21): # local (hop) time
    sliced[gt,lt,:,:,:] = full[gt + lt,:,:,:]

np.save('datasets/datatensor_7984_atl_s.npy', sliced)

