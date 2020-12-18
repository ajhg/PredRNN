for ANO in `seq 1979 1989`; do

wget ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis2/pressure/uwnd.$ANO.nc
wget ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis2/pressure/vwnd.$ANO.nc
wget ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis2/pressure/hgt.$ANO.nc
#wget ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis2/surface/pres.sfc.$ANO.nc
wget ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis2/pressure/rhum.$ANO.nc
wget ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis2/pressure/air.$ANO.nc

done
