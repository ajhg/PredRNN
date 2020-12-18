
for ANO in `seq 1979 1984`; do
cdo sellonlatbox,-60,-20,-60,-20 raw_reans/hgt.$ANO.nc hgt_box.$ANO.nc
cdo sellonlatbox,-60,-20,-60,-20 raw_reans/uwnd.$ANO.nc uwnd_box.$ANO.nc
cdo sellonlatbox,-60,-20,-60,-20 raw_reans/vwnd.$ANO.nc vwnd_box.$ANO.nc
done

for ANO in `seq 1979 1984`; do
cdo sellevel,500 hgt_box.$ANO.nc hgt500_box.$ANO.nc
cdo sellevel,1000 uwnd_box.$ANO.nc uwnd1000_box.$ANO.nc
cdo sellevel,1000 vwnd_box.$ANO.nc vwnd1000_box.$ANO.nc
done

cdo copy hgt500_box.1979.nc hgt500_box.1980.nc hgt500_box.1981.nc hgt500_box.1982.nc hgt500_box.1983.nc hgt500_box.1984.nc hgt_7984.nc
cdo copy uwnd1000_box.1979.nc uwnd1000_box.1980.nc uwnd1000_box.1981.nc uwnd1000_box.1982.nc uwnd1000_box.1983.nc uwnd1000_box.1984.nc uwnd_7984.nc
cdo copy vwnd1000_box.1979.nc vwnd1000_box.1980.nc vwnd1000_box.1981.nc vwnd1000_box.1982.nc vwnd1000_box.1983.nc vwnd1000_box.1984.nc vwnd_7984.nc

cdo merge hgt_7984.nc uwnd_7984.nc vwnd_7984.nc full_grid_7984.nc

cdo outputf,%8.2f,17 full_grid_7984.nc > fullgrid7984_atl_s.txt

rm *box* *_7984.nc
