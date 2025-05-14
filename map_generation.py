from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class ElevationMapGenerator:
    def __init__(self, gk2a_path, srtm_path, dim=512, x_add=75, y_add=115):
        self.gk2a_path = gk2a_path
        self.srtm_path = srtm_path
        self.dim = dim
        self.x_add = x_add
        self.y_add = y_add
        self._load_gk2a()

    def _load_gk2a(self):
        gk2a = nc.Dataset(self.gk2a_path)
        base = (900 - self.dim) // 2
        self.lat = gk2a['lat'][base + self.y_add: -base + self.y_add,
                               base + self.x_add: -base + self.x_add]
        self.lon = gk2a['lon'][base + self.y_add: -base + self.y_add,
                               base + self.x_add: -base + self.x_add]

    def generate(self, return_mask=False, visualize=False):
        srtm = nc.Dataset(self.srtm_path)
        srtm_lat = srtm['lat'][:]
        srtm_lon = srtm['lon'][:]
        srtm_elev = srtm['SRTMGL1_DEM'][0]

   
        lat_margin = lon_margin = 0.1
        lat_min, lat_max = self.lat.min(), self.lat.max()
        lon_min, lon_max = self.lon.min(), self.lon.max()

        lat_mask = (srtm_lat >= lat_min - lat_margin) & (srtm_lat <= lat_max + lat_margin)
        lon_mask = (srtm_lon >= lon_min - lon_margin) & (srtm_lon <= lon_max + lon_margin)

        lat_sub = srtm_lat[lat_mask]
        lon_sub = srtm_lon[lon_mask]
        elev_crop = srtm_elev[np.ix_(lat_mask, lon_mask)].astype(np.float32)

        interp_func = RegularGridInterpolator(
            (lat_sub, lon_sub),
            elev_crop,
            method='nearest',
            bounds_error=False,
            fill_value=np.nan
        )

      
        target_points = np.array([self.lat.flatten(), self.lon.flatten()]).T
        elevation_values = interp_func(target_points)

       
        if isinstance(elevation_values, np.ma.MaskedArray):
            elevation_values = elevation_values.filled(np.nan)

        elevation_map = elevation_values.reshape(self.lat.shape).astype(np.float32)

        elevation_mask = ~np.isnan(elevation_map)
        elevation_input = np.nan_to_num(elevation_map, nan=0.0)

        
        if visualize:
            self._visualize(elevation_input)

        if return_mask:
            return elevation_input, elevation_mask
        else:
            return elevation_input

    def _visualize(self, elevation_input):
        plt.figure(figsize=(5, 4))
        im = plt.imshow(elevation_input, cmap='viridis', vmin=0, vmax=2000)
        cbar = plt.colorbar(im, label='Elevation')
        cbar.locator = ticker.MultipleLocator(500)
        cbar.update_ticks()
        plt.title('Elevation')
        plt.axis('off')
        plt.tight_layout()
        plt.show()





class WaterMapGenerator:
    def __init__(self, gk2a_path, modis_path, dim=512, x_add=75, y_add=115):
        self.gk2a_path = gk2a_path
        self.modis_path = modis_path
        self.dim = dim
        self.x_add = x_add
        self.y_add = y_add
        self._load_gk2a()

    def _load_gk2a(self):
        gk2a = nc.Dataset(self.gk2a_path)
        base = (900 - self.dim) // 2
        self.lat = gk2a['lat'][base + self.y_add: -base + self.y_add,
                               base + self.x_add: -base + self.x_add]
        self.lon = gk2a['lon'][base + self.y_add: -base + self.y_add,
                               base + self.x_add: -base + self.x_add]

    def generate(self, apply_filter=True):
        modis = nc.Dataset(self.modis_path)
        modis_lat = modis['lat'][:]
        modis_lon = modis['lon'][:]
        lc_prop2 = modis['LC_Prop2'][0]
        water_mask = (lc_prop2 == 3).astype(np.float32)

        lat_indexer = RegularGridInterpolator((modis_lat,), np.arange(len(modis_lat)), bounds_error=False, fill_value=np.nan)
        lon_indexer = RegularGridInterpolator((modis_lon,), np.arange(len(modis_lon)), bounds_error=False, fill_value=np.nan)

        lat_idx_raw = lat_indexer(self.lat.flatten())
        lon_idx_raw = lon_indexer(self.lon.flatten())

       
        invalid_mask_flat = np.isnan(lat_idx_raw) | np.isnan(lon_idx_raw)

       
        lat_idx = np.clip(np.nan_to_num(lat_idx_raw, nan=0.0), 0, len(modis_lat) - 1)
        lon_idx = np.clip(np.nan_to_num(lon_idx_raw, nan=0.0), 0, len(modis_lon) - 1)

        coords = np.array([lat_idx, lon_idx])
        interpolated = map_coordinates(water_mask, coords, order=1, mode='mirror').reshape(self.lat.shape).astype(np.float32)

        
        invalid_mask = invalid_mask_flat.reshape(self.lat.shape)
        interpolated[invalid_mask] = 1.0

        x_start, x_end = 450, 512
        y_start, y_end = 0, 230
        interpolated[y_start:y_end, x_start:x_end] = 1.0

        if apply_filter:
            interpolated = gaussian_filter(interpolated, sigma=0.5)

        return interpolated.astype(np.float32)





class VegetationMapGenerator:
    def __init__(self, gk2a_path, modis_path, dim=512, x_add=75, y_add=115):
        self.gk2a_path = gk2a_path
        self.modis_path = modis_path
        self.dim = dim
        self.x_add = x_add
        self.y_add = y_add
        self._load_gk2a()

    def _load_gk2a(self):
        gk2a = nc.Dataset(self.gk2a_path)
        base = (900 - self.dim) // 2
        self.lat = gk2a['lat'][base + self.y_add: -base + self.y_add,
                               base + self.x_add: -base + self.x_add]
        self.lon = gk2a['lon'][base + self.y_add: -base + self.y_add,
                               base + self.x_add: -base + self.x_add]

    def generate(self, apply_filter=True):
        modis = nc.Dataset(self.modis_path)
        modis_lat = modis['lat'][:]
        modis_lon = modis['lon'][:]
        lc_prop2 = modis['LC_Prop2'][0]

        class_to_value = {10: 0.95, 36: 0.5, 40: 0.35}
        veg_map = np.zeros_like(lc_prop2, dtype=np.float32)
        for cls, val in class_to_value.items():
            veg_map[lc_prop2 == cls] = val

        interp_func = RegularGridInterpolator(
            (modis_lat, modis_lon), veg_map, method='linear',
            bounds_error=False, fill_value=0.0
        )
        target_points = np.array([self.lat.flatten(), self.lon.flatten()]).T
        interpolated = interp_func(target_points).reshape(self.lat.shape)

        if apply_filter:
            interpolated = gaussian_filter(interpolated, sigma=0.5)

        return interpolated.astype(np.float32)





import time

if __name__ == "__main__":
    gk2a_path = '/home/work/js/repo/frost-forecasting/map_generating/source/gk2a_ami_ko020lc_latlon.nc'
    srtm_path = '/home/work/js/repo/frost-forecasting/map_generating/source/SRTMGL1_NC.003_30m_aid0001.nc'
    modis_path = '/home/work/js/repo/frost-forecasting/map_generating/source/MCD12Q1.061_500m_aid0001.nc'

    # Elevation Map
    start = time.time()
    print("generating elevation map...")
    elev_gen = ElevationMapGenerator(gk2a_path, srtm_path)
    elev_map = elev_gen.generate()
    if isinstance(elev_map, np.ma.MaskedArray):
        elev_map = elev_map.filled(np.nan)  
    np.save("elevation_map_new.npy", elev_map)
    print(f"--elevation map done! ({time.time() - start:.2f} sec)")

    # Water Map
    start = time.time()
    print("generating water map...")
    water_gen = WaterMapGenerator(gk2a_path, modis_path)
    water_map = water_gen.generate()
    np.save("water_map_new.npy", water_map)
    print(f"--water map done! ({time.time() - start:.2f} sec)")

    # Vegetation Map
    start = time.time()
    print("generating vegetation map...")
    veg_gen = VegetationMapGenerator(gk2a_path, modis_path)
    veg_map = veg_gen.generate()
    np.save("vegetation_map_new.npy", veg_map)
    print(f"--vegetation map done! ({time.time() - start:.2f} sec)")
