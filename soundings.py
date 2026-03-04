import numpy as np
import pandas as pd
import xarray as xr

import thermo as thrm


def parse_data_portal_sounding(sounding_csv):
    """
    Read single sounding CSV into a Pandas dataframe and calculate
    derived properties (potential temperature, specific humidity,
    dew point, wind components).
    """
    df = pd.read_csv(sounding_csv, parse_dates=['timestamp'], index_col=['timestamp'])

    df['temperature'] += thrm.T0
    df['exner'] = thrm.exner(df['pressure'])
    df['theta'] = df['temperature'] / df['exner']

    es = thrm.esat(df['temperature'])
    e = df['relative_humidity'] / 100 * es

    df['qt'] = e * 0.622 / df['pressure']
    df['Td'] = thrm.dewpoint(df['qt'], df['pressure'])

    wind_dir_rad = np.deg2rad(df['heading'])
    df['u'] = df['speed'] * np.sin(wind_dir_rad)
    df['v'] = df['speed'] * np.cos(wind_dir_rad)

    return df


def load_sounding_stations(min_end_year=2025):
    """
    Load active IGRA2 radiosonde stations as an xarray Dataset.

    Parameters:
    ----------
    min_end_year : int
        Only include stations with data up to at least this year.

    Returns:
    -------
    xr.Dataset
        Dataset with dimension 'station' and variables: name, code, lat, lon, elev.
    """

    cols = ['id', 'lat', 'lon', 'elev', 'name', 'start_year', 'end_year', 'nobs']
    df = pd.read_fwf(
        'resources/igra2-station-list.txt',
        header=None,
        names=cols,
        colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (38, 71), (72, 76), (77, 81), (82, 88)],
    )

    active = df[df['end_year'] >= min_end_year].copy().reset_index(drop=True)
    active['code'] = active['id'].str[-5:]

    return xr.Dataset(
        {
            'name': ('station', active['name'].values),
            'code': ('station', active['code'].values),
            'lat':  ('station', active['lat'].values.astype(float)),
            'lon':  ('station', active['lon'].values.astype(float)),
            'elev': ('station', active['elev'].values.astype(float)),
        },
        coords={'station': active.index.values},
    )


def get_nearest_soundings(ds, lat, lon, n=5):
    """
    Return the n stations in ds nearest to the given lat/lon.

    Uses an equirectangular approximation.

    Parameters:
    ----------
    ds : xr.Dataset
        Dataset returned by load_sounding_stations.
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    n : int
        Number of nearest stations to return.

    Returns:
    -------
    xr.Dataset
        Slice of ds with the n nearest stations, sorted by distance.
    """
    dlat = ds['lat'].values - lat
    dlon = (ds['lon'].values - lon) * np.cos(np.radians(lat))
    dist = np.hypot(dlat, dlon)
    idxs = np.argsort(dist)[:n]

    return ds.isel(station=idxs)