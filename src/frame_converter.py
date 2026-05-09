import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A ** 2 - WGS84_B ** 2) / WGS84_B ** 2

EARTH_MU = 3.986004418e14
EARTH_OMEGA = 7.2921150e-5
G0 = 9.80665


def geodetic_to_ecef(lat, lon, h):
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + h) * sin_lat
    return np.array([x, y, z])


def ecef_to_geodetic(x, y, z):
    p = np.sqrt(x * x + y * y)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + WGS84_EP2 * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    h = p / np.cos(lat) - N
    return lat, lon, h


def earth_rotation_angle(t, t_ref=0.0):
    return EARTH_OMEGA * (t - t_ref)


def ecef_to_eci(r_ecef, t, t_ref=0.0):
    theta = earth_rotation_angle(t, t_ref)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s, c, 0.0],
                  [0., 0., 1.]])
    return R @ np.asarray(r_ecef)


def eci_to_ecef(r_eci, t, t_ref=0.0):
    theta = earth_rotation_angle(t, t_ref)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s, 0.0],
                  [-s, c, 0.0],
                  [0., 0., 1.]])
    return R @ np.asarray(r_eci)
