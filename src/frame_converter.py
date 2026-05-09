import jax.numpy as jnp

EARTH_RAD = 6378137.0
EARTH_OMEGA = 7.2921150e-5


def rotate_z(rx, ry, rz, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    rx_new = c * rx - s * ry
    ry_new = s * rx + c * ry
    return rx_new, ry_new, rz


def earth_rotation_angle(t, t_ref=0.0):
    return EARTH_OMEGA * (t - t_ref)


def ecef_to_eci(rx_ecef, ry_ecef, rz_ecef, t, t_ref=0.0):
    return rotate_z(rx_ecef, ry_ecef, rz_ecef,
                    earth_rotation_angle(t, t_ref))


def eci_to_ecef(rx_eci, ry_eci, rz_eci, t, t_ref=0.0):
    return rotate_z(rx_eci, ry_eci, rz_eci,
                    -earth_rotation_angle(t, t_ref))


def geographic_to_cartesian(lat, lon, h):
    r = EARTH_RAD + h

    cos_lat = jnp.cos(lat)

    x = r * cos_lat * jnp.cos(lon)
    y = r * cos_lat * jnp.sin(lon)
    z = r * jnp.sin(lat)

    return x, y, z


def cartesian_to_geographic(x, y, z):
    r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)

    lat = jnp.arcsin(z / r)
    lon = jnp.arctan2(y, x)
    h = r - EARTH_RAD

    return lat, lon, h
