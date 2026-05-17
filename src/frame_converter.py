import jax.numpy as jnp
import casadi as ca
import numpy as np

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


def oe2rv(oe, mu):
    # Convert orbital elements to position and velocity vectors.
    a, e, i, Om, om, nu = oe
    p = a * (1 - e * e)
    r = p / (1 + e * np.cos(nu))

    # Position in perifocal frame
    rv_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])

    # Velocity in perifocal frame
    vv_pf = np.array([-np.sin(nu), e + np.cos(nu), 0.0]) * np.sqrt(mu / p)

    # Rotation matrix from perifocal to inertial frame
    cO, sO = np.cos(Om), np.sin(Om)
    co, so = np.cos(om), np.sin(om)
    ci, si = np.cos(i), np.sin(i)

    R = np.array(
        [
            [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
            [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
            [so * si, co * si, ci],
        ]
    )

    ri = R @ rv_pf
    vi = R @ vv_pf

    return ri, vi


def _cross_product(a, b):
    # Cross product of two 3D vectors
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
    )


def _dot_product(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _smooth_heaviside(x, a_eps=0.1):
    return 0.5 * (1 + ca.tanh(x / a_eps))


def rv2oe(rv, vv, mu):
    # Convert position and velocity to orbital elements with smooth transitions
    eps = 1e-12

    K = ca.vertcat(0.0, 0.0, 1.0)
    hv = _cross_product(rv, vv)
    nv = _cross_product(K, hv)

    n = ca.sqrt(ca.fmax(_dot_product(nv, nv), eps))
    h2 = ca.fmax(_dot_product(hv, hv), eps)
    v2 = ca.fmax(_dot_product(vv, vv), eps)
    r = ca.sqrt(ca.fmax(_dot_product(rv, rv), eps))

    rv_dot_vv = _dot_product(rv, vv)

    ev = ca.vertcat(
        (1 / mu) * ((v2 - mu / r) * rv[0] - rv_dot_vv * vv[0]),
        (1 / mu) * ((v2 - mu / r) * rv[1] - rv_dot_vv * vv[1]),
        (1 / mu) * ((v2 - mu / r) * rv[2] - rv_dot_vv * vv[2]),
    )

    p = h2 / mu
    e = ca.sqrt(ca.fmax(_dot_product(ev, ev), eps))
    a = p / (1 - e * e)
    i = ca.acos(ca.fmax(ca.fmin(hv[2] / ca.sqrt(h2), 1.0 - eps), -1.0 + eps))

    # Smooth transitions for angular elements
    a_eps = 0.1
    nv_dot_ev = _dot_product(nv, ev)

    Om = _smooth_heaviside(nv[1] + eps, a_eps) * ca.acos(
        ca.fmax(ca.fmin(nv[0] / n, 1.0 - eps), -1.0 + eps)
    ) + _smooth_heaviside(-(nv[1] + eps), a_eps) * (
                 2 * np.pi - ca.acos(ca.fmax(ca.fmin(nv[0] / n, 1.0 - eps), -1.0 + eps))
         )

    om = _smooth_heaviside(ev[2], a_eps) * ca.acos(
        ca.fmax(ca.fmin(nv_dot_ev / (n * e), 1.0 - eps), -1.0 + eps)
    ) + _smooth_heaviside(-ev[2], a_eps) * (
                 2 * np.pi - ca.acos(ca.fmax(ca.fmin(nv_dot_ev / (n * e), 1.0 - eps), -1.0 + eps))
         )

    ev_dot_rv = _dot_product(ev, rv)
    nu = _smooth_heaviside(rv_dot_vv, a_eps) * ca.acos(
        ca.fmax(ca.fmin(ev_dot_rv / (e * r), 1.0 - eps), -1.0 + eps)
    ) + _smooth_heaviside(-rv_dot_vv, a_eps) * (
                 2 * np.pi - ca.acos(ca.fmax(ca.fmin(ev_dot_rv / (e * r), 1.0 - eps), -1.0 + eps))
         )

    return ca.vertcat(a, e, i, Om, om, nu)
