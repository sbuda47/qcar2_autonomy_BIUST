# qcar2_autonomy/platform/ports.py

BASE_HIL_PORT = 18960
BASE_VIDEO_PORT = 18961
BASE_GPS_PORT = 18967
BASE_LIDAR_IDEAL_PORT = 18968
BASE_VIDEO3D_PORT = 18965


def resolve_virtual_port(explicit_port, base_port, qcar_id, stride):
    if explicit_port is not None:
        return int(explicit_port)
    return int(base_port + int(qcar_id) * int(stride))