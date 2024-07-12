import numpy as np


def project_point_on_plane(point, plane_point, plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    point_to_plane = plane_point - point
    distance = np.dot(point_to_plane, plane_normal)
    return point + distance * plane_normal, distance


def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))