import numpy as np

def random_sphere(N, radius, center=None):
    """
    Generates N points randomly distributed on a sphere

    Parameters
    ----------
    N: int
    Number of points to generate

    radius: float
    Radius of the sphere

    center: numyp array, optional
    center of the sphere. (0,0,0) default

    Returns
    -------
    Array (3, N) with the points

    """

    u = 2*np.random.random(N)-1
    theta = 2*np.pi*np.random.random(N)

    points = np.array((radius*np.sqrt(1-u**2)*np.cos(theta),
                       radius*np.sqrt(1-u**2)*np.sin(theta), radius*u))

    if center is not None:
        c = np.repeat(center, N)
        c = np.reshape(c, (3, N))
        points += c

    return points


def random_ball(N, radius, center=None):
    """
    Generates N points randomly distributed on a ball
    x^2+y^2+z^y <= 1

    Parameters
    ----------
    N: int
    Number of points to generate

    radius: float
    Radius of the sphere

    Returns
    -------
    Array (3, N) with the points

    """

    r = np.random.random(N)
    x = np.random.normal(0, 1, (3, N))
    norm = np.linalg.norm(x, axis=0)

    points = radius * np.power(r, 1./3.) * x/norm

    if center is not None:
        c = np.repeat(center, N)
        c = np.reshape(c, (3, N))
        points += c

    return points


def random_cube(N, size, center=None):
    """
    Generates N points randomly distributed on cube

    Parameters
    ----------
    N: int
    Number of points to generate

    size: float
    Size of the side of the cube

    Returns
    -------
    Array (3, N) with the points

    """

    x = size*np.random.random((3, N)) - 0.5*size
    face = np.random.randint(0, 3, N)
    side = 2*np.random.randint(0, 2, N)-1
    x[face, np.arange(0, N)] = (0.5*size)*side

    if center is not None:
        c = np.repeat(center, N)
        c = np.reshape(c, (3, N))
        x += c

    return x