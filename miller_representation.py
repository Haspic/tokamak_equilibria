
import numpy as np


class miller:
    """
    See introduction / geometry:
    https://repository.gatech.edu/server/api/core/bitstreams/682350a2-7f91-4a71-9415-747b9181c685/content
    """

    def __init__(self,
                 elongation: float,
                 triangularity: float,
                 R0: float,
                 r: float):
        """
        :param elongation: Kappa / elongation parameter
        :param triangularity: Delta / triangularity parameter
        :param R0: Distance of the center point to the Z-axis
        :param r: Surface radius, parallel to R0 with respect to the R-axis
        """

        self.kappa = elongation
        self.delta = triangularity
        self.x = np.arcsin(self.delta)

        self.R0 = R0
        self.r = r

    def get_coordinates(self, theta: np.ndarray or float):
        """
        Returns a tuple of arrays/values of R, Z coordinates of all miller representation
        values, at the given angles.

        :param theta: angle / array of angles
        :return: ( R-coordinate(s) , Z-coordinate(s) )
        """

        R = self.R0 + self.r * np.cos(theta + self.x * np.sin(theta))
        Z = self.kappa * self.r * np.sin(theta)

        return R, Z
