from typing import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


class KMeans:
    def __init__(self, k: int):
        """
        The k-means algorithm in Python.
        """
        self.centroid_coords = dict()
        self.num_clusters = k

    def fit(self, X: np.array) -> dict:
        """
        Assumes k is a positive integer.

        Returns: dict: contains the dict and covariance of each centroid
        """
        # A: init k random centroids from existing data
        rng = np.random.default_rng()
        centroids = rng.choice(X, size=(self.num_clusters,))
        centroids_assigned_pts = dict(
            zip(
                # scalars mapped to 2D arrays
                range(self.num_clusters),
                [[] for _ in range(self.num_clusters)],
            )
        )
        # B: converge on optimal centroids
        keep_going = True

        while keep_going is True:
            # 1: assign each point to a centroid
            for sample in X:
                centroid_assignment = np.argmin(
                    [np.linalg.norm(sample - centroids, axis=1)]  # Euclidean distance
                )
                centroids_assigned_pts[centroid_assignment].append(sample)
            # 2: update centroid placements themselves
            cap = centroids_assigned_pts  # just an abbreviation
            new_centroids = np.array(
                [
                    np.mean(np.array(cap[centroid_label]), axis=0)
                    for centroid_label in centroids_assigned_pts.keys()
                ]
            )
            # 3: decide if we should continue
            if np.equal(centroids, new_centroids).all():
                keep_going = False
            centroids = new_centroids[:]

        # C: collect the results
        for centroid_label in centroids_assigned_pts:
            self.centroid_coords[centroid_label] = centroids[centroid_label, :]
        return self.centroid_coords

    def predict(self, X: np.array):
        centroids = [coords for coords in self.centroid_coords.values()]
        centroid_assignment = [
            np.argmin(
                [np.linalg.norm(sample - centroids, axis=1)]  # Euclidean distance
            )
            for sample in X
        ]

        return centroid_assignment

    @classmethod
    def fit_and_visualize(
        cls: "KMeans",
        points: np.ndarray,
        axis_labels: OrderedDict[str, str],
        num_clusters: int = 10,  # as per Hw 4 description
        plot_title: str = "",
    ) -> None:
        """TODO[Zain]: add docstring"""
        # find the clusters in the dataset
        kmeans = cls(k=num_clusters)
        kmeans.fit(points)

        # visualize the points - TODO, in the future we can generalize this to work for 2D also
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        ax.scatter(xs, ys, zs, color="g")

        # visualize the cluster centers
        centroids = np.array(list(kmeans.centroid_coords.values()))
        xs = centroids[:, 0]
        ys = centroids[:, 1]
        zs = centroids[:, 2]
        ax.scatter(xs, ys, zs, color="r")

        ax.set_xlabel(axis_labels["x"])
        ax.set_ylabel(axis_labels["y"])
        ax.set_zlabel(axis_labels["z"])

        plt.title(plot_title)
