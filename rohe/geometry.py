from scipy.spatial import ConvexHull
#### Geometry helpers

def midpoint(point_a, point_b, offset=(0, 0)):
    """Find the midpoint of the line defined by `point_a` and `point_b`"""
    return (
        (point_a[0] + point_b[0]) / 2 + offset[0],
        (point_a[1] + point_b[1]) / 2 + offset[1]
    )


def points_circumference(point, radius, n=10):
    """Generate `n` points on `radius` of `point`"""
    return [
        (
            point[0] + numpy.cos(2 * numpy.pi / n * x) * radius,
            point[1] + numpy.sin(2 * numpy.pi / n * x) * radius,
        )
        for x in range(n + 1)
    ]


def buffer_convex_hull(hull: ConvexHull, stretch: float, n: int) -> ConvexHull:
    """Expand the boundaries of a convex hull `hull` and round the borders.
    `stretch` : how far from the original hull vertices to place points
    `n` : how many vertices on the new hull to generate. This controls the
    amount of roundedness at the corners of the polygon.
    """
    new_points = []
    for point in zip(
            hull.points[hull.vertices, 0],
            hull.points[hull.vertices, 1]
    ):
        new_points.extend(points_circumference(point, stretch, n))

    return ConvexHull(new_points)


def build_isogloss(points, padding=.1, roundedness=40):
    # Need 3 points for a convex hull...
    if len(points) == 1:
        points.extend(points_circumference(points[0], .001, 1))
        points.append(midpoint(points[0], points[1], offset=(.001, .001)))
        
    if len(points) == 2:
        points.append(midpoint(points[0], points[1], offset=(.001, .001)))
        
    if len(points) > 2:  
        hull = ConvexHull(points)
        # After computing the complex hull, we expand it and round the edges
        buff_hull = buffer_convex_hull(hull, padding, roundedness)
        isogloss = []

        for simplex in buff_hull.simplices:
            isogloss.append(
                (buff_hull.points[simplex, 0], buff_hull.points[simplex, 1])
            )
        return isogloss
