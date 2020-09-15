from scipy.signal import savgol_filter

class TimeSeries:
    def __init__(self, X, y):
        self.X = X
        self.y = y


def smooth_line(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points

# def smooth_line(points):
#     return savgol_filter(points, 51, 3)