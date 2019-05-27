import rasterio
from rasterio.windows import Window
import math


class GDALDataset(object):

    def __init__(self, path):
        self.data_source = rasterio.open(path)
        self.band_count = self.data_source.count
        self.row_count = self.data_source.height
        self.column_count = self.data_source.width
        self.epsg = self.data_source.crs.to_epsg()

    def generate_bounds(self, size, buffer, batch_size, extent=None):
        size_row = size[0]
        size_column = size[1]
        buffer_row = math.floor(buffer[0] / math.fabs(self.data_source.transform[4]))
        buffer_column = math.floor(buffer[1] / self.data_source.transform[0])

        assert buffer_row < size_row
        assert buffer_column < size_column

        if extent is None:
            row_min = 0
            column_min = 0
            row_max = self.row_count
            column_max = self.column_count
        else:
            x_min = extent[0]
            y_min = extent[1]
            x_max = extent[2]
            y_max = extent[3]

            row_min, column_min = self.data_source.index(x_min, y_max)
            row_max, column_max = self.data_source.index(x_max, y_min)

            if row_min < 0:
                row_min = 0

            if column_min < 0:
                column_min = 0

            if row_max > self.row_count:
                row_max = self.row_count

            if column_max > self.column_count:
                column_max = self.column_count

        all_bounds = []
        bounds = []

        for row in range(row_min, row_max, size_row - buffer_row):
            for column in range(column_min, column_max, size_column - buffer_column):
                _size_row = size_row if row + size_row <= self.row_count else self.row_count - row
                _size_column = size_column if column + size_column <= self.column_count else self.column_count - column

                if len(bounds) < batch_size:
                    bounds.append((row, column, _size_row, _size_column))

                if len(bounds) == batch_size:
                    all_bounds.append(bounds)
                    bounds = []

        if len(bounds) != 0:
            all_bounds.append(bounds)

        return all_bounds, (size_row, size_column, buffer_row, buffer_column)

    def get_data(self, bound):
        row = bound[0]
        column = bound[1]
        size_row = bound[2]
        size_column = bound[3]

        data = self.data_source.read(window=Window(column, row, size_column, size_row))
        data = data.transpose(1, 2, 0)

        return data

    def get_transform(self, row, column):
        x_top_left, y_top_left = self.data_source.xy(row, column)
        x_top_left -= 0.15 / 2
        y_top_left += 0.15 / 2

        return rasterio.transform.from_origin(x_top_left,
                                              y_top_left,
                                              math.fabs(self.data_source.transform[0]),
                                              math.fabs(self.data_source.transform[4]))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data_source.close()
