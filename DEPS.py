import xlwings as xw
import numpy as np
from scipy.interpolate import CubicSpline

def filter_nan_values(x_col, y_col):
    """
    Filters out NaN values from the input arrays.
    
    Parameters:
    x_col (array-like): Array of x values
    y_col (array-like): Array of y values
    
    Returns:
    tuple: Filtered x_col and y_col arrays
    """
    mask = ~np.isnan(x_col) & ~np.isnan(y_col)
    return x_col[mask], y_col[mask]

@xw.func
@xw.arg('x_col', np.array)
@xw.arg('y_col', np.array)
def interpolate_for_x(x_col, y_col, target_y):
    """
    This function interpolates x based on given y values using cubic spline interpolation.
    
    Parameters:
    x_col (array-like): Array of x values from Excel
    y_col (array-like): Array of y values from Excel
    target_y (float): The y value for which we want to interpolate target_x
    
    Returns:
    float: Interpolated x value
    """
    if len(x_col) != len(y_col):
        raise ValueError("x and y columns must have the same length")
    
    # Filter out NaN values
    x_col, y_col = filter_nan_values(x_col, y_col)
    
    if len(x_col) < 2 or len(y_col) < 2:
        raise ValueError("Not enough data points after filtering NaN values")
    
    # Create a cubic spline interpolation function
    spline = CubicSpline(y_col, x_col)
    
    # Interpolate the target x value for the given target y
    target_x = float(spline(target_y))
    
    return target_x

@xw.func
@xw.arg('x_col', np.array)
@xw.arg('y_col', np.array)
def interpolate_for_y(x_col, y_col, target_x):
    """
    This function interpolates y based on given x values using cubic spline interpolation.
    
    Parameters:
    x_col (array-like): Array of x values from Excel
    y_col (array-like): Array of y values from Excel
    target_x (float): The x value for which we want to interpolate y
    
    Returns:
    float: Interpolated y value
    """
    if len(x_col) != len(y_col):
        raise ValueError("x and y columns must have the same length")
    
    # Filter out NaN values
    x_col, y_col = filter_nan_values(x_col, y_col)
    
    if len(x_col) < 2 or len(y_col) < 2:
        raise ValueError("Not enough data points after filtering NaN values")
    
    # Create a cubic spline interpolation function
    spline = CubicSpline(x_col, y_col)
    
    # Interpolate the target y value for the given target x
    target_y = float(spline(target_x))
    
    return target_y
