
import pandas as pd
import numpy as np


def escalar_datos(X, y=None, method='standard'):

    if method == 'standard':
        # z = (x - μ) / σ
        media = X.mean()  
        std = X.std()      
        
        X_scaled = (X - media) / std
        
        # Guardar parámetros para poder escalar datos nuevos después
        scaler_params = {
            'method': 'standard',
            'mean': media,
            'std': std
        }
        
    else:  # minmax
        # MinMaxScaler manual: x_scaled = (x - min) / (max - min)
        x_min = X.min()
        x_max = X.max()
        
        X_scaled = (X - x_min) / (x_max - x_min)
        
        # Guardar parámetros
        scaler_params = {
            'method': 'minmax',
            'min': x_min,
            'max': x_max
        }
        
    return X_scaled, scaler_params
