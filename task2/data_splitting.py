

import pandas as pd
import numpy as np


def dividir_datos(X, y, test_size=0.2, random_state=42):
  
    np.random.seed(random_state)
    
    # Obtener índices por clase para estratificación
    indices_clase_0 = y[y == 0].index.tolist()
    indices_clase_1 = y[y == 1].index.tolist()
    
    # Mezclar aleatoriamente cada clase
    np.random.shuffle(indices_clase_0)
    np.random.shuffle(indices_clase_1)
    
    # Calcular cuántos samples de test por clase
    n_test_clase_0 = int(len(indices_clase_0) * test_size)
    n_test_clase_1 = int(len(indices_clase_1) * test_size)
    
    # Dividir índices en train y test por clase
    test_indices = indices_clase_0[:n_test_clase_0] + indices_clase_1[:n_test_clase_1]
    train_indices = indices_clase_0[n_test_clase_0:] + indices_clase_1[n_test_clase_1:]
    
    # Crear conjuntos de train y test
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    
    print(f"✅ Datos divididos: {len(X_train)} train ({(1-test_size)*100:.0f}%), {len(X_test)} test ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test