from data_loading import limpiar_datos
from feature_selection import seleccionar_features
from data_scaling import escalar_datos
from data_splitting import dividir_datos


def main():
    
    print("\n" + "="*70)
    print("PREPARACIÓN DE DATOS - DETECCIÓN DE PHISHING")
    print("="*70 + "\n")
    
    filepath = 'dataset_phishing.csv'
    
    # PASO 1: Cargar y limpiar datos 
    df_clean = limpiar_datos(filepath)
    
    # PASO 2: Seleccionar features
    top_2_features, X_all, y = seleccionar_features(df_clean, n_features=2)
    print(f"✅ Features seleccionadas: {top_2_features}")
    
    # PASO 3: Escalar datos
    X_scaled_all, scaler_all = escalar_datos(X_all, y, method='standard')
    
    # PASO 4: Dividir datos
    X_train_all, X_test_all, y_train, y_test = dividir_datos(X_scaled_all, y)
    
    # Preparar dataset 2D para visualización de fronteras
    print(f"\n{'='*70}")
    print("DATASET 2D PARA VISUALIZACIÓN DE FRONTERAS")
    print("="*70)
    
    X_2d = df_clean[top_2_features]
    X_2d_scaled, scaler_2d = escalar_datos(X_2d, y, method='standard')
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = dividir_datos(X_2d_scaled, y)
    
    print(f"✅ Dataset 2D: {X_train_2d.shape[1]} features, {X_train_2d.shape[0]} train, {X_test_2d.shape[0]} test")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print("="*70)
    print(f"📊 Total features: {X_train_all.shape[1]}")
    print(f"📊 Training samples: {X_train_all.shape[0]} (80%)")
    print(f"📊 Test samples: {X_test_all.shape[0]} (20%)")
    print(f"📊 Gráficos: correlaciones_features.png, top_features_scatter.png, feature_space_2d.png")
    
    # Retornar datos procesados
    return {
        'df_clean': df_clean,
        'X_train_all': X_train_all,
        'X_test_all': X_test_all,
        'X_train_2d': X_train_2d,
        'X_test_2d': X_test_2d,
        'y_train': y_train,
        'y_test': y_test,
        'scaler_all': scaler_all,
        'scaler_2d': scaler_2d,
        'top_2_features': top_2_features
    }


if __name__ == "__main__":
    datos_procesados = main()