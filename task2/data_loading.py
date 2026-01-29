import pandas as pd


def limpiar_datos(filepath):
    
    # Cargar dataset
    df = pd.read_csv(filepath)
    
    # Eliminar columna 'url'
    df_clean = df.drop(columns=['url'])
    
    # Codificar: legitimate=0, phishing=1
    df_clean['status'] = df_clean['status'].map({
        'legitimate': 0,
        'phishing': 1
    })
        
    return df_clean
