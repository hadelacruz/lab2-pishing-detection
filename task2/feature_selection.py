import pandas as pd
import matplotlib.pyplot as plt


def seleccionar_features(df, n_features=2, output_dir='.'):


    # Separar features y target
    X = df.drop(columns=['status'])
    y = df['status']
    
    # Calcular correlación con la variable objetivo
    correlaciones = X.corrwith(y).abs().sort_values(ascending=False)
    
    # Seleccionar las 2 mejores features
    top_features = correlaciones.head(n_features).index.tolist()
    
    # Crear visualización de las correlaciones
    plt.figure(figsize=(12, 6))
    correlaciones.head(15).plot(kind='barh', color='steelblue')
    plt.xlabel('Correlación Absoluta con Status')
    plt.title('Top 15 Features por Correlación con Variable Objetivo')
    plt.tight_layout()
    filepath_corr = f'{output_dir}/correlaciones_features.png'
    plt.savefig(filepath_corr, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualización de las 2 mejores features
    if n_features <= 3:
        fig, axes = plt.subplots(1, n_features, figsize=(7*n_features, 5))
        if n_features == 1:
            axes = [axes]
        
        for idx, feature in enumerate(top_features):
            axes[idx].scatter(df[feature], df['status'], alpha=0.5, s=10)
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Status (0=Legit, 1=Phishing)')
            axes[idx].set_title(f'Relación: {feature} vs Status')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath_scatter = f'{output_dir}/top_features_scatter.png'
        plt.savefig(filepath_scatter, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Visualización 2D para frontera de decisión
    if n_features == 2:
        plt.figure(figsize=(10, 8))
        colors = ['blue' if status == 0 else 'red' for status in y]
        plt.scatter(df[top_features[0]], df[top_features[1]], 
                    c=colors, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        plt.xlabel(top_features[0])
        plt.ylabel(top_features[1])
        plt.title('Espacio de Features 2D para Frontera de Decisión')
        plt.legend(['Legitimate', 'Phishing'], loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath_2d = f'{output_dir}/feature_space_2d.png'
        plt.savefig(filepath_2d, dpi=300, bbox_inches='tight')
        plt.close()
    
    return top_features, X, y
    
