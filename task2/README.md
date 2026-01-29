# Task 2: Preparación de Datos - Detección de Phishing

## 🚀 Cómo ejecutar

```bash
cd task2
python main.py
```

## 📊 Qué hace

1. **Carga** el dataset y elimina columna 'url'
2. **Codifica** status: legitimate=0, phishing=1
3. **Selecciona** las 2 features con mayor correlación (`google_index`, `page_rank`)
4. **Escala** datos usando StandardScaler (implementado desde cero)
5. **Divide** en train (80%) y test (20%) con estratificación

## 📈 Salida

- **Datos procesados**: 87 features, 9,144 train, 2,286 test
- **Gráficos generados**:
  - `correlaciones_features.png` - Top 15 features
  - `top_features_scatter.png` - Scatter plots de mejores features
  - `feature_space_2d.png` - Espacio 2D para fronteras de decisión

## 📦 Estructura modular

- `data_loading.py` - Limpieza y codificación
- `feature_selection.py` - Selección por correlación
- `data_scaling.py` - StandardScaler manual
- `data_splitting.py` - Train/test split manual
- `main.py` - Pipeline completo

## ✅ Sin sklearn

Todo implementado desde cero con **pandas/numpy** para entender cómo funciona internamente.
