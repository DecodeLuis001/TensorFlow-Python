import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Importando Datos
house_df = pd.read_csv("precios_hogares.csv")

# VISUALIZACION
sns.scatterplot(x='sqft_living', y='price', data=house_df)

# correlacion
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(house_df.corr(), annot=True)

# LIMPIEZA DE DATOS
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']
X = house_df[selected_features]
y = house_df['price']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = y.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)

# ENTRENAMIENTO
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25)

# Definiendo modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=100, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

model.summary()
model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2)

# Evaluando Modelo
epochs_hist.history.keys()

# Grafico
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso del Modelo durante Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])

# Prediccion
X_test_1 = np.array([[4, 3, 1960, 5000, 1, 2000, 3000]])
X_test_scaled_1 = scaler.transform(X_test_1)
y_predict_1 = model.predict(X_test_scaled_1)
y_predict_1 = scaler.inverse_transform(y_predict_1)
