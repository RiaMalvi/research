import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# Load data from Excel file
df = pd.read_csv('WsnData.csv')
print("Raw Sensor Data:")
print(df)

numeric_cols = ['X', 'Y', 'SensorData', 'BatteryLife', 'Temperature']
fused_data = df[numeric_cols].mean(axis=1)

# Individual Sensor Data Plots
plt.figure(figsize=(15, 10))
colors = ['r', 'g', 'b', 'c', 'm']  # list of colors
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 2, i + 1)
    plt.plot(df[col], color=colors[i], label=col)
    plt.title(f'{col} Data', fontsize=10)
    plt.xlabel('Time Step', fontsize=9)
    plt.ylabel(f'Sensor Reading ({col} Unit)', fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8)
    plt.gca().spines['top'].set_visible(False)  # remove top border
    plt.gca().spines['right'].set_visible(False)  # remove right border
    plt.gca().spines['left'].set_linewidth(1)  # set left border width
    plt.gca().spines['bottom'].set_linewidth(1)  # set bottom border width
plt.tight_layout()
plt.show()

# Fused data calculation
fused_data = df[numeric_cols].mean(axis=1)

# Plot fused data
plt.figure(figsize=(15, 6))
plt.plot(fused_data, color='green', linewidth=1)
plt.title('Fused Sensor Data', fontsize=10)
plt.xlabel('Time Step', fontsize=9)
plt.ylabel('Fused Reading (Sensor Unit)', fontsize=9)
plt.xticks(np.arange(0, len(fused_data), max(1, len(fused_data) // 10)), rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.xlim(0, len(fused_data))
plt.ylim(fused_data.min() - 0.1, fused_data.max() + 0.1)
plt.tight_layout()
plt.show()

# Create dataset for LSTM
def create_dataset(data, p):
    X, y = [], []
    for i in range(len(data) - p):
        X.append(data[i:i + p])
        y.append(data[i + p])
    return np.array(X), np.array(y)

# Hyperparameters
p = 8  # Number of past observations

# Create dataset
X, y = create_dataset(fused_data, p)

# Reshape data for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(p, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model with history to capture training loss
history = model.fit(X, y, epochs=85, verbose=1)

# Plot training & validation loss values
plt.figure(figsize=(15, 6))
plt.plot(history.history['loss'], color='red', linewidth=2)
plt.title('Model Training Loss', fontsize=10)
plt.xlabel('Epoch', fontsize=9)
plt.ylabel('Loss', fontsize=9)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()

# Predict
y_pred = model.predict(X)

# Calculate residuals
residuals = np.abs(y - y_pred.flatten())

# Distribution Plots
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=df[numeric_cols], kde=True)
plt.title('Sensor Data Distribution', fontsize=10)
plt.xlabel('Sensor Reading', fontsize=9)
plt.ylabel('Count', fontsize=9)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(1, 2, 2)
sns.histplot(data=residuals, kde=True)
plt.title('Residuals Distribution', fontsize=10)
plt.xlabel('Residual Value', fontsize=9)
plt.ylabel('Count', fontsize=9)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.tight_layout()
plt.show()

# Define anomaly threshold
threshold = 4.0 * np.std(residuals)

# Detect anomalies
lstm_anomalies = np.where(residuals > threshold)[0]

# One-Class Support Vector Machines (OCSVM)
ocsvm = OneClassSVM(nu=0.1, gamma='auto')  # Adjust parameters as needed
ocsvm.fit(residuals.reshape(-1, 1))
ocsvm_anomalies = ocsvm.predict(residuals.reshape(-1, 1)) == -1

# Isolation Forests
isolation_forest = IsolationForest(contamination=0.1)  # Adjust parameters as needed
isolation_forest.fit(residuals.reshape(-1, 1))
isolation_forest_anomalies = isolation_forest.predict(residuals.reshape(-1, 1)) == -1

# Majority Voting
ensemble_anomalies = (ocsvm_anomalies.astype(int) + isolation_forest_anomalies.astype(int) + (residuals > threshold).astype(int)) >= 2

# Plot detected anomalies for comparison
fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

# LSTM anomalies
axs[0].plot(fused_data, label='Fused Data', color='orange', linewidth=1)
axs[0].scatter(lstm_anomalies + p, fused_data[lstm_anomalies + p], color='lightgreen', label='Anomalies (LSTM)', s=15)
axs[0].set_title('Anomalies (LSTM)', fontsize=10)
axs[0].legend(fontsize=8)

# OCSVM anomalies
axs[1].plot(fused_data, label='Fused Data', color='orange', linewidth=1)
axs[1].scatter(np.where(ocsvm_anomalies)[0] + p, fused_data[np.where(ocsvm_anomalies)[0] + p], color='lightgreen', label='Anomalies (OCSVM)', s=15)
axs[1].set_title('Anomalies (OCSVM)', fontsize=10)
axs[1].legend(fontsize=8)

# Isolation Forest anomalies
axs[2].plot(fused_data, label='Fused Data', color='orange', linewidth=1)
axs[2].scatter(np.where(isolation_forest_anomalies)[0] + p, fused_data[np.where(isolation_forest_anomalies)[0] + p], color='lightgreen', label='Anomalies (Isolation Forest)', s=15)
axs[2].set_title('Anomalies (Isolation Forest)', fontsize=10)
axs[2].legend(fontsize=8)

# Ensemble anomalies
axs[3].plot(fused_data, label='Fused Data', color='orange', linewidth=1)
axs[3].scatter(np.where(ensemble_anomalies)[0] + p, fused_data[np.where(ensemble_anomalies)[0] + p], color='lightgreen', label='Anomalies (Ensemble)', s=15)
axs[3].set_title('Anomalies (Ensemble)', fontsize=10)
axs[3].legend(fontsize=8)

for ax in axs:
    ax.set_xlabel('Time Step', fontsize=9)
    ax.set_ylabel('Sensor Reading', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.show()

# Suggest preventive measures
def suggest_preventive_measures(anomaly_indices, df):
    measures = []
    for idx in anomaly_indices:
        row = df.iloc[idx]
        if row['X'] > df['X'].quantile(0.95) or row['Y'] > df['Y'].quantile(0.95):
            measures.append((idx, 'Reposition the sensor to avoid high X/Y coordinates. Ensure the sensor is within the designated monitoring area to avoid false readings. Verify the sensor mount is secure and recalibrate its coordinates if necessary.'))
        elif row['SensorData'] > df['SensorData'].quantile(0.95):
            measures.append((idx, 'Check the sensor for potential malfunction. Inspect for any physical damage, dust accumulation, or loose connections. Perform a sensor calibration to ensure accuracy. If issues persist, consider replacing the sensor.'))
        elif row['SensorData'] < df['SensorData'].quantile(0.05):
            measures.append((idx, 'Verify the sensor’s operation as it might be under-reporting. Check the sensor’s positioning and ensure it’s not obstructed. Inspect for any signal interference and ensure the sensor is clean and functional.'))
        elif row['BatteryLife'] < df['BatteryLife'].quantile(0.05):
            measures.append((idx, 'Replace the sensor battery immediately to ensure continuous operation. Regularly monitor battery levels and establish a battery replacement schedule. Consider using high-capacity batteries or a backup power source.'))
        elif row['Temperature'] > df['Temperature'].quantile(0.95):
            measures.append((idx, 'Adjust the environmental conditions to maintain optimal temperature for the sensor. Ensure proper ventilation and consider installing cooling systems if necessary. Regularly monitor the ambient temperature and sensor’s heat generation.'))
        elif row['Temperature'] < df['Temperature'].quantile(0.05):
            measures.append((idx, 'Check for abnormal low temperatures that might affect sensor performance. Ensure the sensor environment is adequately heated or insulated. Inspect for drafts or sources of cold air and mitigate them.'))
        else:
            measures.append((idx, 'No specific measure recommended. Monitor the sensor closely for any further anomalies. Consider performing a general maintenance check on the sensor to ensure its overall functionality.'))
    return measures

# Get ensemble anomaly indices
ensemble_anomaly_indices = np.where(ensemble_anomalies)[0]

# Get preventive measures for ensemble anomalies
ensemble_preventive_measures = suggest_preventive_measures(ensemble_anomaly_indices, df)

# Create a scatter plot
plt.figure(figsize=(15, 6))

# Assuming you have defined the list `ensemble_preventive_measures` and the DataFrame `df` correctly
for idx, measure in ensemble_preventive_measures:
    row = df.iloc[idx]
    plt.scatter(row['X'], row['Y'], label=measure if measure not in plt.gca().get_legend_handles_labels()[1] else '')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Ensemble Anomalies - Preventive Measures')

# Move the legend outside the plot area and adjust its size
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Adjust the layout to prevent the legend from being clipped
plt.tight_layout()

# Display the plot
plt.show()

# Plot residuals with anomaly threshold
plt.figure(figsize=(15, 6))
plt.plot(residuals, label='Residuals', color='blue', linewidth=1)
plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold', linewidth=1)
plt.title('Residuals and Anomaly Detection Threshold', fontsize=10)
plt.xlabel('Time Step', fontsize=9)
plt.ylabel('Residual Value', fontsize=9)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(fontsize=8)
plt.show()

print("Anomalies detected at indices (LSTM):", lstm_anomalies)
print("Anomalies detected at indices (OCSVM):", np.where(ocsvm_anomalies)[0])
print("Anomalies detected at indices (Isolation Forest):", np.where(isolation_forest_anomalies)[0])
print("Anomalies detected at indices (Ensemble):", np.where(ensemble_anomalies)[0])
