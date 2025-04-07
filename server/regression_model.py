import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # Prevent overflow
from sklearn.model_selection import train_test_split # Split into train and validation sets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Linear regression model in PyTorch
class TorchLinearRegressionModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(TorchLinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Load and prepare the data
        current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(current_dir, 'data.csv')
        self.data = pd.read_csv(csv_path)

        self.data['Date_dt'] = pd.to_datetime(self.data['# Date'])
        self.min_date = self.data['Date_dt'].min()
        self.data['# Date'] = (self.data['Date_dt'] - self.min_date).dt.days

        # Split the data 
        train_df, val_df = train_test_split(self.data, test_size=0.15, shuffle=False)

        self.train_df = train_df.copy()
        self.val_df = val_df.copy()

        # Initialize scalers and fit only on train
        self.scaler_date = MinMaxScaler()
        self.scaler_receipt = MinMaxScaler()

        self.train_df['# Date_scaled'] = self.scaler_date.fit_transform(train_df[['# Date']])
        self.train_df['Receipt_Count_scaled'] = self.scaler_receipt.fit_transform(train_df[['Receipt_Count']])
        
        self.val_df['# Date_scaled'] = self.scaler_date.transform(val_df[['# Date']])
        self.val_df['Receipt_Count_scaled'] = self.scaler_receipt.transform(val_df[['Receipt_Count']])
        
        # Convert to torch tensors
        self.x_train = torch.tensor(self.train_df['# Date_scaled'].values, dtype=torch.float32).view(-1, 1)
        self.y_train = torch.tensor(self.train_df['Receipt_Count_scaled'].values, dtype=torch.float32).view(-1, 1)
        self.x_val = torch.tensor(self.val_df['# Date_scaled'].values, dtype=torch.float32).view(-1, 1)
        self.y_val = torch.tensor(self.val_df['Receipt_Count_scaled'].values, dtype=torch.float32).view(-1, 1)

        # Initialize model, loss, and optimizer
        self.model = TorchLinearRegressionModel()
        self.criteria = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.train_model()

    def train_model(self):
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad() # Clear the gradients 
            outputs = self.model(self.x_train)
            loss = self.criteria(outputs, self.y_train)
            loss.backward() # Backpropagation
            self.optimizer.step()

    def compute_metrics(self):
        self.model.eval()
        with torch.no_grad(): # No gradients needed
            preds_scaled = self.model(self.x_val)
            mse_val_scaled = self.criteria(preds_scaled, self.y_val).item()

        return mse_val_scaled

    def visualize_data(self):
        # Predict future 366 days 
        last_day = self.data['# Date'].max()
        future_days = np.arange(last_day + 1, last_day + 367).reshape(-1, 1)
        future_scaled = self.scaler_date.transform(future_days)

        x_future = torch.tensor(future_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            future_preds_scaled = self.model(x_future).numpy()
        future_preds = self.scaler_receipt.inverse_transform(future_preds_scaled)

        future_dates = pd.to_datetime(self.min_date) + pd.to_timedelta(future_days.flatten(), unit='D')
        future_df = pd.DataFrame({
            '# Date': future_dates,
            'Predicted_Receipt_Count': future_preds.flatten()
        })

        # Fitted line
        x_orig = np.linspace(self.train_df['# Date'].min(), self.val_df['# Date'].max(), 100)
        x_orig_scaled = self.scaler_date.transform(x_orig.reshape(-1, 1))
        x_orig_tensor = torch.tensor(x_orig_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_orig_scaled = self.model(x_orig_tensor).numpy()
        y_orig = self.scaler_receipt.inverse_transform(y_orig_scaled)

        # Plot everything
        plt.figure(figsize=(12, 6))
        train_dates = pd.to_datetime(self.min_date) + pd.to_timedelta(self.train_df['# Date'], unit='D')
        val_dates = pd.to_datetime(self.min_date) + pd.to_timedelta(self.val_df['# Date'], unit='D')

        plt.scatter(train_dates, self.train_df['Receipt_Count'], color='blue', label='Train Data')
        plt.scatter(val_dates, self.val_df['Receipt_Count'], color='orange', label='Validation Data')
        plt.plot(pd.to_datetime(self.min_date) + pd.to_timedelta(x_orig.flatten(), unit='D'), y_orig, color='red', label='Fitted Line')
        plt.scatter(future_df['# Date'], future_df['Predicted_Receipt_Count'], color='purple', label='Future Prediction', marker='x', s=10)

        # Label start of each future month
        future_df['Month'] = future_df['# Date'].dt.to_period('M')
        monthly_labels_df = future_df.groupby('Month').first().reset_index()
        for _, row in monthly_labels_df.iterrows():
            plt.annotate(f"{int(row['Predicted_Receipt_Count'])}", 
                         (row['# Date'], row['Predicted_Receipt_Count']),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='darkgreen')

        plt.title(f'Linear Regression (lr={self.learning_rate}, epochs={self.epochs})')
        plt.xlabel('Date')
        plt.ylabel('Receipt Count')
        plt.legend()

        output_dir = os.path.join(os.path.dirname(__file__), 'static')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'generated_plot_torch.png')
        plt.savefig(output_path)
        plt.close()

        mse = self.compute_metrics()
        return output_path, future_df, mse
