import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

class Plotter:
    def __init__(self, data_file):
        self.data = pd.read_json(data_file)
        self.plot_folder = "plots"
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        self.evaluation_metrics = self.evaluate_model_performance()
    
    def evaluate_model_performance(self):
        gt_corners = self.data['gt_corners'].values
        rb_corners = self.data['rb_corners'].values

        mae = np.mean(np.abs(gt_corners - rb_corners))
        mse = np.mean((gt_corners - rb_corners) ** 2)
        rmse = np.sqrt(mse)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
    
    def draw_plots(self):
        plot_paths = []
        
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(self.data))

        plt.bar(index, self.data['gt_corners'], bar_width, label='Ground Truth', color='blue')
        plt.bar(index + bar_width, self.data['rb_corners'], bar_width, label='Model Prediction', color='red', alpha=0.7)
        
        plt.xlabel('Room')
        plt.ylabel('Number of corners')
        plt.title('Ground truth and model predictions')
        plt.xticks(index + bar_width / 2, self.data['name'], rotation=90)
        plt.legend()
        
        plt.text(0.01, 0.95, f"MAE: {self.evaluation_metrics['MAE']:.2f}\nMSE: {self.evaluation_metrics['MSE']:.2f}\nRMSE: {self.evaluation_metrics['RMSE']:.2f}",
                 transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plot_path = os.path.join(self.plot_folder, 'gt_vs_rb_corners.png')
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plt.clf()

        deviation_cols = ['mean', 'max', 'min', 'floor_mean', 'floor_max', 'floor_min', 'ceiling_mean', 'ceiling_max', 'ceiling_min']
        for col in deviation_cols:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='name', y=col, data=self.data)
            plt.xticks(rotation=90)
            plt.title(f"Deviation {col} values")
            plt.tight_layout()
            plot_path = os.path.join(self.plot_folder, f'{col}_deviation.png')
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.clf()
        
        return plot_paths
    

if __name__ == "__main__":
    data_file = 'deviation.json'
    plotter = Plotter(data_file)
    plot_paths = plotter.draw_plots()