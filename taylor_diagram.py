import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class TaylorDiagram:
    def __init__(self, csv_file, markers=None, marker_size=10):
        self.csv_file = csv_file
        self.markers = markers if markers is not None else ['*', 's', '^', 'D', 'x']
        self.marker_size = marker_size
        self.observed, self.models = self.load_data()
        self.std_dev_obs = np.std(self.observed)
        self.correlations = {model: np.corrcoef(self.observed, prediction)[0, 1] for model, prediction in self.models.items()}
        self.std_devs = {model: np.std(prediction) for model, prediction in self.models.items()}

    def load_data(self):
        df = pd.read_csv(self.csv_file)
        observed = df.iloc[:, 1]
        models = {column: df[column].values for column in df.columns[2:]}
        return observed, models

    def plot(self):
        fig = plt.figure(dpi=100)
        plt.rcParams.update({'font.size': 14})
        ax1 = fig.add_subplot(111, polar=True)

        for i, (model, prediction) in enumerate(self.models.items()):
            corr_radians = np.arccos(self.correlations[model])
            ax1.plot(corr_radians, self.std_devs[model], self.markers[i % len(self.markers)], label=model, markersize=self.marker_size)

        ax1.plot(0, self.std_dev_obs, 'o', label='Observed', markersize=self.marker_size)

        correlation_values = np.arange(0, 1.01, 0.1)
        correlation_values_additional = np.arange(0.9, 1.01, 0.02)
        correlation_values_all = np.concatenate([correlation_values, correlation_values_additional])
        correlation_radians = np.arccos(correlation_values_all)

        ax1.set_xticks(correlation_radians)
        ax1.set_xticklabels([f'{corr:.2f}' for corr in correlation_values_all], rotation=45, fontsize=15)

        ax1.set_title('Taylor Diagram', pad=20, fontsize=20)
        ax1.set_thetamin(0)
        ax1.set_thetamax(90)

        ax1.set_xlabel('Standard Deviation of Predicted Values', labelpad=20, fontsize=15)
        ax1.set_ylabel('Standard Deviation of Observed Values', labelpad=40, fontsize=15)

        max_std_dev = max([self.std_dev_obs] + list(self.std_devs.values()))
        corr_grid, std_grid = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, max_std_dev * 1.5, 100))
        rms_grid = np.sqrt(std_grid**2 + self.std_dev_obs**2 - 2 * std_grid * self.std_dev_obs * corr_grid)
        corr_grid_radians = np.arccos(corr_grid)

        contours = ax1.contour(corr_grid_radians, std_grid, rms_grid, levels=5, colors='Blue')
        ax1.clabel(contours, inline=True, fontsize=15)

        circle_radius = self.std_dev_obs
        theta = np.linspace(0, np.pi/2, 100)
        r = np.full_like(theta, circle_radius)
        ax1.plot(theta, r, 'k--')

        correlation_label = f'Standard Deviation: {round(circle_radius, 2)}'
        ax1.text(np.pi/4, circle_radius, correlation_label, rotation=-45, horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax1.text(np.pi/4, std_grid.max() + 0.025, 'Correlation Coefficient', rotation=-45, horizontalalignment='center', verticalalignment='center', fontsize=15)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    results_file = 'data.csv'
    taylor_diagram = TaylorDiagram(results_file)
    taylor_diagram.plot()
