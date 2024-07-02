import numpy as np
import matplotlib.pyplot as plt

# # Sample Data
# observed =	[-20.95,	-21.19,	-20.87,	-21.45,	-20.54,	-17.21,	-23.84,	-20.24,	-27.97,	-22.54]


# models = { 
# 'SVR':	    [-20.65109211,	-21.20549315,	-20.7510044,	-20.83592049,	-20.98843335,	-21.4409898,	-22.01751712,	-23.86781586,	-27.36404707,	-23.09769981],
# 'KNN':	    [-21.794,   	-21.852,    	-20.994,    	-22.97,      	-20.76,     	-20.454,	-22.85,	-21.088,	-28.436,	-23.186],
# 'RF':	    [-21.9035,  	-23.7057,   	-20.8211,   	-22.0357,   	-20.849,	-20.1511,	-21.7886,	-21.9073,	-28.4361,	-23.1106],
# 'Ext':	    [-21.414,   	-23.2133,   	-20.8411,   	-22.1395,   	-20.9297,	-19.4117,	-23.2275,	-20.9322,	-28.6252,	-23.9064],
# 'AdaBt':	[-21.156,   	-26.41571429,   -21.14222222,	-21.156,    	-21.14222222,	-20.504,	-21.2425,	-22.14875,	-28.5,	-22.66],
# 'Grd':	    [-20.89690913,	-26.12561313,	-20.75879693,	-20.69947369,	-20.75879693,	-19.55562787,	-21.60343584,	-20.88805057,	-28.394469,	-23.88155181],
# 'Bag':	    [-22.534,   	-23.91,     	-21.47,     	-21.852,    	-20.834,	-20.319,	-21.981,	-21.997,	-28.174,	-23.938],
# 'CAT':	    [-21.14375406,	-25.45952815,	-20.61943462,	-21.36035389,	-20.88473182,	-19.97346556,	-22.5190788,	-20.94629424,	-28.55959029,	-23.70974279],
# 'Model0':	[-20.91843197,	-21.40444063,	-21.09891225,	-21.74882407,	-21.22673046,	-18.28103194,	-23.04884245,	-20.88860135,	-28.10198211,	-21.53063821],
# 'Model1':	[-20.7685141,	-21.6756648,	-19.90207233,	-22.51584542,	-19.95279637,	-18.54627555,	-22.49751393,	-20.63164276,	-27.42358004,	-23.23572356],
# 'Model2':	[-21.81083333,	-21.41916667,	-20.81333333,	-21.74571429,	-20.89666667,	-18.07,	-21.81083333,	-21.68166667,	-28.37,	-21.41916667],
# 'Model3':	[-21.25842105,	-20.95571429,	-20.62,       	-21.25842105,	-20.62642857,	-19.62,	-22.31857143,	-20.91909091,	-28.612,	-23.38615385],
# 'Model4':	[-20.5882,  	-20.9461,   	-20.2386,   	-22.3283,   	-20.2412,	-19.0561,	-21.704,	-20.3575,	-28.1704,	-23.4914],
# }
import pandas as pd
import os
results_file = os.path.join('data.csv')
df = pd.read_csv(results_file)
observed = df.iloc[:, 1]
df = df.iloc[:, 2:]
models = {}

# folder_path = os.path.join('.\\Figures\\Taylor')
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)

                        
for column in df.columns:
    models[column] = df[column].values


markers = [ '*', 's','^', 'D', 'x']#, '*', "p", "P", "X", "d", "h","1","o"]
marker_size = 10
std_dev_obs = np.std(observed)
correlations = {model: np.corrcoef(observed, prediction)[0,1] for model, prediction in models.items()}
std_devs = {model: np.std(prediction) for model, prediction in models.items()}

fig = plt.figure(dpi=100)
plt.rcParams.update({'font.size': 14})
ax1 = fig.add_subplot(111, polar=True)

for i, (model, prediction) in enumerate(models.items()):
    corr_radians = np.arccos(correlations[model])
    ax1.plot(corr_radians, std_devs[model], markers[i], label=model, markersize=marker_size)  

ax1.plot(0, std_dev_obs, 'o', label='Observed', markersize=marker_size)   

correlation_values = np.arange(0, 1.01, 0.1)  
correlation_values_additional = np.arange(0.9, 1.01, 0.02)  
correlation_values_all = np.concatenate([correlation_values, correlation_values_additional])  
correlation_radians = np.arccos(correlation_values_all)

ax1.set_xticks(correlation_radians)
ax1.set_xticklabels([f'{corr:.2f}' for corr in correlation_values_all], rotation=45,fontsize=15)  

ax1.set_title('Taylor Diagram', pad=20,fontsize=20)

ax1.set_thetamin(0)
ax1.set_thetamax(90)

ax1.set_xlabel('Standard Deviation of Predicted Values', labelpad=20,fontsize=15)
ax1.set_ylabel('Standard Deviation of Observed Values', labelpad=40,fontsize=15)

max_std_dev = max([std_dev_obs] + list(std_devs.values()))
corr_grid, std_grid = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, max_std_dev * 1.5, 100))
rms_grid = np.sqrt(std_grid**2 + std_dev_obs**2 - 2 * std_grid * std_dev_obs * corr_grid)
corr_grid_radians = np.arccos(corr_grid)

contours = ax1.contour(corr_grid_radians, std_grid, rms_grid, levels=5, colors='Blue')
ax1.clabel(contours, inline=True, fontsize=15)

circle_radius = std_dev_obs 
theta = np.linspace(0, np.pi/2, 100)
r = np.full_like(theta, circle_radius)
ax1.plot(theta, r, 'k--')  


correlation_label = f'Standard Deviation: {round(circle_radius, 2)}'
ax1.text(np.pi/4, circle_radius, correlation_label, rotation=-45, horizontalalignment='center', verticalalignment='center', fontsize=12)
ax1.text(np.pi/4, std_grid.max() + 0.025, 'Correlation Coefficient', rotation=-45, horizontalalignment='center', verticalalignment='center', fontsize=15)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1),fontsize=10)
# svg_path = os.path.join(folder_path, 'Taylor.svg')
plt.tight_layout()
# plt.savefig(svg_path, dpi=400)
plt.show()
plt
