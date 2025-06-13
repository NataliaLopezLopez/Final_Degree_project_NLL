from scipy import stats

import numpy as np
import multiprocess as mp
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from egg_utils_2 import eeg
import mne
import matplotlib.colors as colors

# 1. Here we are defining the parameters 

number_of_subjects = 109 #shouldn't be larger that 108, dataset has 109 but subject 109 has some not valid values at the end (this should be automated by some error control)
# if larger than 97, the actual number of subjects is number_of_subjects-1, since subject 97 is removed for the same reasons as subject 109
filt_mode = 'raw' # 'raw' or 'filt', for considering only the alpha band
word_length = 4 # Word length
lag = 1 # Spatial lag
analysis_mode = 'temporal' # Controls the symbol construction (temporal or spatial) 
# and how the averages are skewrformed
# 'ensemble' : spatial analysis, but for each time, the mean value of Sskew in all subjects if provided
# This analysis corresponds to the one of Boaretto et al. (2023), and outputs the data used to
# plot Fig.4 of Gancio et al. (2024). It provides Sskew for diferente arrengement:
# linear: straight foward, as come from the dataset, ordering (first approa of Boaretto et al. (2023))
# boaretto's best: best arrengement of electrodes find by Boaretto et al. (2023)
# horizontal symbols
# vertical symbosl
#
# 'spatial': Spatial analysis as skewrformed by Gancio et al. (2024), an average in time of the Sskew values is provided 
# (one averaged quantity for each subject). This provied the data for Fig. 5,and partial of Figs. 7 and 8 of  Gancio et al. (2024)
#
# 'temporal':  Temporal analysis as skewrformed by Gancio et al. (2024), an average in space of the Permutation Entropy (skew)
#  values is provided (one averaged quantity for each subject). This provied the data for Fig. 6, and the additional
# data for Figs. 7 and 8 of  Gancio et al. (2024)



# 2. Here what we are doing is to create the objects 

eeg_open = eeg(number_of_subjects, filt_mode, run=1)  # Oskewn eyes 
eeg_closed = eeg(number_of_subjects, filt_mode, run=2)  # Closed eyes 


# Parameters 
for eeg_obj in [eeg_open, eeg_closed]:
    eeg_obj.L = word_length
    eeg_obj.lag = lag
    eeg_obj.file_path = '/Users/natalialopezlopezicloud.com/Desktop/Escritorio2/GAIA/eeg-spatial-analysis-main/files-2'
    eeg_obj.cut_up = 30
    eeg_obj.cut_low = 12


# 3. Here what we are doing is to load the data. 

eeg_open.load_data()
eeg_closed.load_data()


# 4. Here we analize the data

if analysis_mode == 'temporal':
    startTime = datetime.now()

    # oskewn 
    if __name__ == '__main__':  
        pool = mp.Pool(mp.cpu_count())
        skew_eyes_open = pool.map(eeg_open.skewness_channel, range(eeg_open.subjects))
        pool.close()
        pool.join()
    av_open = np.mean(np.array(skew_eyes_open),axis=0)
    print('skew Eyes Open - mean =', np.mean(skew_eyes_open), ', std =', np.std(skew_eyes_open))

    # closed 
    if __name__ == '__main__':  
        pool = mp.Pool(mp.cpu_count())
        skew_eyes_closed = pool.map(eeg_closed.skewness_channel, range(eeg_closed.subjects))
        pool.close()
        pool.join()
    av_closed = np.mean(np.array(skew_eyes_closed),axis=0)  # avarage por canal de los sujetos para 64 canales 
    print('skew Eyes Closed - mean =', np.mean(skew_eyes_closed), ', std =', np.std(skew_eyes_closed))

    # time of the execution 
    print('Time elapsed:' + str(datetime.now() - startTime))
    print('Process completed.')


np.save('EC_skew_raw_4_1_w',np.array(skew_eyes_closed))
np.save('EO_skew_raw_4_1_w',np.array(skew_eyes_open))


# 5. CALCULAR p-VALUE ENTRE EO Y EC POR CANAL (PRUEBA t DE STUDENT PAREADA)
skew_eyes_closed=np.array(skew_eyes_closed) # dimension 
skew_eyes_open=np.array(skew_eyes_open) # dimensión 

t_stats = []
p_values = []

for chanel in range(64):
    t_stat, p_val = stats.ttest_ind(skew_eyes_open[:, chanel], skew_eyes_closed[:, chanel], equal_var=False)
    t_stats.append(t_stat)
    p_values.append(p_val)

#### hay que pasar las listas a arrays 
t_stats = np.array(t_stats)
p_values = np.array(p_values)



# Establecer un umbral de significancia para resaltar los electrodos donde p < 0.05
significance_mask = p_values < 0.05

# 6. VISUALIZACIÓN mne.viz.plot_topomap

name ='files-2/S001/S001R01.edf' #This is just a random subject in order to get the montage
raw = mne.io.read_raw_edf(name,verbose=None)
#raw.plot()
raw.load_data()
# Set montage
mne.datasets.eegbci.standardize(raw)
raw.set_montage("standard_1005")
###########################################

# Asignar los valores promedio de Permutation Entropy (skew) para ojos abiertos (EO) y cerrados (EC)
skew_eyes_open_avg = av_open  # Promedio por canal para ojos abiertos
skew_eyes_closed_avg = av_closed  # Promedio por canal para ojos cerrados

# Obtener las posiciones de los electrodos desde el objeto eeg_oskewn.
# Solo se toman las dos primeras columnas para obtener las coordenadas 2D.
#positions = np.array(eeg_oskewn.get_pos())[:, :-1]

# Calcular la diferencia entre los valores de skew para EO y EC (diferencia por canal)
skew_dif = skew_eyes_open_avg - skew_eyes_closed_avg


# Determinar el rango de colores compartido para EO y EC.
# Este rango se basa en el mínimo y máximo global entre ambas condiciones.
vmin_eo_ec = min(skew_eyes_open_avg.min(), skew_eyes_closed_avg.min())
vmax_eo_ec = max(skew_eyes_open_avg.max(), skew_eyes_closed_avg.max())

# Determinar el rango de colores para el mapa de diferencias (dinámico).
vmin_diff, vmax_diff = skew_dif.min(), skew_dif.max()

# Crear un lienzo para tres gráficos en una fila y con espacio adicional para el mapa de diferencias.
fig, axes = plt.subplots(1, 4, figsize=(24, 6), gridspec_kw={'width_ratios': [1, 1, 1.2, 1]})

# Gráfico 1: EO 
# Crear el mapa topográfico para los valores de skew en ojos abiertos.
im_eo, _ = mne.viz.plot_topomap(
    data=skew_eyes_open_avg,  # Datos promedio por canal para EO
    pos=raw.info,          # Coordenadas 2D de los electrodos
    cmap='plasma',          # Mapa de colores tipo plasma (morado a amarillo)
    contours=0,            # Número de contornos para destacar las variaciones
    image_interp='cubic',   # Interpolación suave entre puntos
    vlim=(vmin_eo_ec, vmax_eo_ec),  # Rango de colores compartido con EC
    axes=axes[0],           # Dibuja en el primer gráfico
    show=False              # No mostrar el gráfico aún
)
axes[0].set_title("EO (Eyes Open)")  # Título del gráfico

# Gráfico 2: EC 
# Crear el mapa topográfico para los valores de skew en ojos cerrados.
im_ec, _ = mne.viz.plot_topomap(
    data=skew_eyes_closed_avg,  # Datos promedio por canal para EC
    pos=raw.info,            # Coordenadas 2D de los electrodos
    cmap='plasma',            # Mapa de colores tipo plasma (morado a amarillo)
    contours=0,              # Número de contornos para destacar las variaciones
    image_interp='cubic',     # Interpolación suave entre puntos
    vlim=(vmin_eo_ec, vmax_eo_ec),  # Rango de colores compartido con EO
    axes=axes[1],             # Dibuja en el segundo gráfico
    show=False                # No mostrar el gráfico aún
)
axes[1].set_title("EC (Eyes Closed)")  # Título del gráfico


# ---- Gráfico 3: (EO - EC) con significancia
# Crear el mapa topográfico para la diferencia entre EC y EO.
im_diff, _ = mne.viz.plot_topomap(
    data=skew_dif,              # Datos de diferencia por canal (EC - EO)
    pos=raw.info,            # Coordenadas 2D de los electrodos
    cmap='coolwarm',          # Mapa de colores azul-rojo para valores negativos y positivos
    contours=0,              # Número de contornos para destacar las variaciones
    image_interp='cubic',     # Interpolación suave entre puntos
    vlim=(vmin_diff, vmax_diff),  # Rango dinámico para las diferencias
    axes=axes[2],             # Dibuja en el tercer gráfico
    show=False                # No mostrar el gráfico aún
)
axes[2].set_title("Difference EO and EC")  # Título del gráfico

# Barra de color compartida para EO y EC 
# Crear una barra de color unificada que abarque EO y EC.
cbar = fig.colorbar(
    im_eo,                   # Usar el gráfico de EO como referencia
    ax=[axes[0], axes[1]],   # Abarcar los gráficos de EO y EC
    orientation='horizontal',  # Barra horizontal
    fraction=0.05,           # Tamaño relativo de la barra
    pad=0.2                  # Espaciado entre la barra y los gráficos
)
cbar.set_label("skewness (EO and EC)")  # Etiqueta para la barra

# Barra de color para el mapa de diferencias 
# Crear una barra de color indeskewndiente para el mapa de diferencias.
cbar_diff = fig.colorbar(
    im_diff,                # Usar el gráfico de diferencias como referencia
    ax=axes[2],             # Solo aplica al gráfico de diferencias
    orientation='horizontal',  # Barra horizontal
    fraction=0.05,          # Tamaño relativo de la barra
    pad=0.2                 # Espaciado entre la barra y el gráfico
)
cbar_diff.set_label("skewness (Difference)")  # Etiqueta para la barra

# ---- Gráfico 4: p-values
# Crear el mapa topográfico para los valores de p.

im_pval, _ = mne.viz.plot_topomap(
    data=p_values,            # Datos de p-value por canal
    pos=raw.info,            # Coordenadas 2D de los electrodos
    cmap='plasma',         # Mapa de colores inverso para resaltar valores bajos
    contours=0,              # Número de contornos para destacar las variaciones
    image_interp='linear',     # Interpolación suave entre puntos
    #vlim=(0, 1),              # El p-value está entre 0 y 1
    cnorm=colors.LogNorm(vmin=p_values.min(), vmax=p_values.max()), 
    axes=axes[3],             # Dibuja en el cuarto gráfico
    show=False                # No mostrar el gráfico aún
)
axes[3].set_title("p-values (EO vs EC)")  # Título del gráfico de p-values

# Barra de color para el mapa de p-values
cbar_pval = fig.colorbar(
    im_pval,                # Usar el gráfico de p-values como referencia
    ax=axes[3],             # Solo aplica al gráfico de p-values
    orientation='horizontal',  # Barra horizontal
    fraction=0.05,          # Tamaño relativo de la barra
    pad=0.2                 # Espaciado entre la barra y el gráfico
)
cbar_pval.set_label("p-value (EO vs EC)")  # Etiqueta para la barra de p-values


# Ajustar el diseño automáticamente para evitar solapamientos.
#plt.tight_layout()

# Guardar la figura en un archivo de alta resolución.
font = {'size'   : 28}
matplotlib.rc('font', **font)
plt.savefig("skew_topomap_corrected_range_raw_4_1_W.png", dpi=300)
# Mostrar la figura en pantalla.
plt.show()





# 5 . Visualization of the data. 

#box = plt.boxplot([skew_eyes_closed, skew_eyes_oskewn], patch_artist=True)
#[box['boxes'][i].set_facecolor(color) for i, color in enumerate(['#ADD8E6', '#FFA07A'])]

#plt.boxplot([skew_eyes_closed, skew_eyes_oskewn])
#plt.xticks([1,2], ["EC", "EO"])
#plt.savefig("skew_L={}_lag={}.png".format(word_length, lag), dpi=300)
#plt.savefig("skew_L={}_lag={}_alfa.png".format(word_length, lag), dpi=300)
#plt.savefig("skew_L={}_lag={}_delta.png".format(word_length, lag), dpi=300)
#plt.savefig("skew_L={}_lag={}_Theta.png".format(word_length, lag), dpi=300)
#lt.savefig("skew_L={}_lag={}_beta.png".format(word_length, lag), dpi=300)
#plt.savefig("skew_L={}_lag={}_gamma.png".format(word_length, lag), dpi=300)

### HERE WE APPLY NOTCH INSTEAD OF FILT 

#plt.savefig("skew_L={}_lag={}.png".format(word_length, lag), dpi=300)
#plt.savefig("skew_L={}_lag={}_alfa_notch.png".format(word_length, lag), dpi=300)
#plt.savefig("skew_L={}_lag={}_delta_notch.png".format(word_length, lag), dpi=300)
#plt.savefig("skew_L={}_lag={}_Theta_notch.png".format(word_length, lag), dpi=300)
#plt.savefig("skew_L={}_lag={}_beta_notch.png".format(word_length, lag), dpi=300)
#plt.savefig("skew_L={}_lag={}_gamma_notch.png".format(word_length, lag), dpi=300)




