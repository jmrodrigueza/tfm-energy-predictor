# Repositorio de código para el TFM de UNIR "Predicción de la energía eléctrica generada a través de variables climatológicas mediante Deep Learning"
**Autor**: José Manuel Rodríguez Alves <br/>
**Web Personal**: [https://www.jmrodriguezalves.es/](https://www.jmrodriguezalves.es/) <br/>
**LinkedIn
<a href="https://es.linkedin.com/in/josemralves">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="15" height="15">
</a>:** [https://es.linkedin.com/in/josemralves](https://es.linkedin.com/in/josemralves)<br/>
---
# Resumen del TFM
<p align="justify">
En este artículo se presenta un software desarrollado que permite la predicción energética y emisiones de España basándose principalmente variables meteorológicas obtenidas de las fuentes abiertas: AEMET, EUMETSAT y REE. Debido a la baja resolución temporal disponible en abierto de las variables meteorológicas, se ha desarrollado un modelo de imputación de datos meteorológicos basado en redes LSTM que ha permitido el desarrollo de los modelos energéticos con resolución horaria. El software desarrollado suministra información predictiva acerca de la demanda, producción y emisiones de CO<sub>2</sub> equivalente, y sirve de apoyo durante proceso de selección de inversiones en energías renovables. Se ha implementado con éxito una web visualmente atractiva basada en tecnologías Angular, Flask y utilizando modelos Keras predictivos energéticos basados en transformers. Los resultados muestran que es posible predecir las variables energéticas haciendo uso de variables meteorológicas con una resolución horaria y con un error aceptable.
</p>

# Estructura del código
El proyecto se compone de varias partes:
* **/AEMET-data**: Contiene los datos meteorológicos de AEMET.
* **/common**: Contiene los ficheros comunes a todos los módulos.
* **/energy_meteo_files**: Contiene los datos de entrenamiento y test de los modelos, el fichero "energy_meteo_data.parquet".
* **/EUMESAT-data**: Contiene los datos meteorológicos de EUMETSAT.
* **/REE-data**: Contiene los datos de energía, demanda, generación y emisiones de REE.
* **/tfm-energy-predictor-backend**: Contiene el código del backend de la Web.
* **/tfm-energy-predictor-web**: Contiene el código del frontend Angular.
* **/tfm-energy-predictor-hugging-face**: Contiene el backend que ejecuta la inferencia en HuggingFace.

### Ficheros de código
* **Dockerfile**: Fichero de configuración de Docker. Sirve para crear una imagen Docker de la web.
* **energy_base_predictor.py**: Clase base del modelo de _transformers_ para la predicción de energía y emisiones.
* **energy_demand_predictor.py**: Modelo de transformers para la predicción de la demanda energética, utiliza la clase base energy_base_predictor.py.
* **energy_emission_predictor.py**: Modelo de transformers para la predicción de las emisiones de CO<sub>2</sub> equivalente, utiliza la clase base energy_base_predictor.py.
* **energy_gen_predictor.py**: Modelo de transformers para la predicción de la generación energética, utiliza la clase base energy_base_predictor.py.
* **energy_sim_emi_predictor.py**: Modelo de transformers para la predicción de las emisiones de CO<sub>2</sub> equivalente, utiliza la clase base energy_base_predictor.py.
* **httpd.conf**: Fichero de configuración de Apache.
* **meteo_energy_databuilder.py**: Clase para la construcción de los datos de entrenamiento.
* **observation_data_builder.py**: Clase para la construcción de los datos de observación.
* **README.md**: Este fichero.
* **ree_predictors_comparator.py**: Clase para la comparación de los modelos predictivos.
* **requirements.txt**: Fichero con las dependencias del proyecto python.
* **TFM_notebook_colab.ipynb**: Notebook de Google Colab con el código utilizado para entrenar en Google Colab.
* **train_models.py**: Clase para el entrenamiento de los modelos predictivos.
