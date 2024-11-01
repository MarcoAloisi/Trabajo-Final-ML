# Predicción del % de Victorias de los Top 50 Jugadores de Tennis

Este proyecto utiliza técnicas de Machine Learning para predecir el porcentaje de victorias de los 50 mejores jugadores de tenis. La predicción se realiza en base a datos históricos de la temporada actual (últimas 52 semanas) y se enfoca en optimizar la precisión de los modelos para ofrecer estimaciones confiables del rendimiento de los jugadores.

Para entrenar el modelo, se usaron datos de los jugadores ubicados en los rankings del 51 al 100, así como jugadores del circuito Challenger.

## Obtención y Preprocessing de los Datos

Para conseguir los datos, se implementó un proceso de web scraping con Selenium en la página [Tennis Abstract](https://www.tennisabstract.com/cgi-bin/leaders.cgi). Tras la obtención de los datos, se realizó una limpieza de la tabla, eliminando porcentajes (`%`) y transformando variables de tipo `object` a tipos numéricos (`int` y `float`) o categóricos según fuera necesario.

### Feature Engineering

Se llevó a cabo un proceso de ingeniería de características para mejorar el rendimiento del modelo:

- **Creación de Nuevas Variables**: Se generaron variables adicionales con un significado específico para captar mejor el rendimiento de los jugadores.
- **Selección de Variables**: Se eliminaron todas las variables con una correlación superior a 0.8 con la variable objetivo y aquellas que presentaban muy poca correlación. Por ejemplo, la variable de porcentaje de sets ganados mostraba una correlación de 0.98 con la variable objetivo, lo que ocasionaba que el modelo se basara únicamente en esta variable para realizar sus predicciones. Al eliminar estas variables altamente correlacionadas, se mejoró la diversidad de las características utilizadas, reduciendo la multicolinealidad y optimizando el rendimiento del modelo.

## Requisitos Previos Para el Docker

Para ejecutar este proyecto, asegúrate de tener Docker Desktop instalado. 
Esto permitirá crear un contenedor con el entorno necesario para la ejecución del notebook y las librerías de Machine Learning requeridas.

Tambien es necesario descargar la carpeta "app" del github y guardarla en una locacion que accederas luego desde Docker Desktop, por ejemplo en el escritorio

## Ejecución

1. Abrir la app de Docker Desktop
2. Abrir la terminal y navegar hasta la carpeta app que mencione anteriormente (la cual debes de descargar de este github): cd ruta/a/la/carpeta/app
4. Una vez dentro de la carpeta app, ejecutar el siguiente comando:
docker build -t gradio-app .
docker run -p 7860:7860 gradio-app
5. Una vez ejecutado, acceder al link que saldra en la pantalla de Docker que seria este: http://localhost:7860/
6. Dentro de la app cargar el archivo df_combined_predict.csv que lo puedes descargar de la carpeta "data" de este mismo github
7. Por ultimo pulsar predict y esperar los resultados
