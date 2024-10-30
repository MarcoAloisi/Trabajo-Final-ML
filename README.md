# Predicción del % de Victorias de los Top 50 Jugadores de Tennis

Este proyecto utiliza técnicas de Machine Learning para predecir el porcentaje de victorias de los 50 mejores jugadores de tenis. La predicción se realiza en base a datos históricos de la temporada actual (últimas 52 semanas) y se enfoca en optimizar la precisión de los modelos para ofrecer estimaciones confiables del rendimiento de los jugadores.

Para entrenar el modelo, se usaron datos de los jugadores ubicados en los rankings del 51 al 100, así como jugadores del circuito Challenger.

## Obtención y Preparación de los Datos

Para conseguir los datos, se implementó un proceso de web scraping con Selenium en la página [Tennis Abstract](https://www.tennisabstract.com/cgi-bin/leaders.cgi). Tras la obtención de los datos, se realizó una limpieza de la tabla, eliminando porcentajes (`%`) y transformando variables de tipo `object` a tipos numéricos (`int` y `float`) o categóricos según fuera necesario.

### Ingeniería de Características

Se llevó a cabo un proceso de ingeniería de características para mejorar el rendimiento del modelo:

- **Creación de Nuevas Variables**: Se generaron variables adicionales con un significado específico para captar mejor el rendimiento de los jugadores.
- **Selección de Variables**: Se eliminaron todas las variables con una correlación superior a 0.8 con la variable objetivo y aquellas que presentaban muy poca correlación, con el fin de optimizar el rendimiento del modelo y reducir la multicolinealidad.

## Requisitos Previos Para el Docker

Para ejecutar este proyecto, asegúrate de tener Docker Desktop instalado. 
Esto permitirá crear un contenedor con el entorno necesario para la ejecución del notebook y las librerías de Machine Learning requeridas.

Tambien es necesario descargar la carpeta app del github y guardarla en una locacion que accederes luego desde Docker Desktop

## Ejecución

1- Abrir la app de Docker Deskptop
2- Abrir la terminal y navegar hasta la carpetta app que mencione anteriormente (la cual debes de descargar de este github)
3- Una vez dentro de la carpeta app, ejecutar el siguiente comando:
docker build -t gradio-app .
docker run -p 7860:7860 gradio-app
4- Una vez ejecutado, acceder al link que saldra en la pantalla de Docker que seria este: http://localhost:7860/
5- Dentro de la app cargar el archivo df_combined_predict.csv que lo puedes descargar de la carpeta "data" de este mismo github
6- Por ultimo pulsar predict y esperar los resultados
