[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10412164&assignment_repo_type=AssignmentRepo)
# Tarea 3

Código base para la Tarea 3 del curso Aprendizaje Automático.

En esta tarea se profundizan conceptos de clasificación revisados en el curso. En particular, se explorará la regresión logística, usando distintas técnicas de descenso de gradiente. 

La solución de esta tarea requiere conocer y comprender la teoría asociada a los métodos de clasificación y optimización, para poder enfrentar los problemas típicos que aparecen a la hora de su programación.

El objetivo principal de implementar los métodos es crear una _intuición_ del comportamiento de estos métodos "básicos" y sus 
hiperparámetros, antes de empezar a trabajar con los métodos más complejos en la segunda parte del curso.

Para probar los métodos se utilizará un conjunto de datos de tres especies de pingüinos en el archipiélago Palmer en la Antártica.  Se utilizarán la longitud y profundidad del pico, la longitud de la aleta y la masa de cada espécimen como características de entrada.  Las clases a utilizar en los experimentos de clasificación serán el sexo, la isla de proveniencia, o la especie de los pingüinos.  Este <a href="https://www.kaggle.com/parulpandey/penguin-dataset-the-new-iris/data">conjunto de datos</a> de Allison Horst es apto para experimentaciones iniciales en estas temáticas de clasificación, y es realista en el sentido de que trae defectos por datos faltantes y algunos datos perdidos (llamados outliers).

## Archivos

- penguins_size.csv: datos de entrenamiento
- loadpenguindata.m: función para leer archivo penguins_size.csv
- optimizer.m: clase de optimización, que debe ser completada con los métodos solicitados.
- linreg_main.m: ejemplo de regresión lineal usado para probar algoritmos de descenso de gradiente.
- logreg_main.m: archivo principal que ejecuta punto por punto lo que debe hacerse en la tarea
- logreg_hyp.m: plantilla para implementar la hipótesis
- logreg_loss.m: plantilla para implementar la función de pérdida
- logreg_gradloss.m: plantilla para implementar el gradiente de la función de pérdida
