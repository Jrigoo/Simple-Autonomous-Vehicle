# Autonomous Driving

En este repositorio encontrarás todo el código para la realización de un carro autonomo utilizando un GPS y IMU. En este ejemplo, el vehículo es capaz de leer data de una ruta preestablecida en un .txt , interpretarla y moverse de forma autonoma por la misma

Se utilizó el simulador Morai Sim. Recomiendo que puedas leer un poco más acerca del simulador [acá](https://morai-sim--drive-user-manual--en-22-r2.scrollhelp.site/msdume2/installation-and-setup). El mismo permite la conexión del código al vehículo utilizando UDP Sockets.

## Paquetes

Se recomienda utilizar un ambiente virtual para descargar los paquetes. Para esto se puede hacer con el siguiente comando:

```
py -m venv env
```

Una vez creado el ambiente virtual, accedes al mismo dependiendo si tu terminal es CMD o Bash

```
# CMD
env/Scripts/activate

# Bash
cd env/Scripts && . activate && cd ../..
```

Una vez dentro del ambiente virtual. Descargas los paquetes necesarios de la siguiente manera:

```
pip install -r requirements.txt
```

## Estructura

La estructura de los folders es la siguiente:

```
 ./
 |-- network
	 |-- receiver
	 |-- sender
|-- scripts
	|-- functions.py
	|-- pid.py
	|-- sensors.py
	|-- vision.py
	|-- waypoint.py
|-- tests
|-- params.json
|-- main.py

```

1.  La carpeta `network` posee diferentes clases para facilitar la comunicación entre el simulador y el código de python:
    - `receiver` posee las clases necesarias para mover el vehículo mediante diferentes comandos
    - `sender` posee las clases necesarias para obtener información acerca del estado del vehículo
2.  La carpeta `scripts` posee diferentes clases y archivos que permiten el control autonomo del vehículo por la ruta predefinida :
    - `lib` contiene archvios que permiten la conectividad con los sensores del vehículo
    - `routes` contiene archvios con rutas predefinidas
    - `functions.py` contiene varias funciones utiles y necesarias para el algoritmo
    - `pid.py` contiene una clase que permite el uso de un [controlador PID](https://www.omega.co.uk/prodinfo/pid-controllers.html#:~:text=A%20PID%20controller%20is%20an,most%20accurate%20and%20stable%20controller.). Para esta primera versión no se utiliza
    - `sensors.py` contiene una clase que inicializa todos los sensores que serán utilizados a lo largo del proyecto y permite mediante diferentes métodos obtener data de los mismos
    - `vision.py` contiene una clase que permite el reoconocimiento de las lineas de carril de la calle. Sin embargo, para esta primera versión no se utilizó, puesto que el modelo le faltan mejoras
    - `waypoint.py` contiene una clase que permite la autonomía vehícular. El algoritmo es similar a un controlador [Pure Pursuit](https://la.mathworks.com/help/nav/ug/pure-pursuit-controller.html) utilizado para controlar el timón de un carro de un punto a otro.
3.  La carpeta `tests` contiene varios archivos en los cuales se realizaron diferentes pruebas y versiones
4.  `main.py` el archivo principal que corre el código de python
5.  `params.json` mediante este archivo se configuran los puertos y ips para la conexión entre el simulador y el código de python
