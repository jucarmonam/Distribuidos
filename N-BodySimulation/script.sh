#!/bin/sh
echo "------------------------------------------------"
echo "Sistemas distribuidos - práctica 1"
echo "------------------------------------------------"
echo "Compilando el programa ..."
#Compilar el programa
sudo g++ -fopenmp nBodyOpenMP.cpp -o nBodyOpenMP -lsfml-graphics -lsfml-window -lsfml-system
echo "Compilación terminada, realizando pruebas ..."
#Pruebas con diferentes parámetros
#Pruebas con 500 particulas
echo "\n------------------------------------------------------------------------------\nPRUEBAS 720p\n------------------------------------------------------------------------------\n">> results.txt
./nBodyOpenMP 500 1
./nBodyOpenMP 500 2
./nBodyOpenMP 500 4
./nBodyOpenMP 500 8
./nBodyOpenMP 500 16
./nBodyOpenMP 500 32

#Pruebas con 1000 particulas
echo "\n------------------------------------------------------------------------------\nPRUEBAS 1080p\n------------------------------------------------------------------------------\n">> results.txt
./nBodyOpenMP 1000 1
./nBodyOpenMP 1000 2
./nBodyOpenMP 1000 4
./nBodyOpenMP 1000 8
./nBodyOpenMP 1000 16
./nBodyOpenMP 1000 32

#Pruebas con 2000 particulas
echo "\n------------------------------------------------------------------------------\nPRUEBAS 4k\n------------------------------------------------------------------------------\n">> results.txt
./nBodyOpenMP 2000 1
./nBodyOpenMP 2000 2
./nBodyOpenMP 2000 4
./nBodyOpenMP 2000 8
./nBodyOpenMP 2000 16
./nBodyOpenMP 2000 32

echo "Pruebas terminadas"