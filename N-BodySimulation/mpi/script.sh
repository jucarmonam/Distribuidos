#!/bin/bash
echo "------------------------------------------------"
echo "Computación paralela y distribuida - MPI"
echo "------------------------------------------------"
echo "Compilando el programa ..."
sudo mpiCC nBodyMPI.cpp -o nBodyMPI -lsfml-graphics -lsfml-window -lsfml-system
echo "Compilación terminada, realizando pruebas ..."
for res in {600,1000,2000}
do
    for ((c=1; c<=32; c*=2))
    do
        mpirun -np $c --hostfile mpi_hosts ./nBodyMPI $res >> /home/juarodriguezc/results.txt
    done
done


printf "\n Pruebas terminadas, consulte el archivo 'meansMPI.csv' para un resumen \n "