#!/bin/bash
echo "------------------------------------------------"
echo "SIstemas distribuidos"
echo "------------------------------------------------"
echo "Obteniendo información de la GPU del sistema ..."
#Hacer make del deviceQuery modificado
cd deviceQuery
make >/dev/null
gpuInfo=$(./deviceQuery)
#Verificar que se tenga GPU Nvidia
if [ "$gpuInfo" = "-1" ]; then
    echo "La GPU del sistema no es compatible con CUDA"
else
    echo "La GPU del sistema es compatible con CUDA"
    
    mp=$(echo "$gpuInfo" | cut -d "_" -f 1)
    cores=$(echo "$gpuInfo" | cut -d "_" -f 2)
    name=$(echo "$gpuInfo" | cut -d "_" -f 3)

    echo "GPU: $name"
    echo "Se tienen $((mp)) multiprocesadores y $((cores)) cores por multiprocesador"

    echo "Compilando el programa ..."
    cd ../
    nvcc nBodyCUDA.cu -o nBodyCUDA -lsfml-graphics -lsfml-window -lsfml-system
    echo "Compilación terminada, realizando pruebas ..."

    for res in {500,1000,2000}
    do
        printf "\n-----------------------------------------------------------------------------\nPRUEBAS $res p\n------------------------------------------------------------------------------\n"
        for (( i=1; i<=2*$mp; i=i*2 ))
        do
            for (( j=1; j<=2*$cores; j=j*2))
            do
                ./nBodyCUDA $res $i $j
            done
        done
    done

    echo "Pruebas terminadas, consulte el archivo 'meanssCUDA.csv' para un resumen"
fi