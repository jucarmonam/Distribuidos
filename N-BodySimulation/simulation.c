#include<stdlib.h>
#include<stdio.h>
#include<math.h>

#define R_ARGS 1
#define DIMENSIONS 3

/*Variable para el número de particulas*/
int Nparticles;

/*Crear las tres matrices para posiciones, nuevas posiciones y velocidades de cada particulas*/
int *positions, *newPositions, *speeds;

/*Arreglo para las masa de cada particula*/
int *masses;


int main(int argc,char* argv[])
{
	int step,burst;
	int nparticle = 8192; 
	/* MUST be a nice power of two for simplicity */
	int nparticle2 = nparticle/2;
	int nstep = 100;
	int nburst = 20; 
	/* MUST divide the value of nstep without remainder */
	int nthread = 64; 
	/* chosen for ATI Radeon HD 5870 */
	float dt = 0.0001;
	float eps = 0.0001;

	float ax,ay,az,dx,dy,dz;

	/*Verificar que la cantidad de argumentos sea la correcta*/
    if ((argc - 1) < R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        exit(1);
    }

	Nparticles = atoi(*(argv + 1));

	masses = malloc(Nparticles * sizeof(float));
	positions = (float *)malloc(Nparticles * DIMENSIONS * sizeof(float));
	newPositions = (float *)malloc(Nparticles * DIMENSIONS * sizeof(float));
	speeds = (float *)malloc(Nparticles * DIMENSIONS * sizeof(float));

	srand(time(0));
	for(int i=0; i<Nparticles; i++) { 
      *(positions + (i * DIMENSIONS + 0)) = rand() % 20;
      *(positions + (i * DIMENSIONS + 1)) = rand() % 20;
	  *(positions + (i * DIMENSIONS + 2)) = rand() % 20;
	}

	for(int i=0; i<Nparticles; i++) { 
		ax=0.0;
      	ay=0.0;
      	az=0.0;
		for(int j=0; j<Nparticles; j++) { 
			/* recorremos sobre todas las Nparticles "j" */
			dx = *(positions + (j * DIMENSIONS + 0)) - *(positions + (i * DIMENSIONS + 0));        
			dy = *(positions + (j * DIMENSIONS + 1)) - *(positions + (i * DIMENSIONS + 1));  
			dz = *(positions + (j * DIMENSIONS + 2)) - *(positions + (i * DIMENSIONS + 2));  

			float invr = 1.0/sqrt(dx*dx + dy*dy + dz*dz + eps);
			float invr3 = invr*invr*invr;
			float f=masses[j]*invr3;

			/* acumulamos la aceleracion de la atraccion gravitacional*/
			ax += f*dx; 
			ay += f*dy;
			az += f*dx;
     	}

		/* actualizamos posicion de particula "i" */
		*(newPositions + (i * DIMENSIONS + 0)) = *(positions + (i * DIMENSIONS + 0)) + dt * *(speeds + (i * DIMENSIONS + 0)) + 0.5*dt*dt*ax; 
		*(newPositions + (i * DIMENSIONS + 1)) = *(positions + (i * DIMENSIONS + 1)) + dt * *(speeds + (i * DIMENSIONS + 1)) + 0.5*dt*dt*ay;
		*(newPositions + (i * DIMENSIONS + 2)) = *(positions + (i * DIMENSIONS + 2)) + dt * *(speeds + (i * DIMENSIONS + 2)) + 0.5*dt*dt*az;

		/* actualizamos velocidad de particula "i" */
		*(speeds + (i * DIMENSIONS + 0)) += dt * ax; 
		*(speeds + (i * DIMENSIONS + 1)) += dt * ay;
		*(speeds + (i * DIMENSIONS + 2)) += dt * az;
	}

	for(int i=0;i<Nparticles;i++) { 
		/* actulizamos las posiciones en los arreglos originales */
		*(positions + (i * DIMENSIONS + 0)) = *(newPositions + (i * DIMENSIONS + 0));
		*(positions + (i * DIMENSIONS + 1)) = *(newPositions + (i * DIMENSIONS + 1));
		*(positions + (i * DIMENSIONS + 2)) = *(newPositions + (i * DIMENSIONS + 2));
	}

	return 0;
}