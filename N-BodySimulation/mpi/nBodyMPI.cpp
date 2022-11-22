#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>

#define R_ARGS 1
#define DIMENSIONS 3
//Verificar que el tamaño del PAD sea óptimo
#define PAD 16

/*Variable para el número de particulas*/
int nParticles;
double GravityConstant = 1.0;

double softening = 0.1;
double dt = 0.01;

//Variables para el tamano de la pantalla
auto const screen_width = 1280;
auto const screen_height = 720;

/*Variable para el número de hilos*/
int nThreads = 0;

struct particle
{
	double pos_x, pos_y, vel_x, vel_y;
	int mass;
};

double dist(sf::Vector2f dif)
{
	return sqrt((dif.x * dif.x) + (dif.y * dif.y));
}

std::random_device rng;
std::mt19937 dev(rng());

int getRandomInt(int low, int high)
{
	std::uniform_int_distribution<> var(low, high);
	return var(rng);
}

sf::Vector2f normalise(sf::Vector2f dif)
{
	double distance = dist(dif);
	return sf::Vector2f(dif.x / distance, dif.y / distance);
}

void calculateNewPosition(particle* newParticles, particle* particles, int processId, int numProcs){
    double ax,ay,az,dx,dy,dz;

    int startPos = (processId < (nParticles) % numProcs) ? ((nParticles) / numProcs) * processId + processId : ((nParticles) / numProcs) * processId + (nParticles) % numProcs;
    int endPos = (processId < (nParticles) % numProcs) ? startPos + ((nParticles) / numProcs) : startPos + ((nParticles) / numProcs) - 1;

    for (; startPos <= endPos; startPos++){
        ax=0.0;
        ay=0.0;
        for (int j = 0; j < nParticles; j++)
        {
            //recorremos sobre todas las nParticles "j"
            dx = particles[j].pos_x - particles[startPos].pos_x;        
            dy = particles[j].pos_y - particles[startPos].pos_y;  

            //matrix that stores 1/r^3 for all particle pairwise particle separations 
            double inv_r3 = sqrt(dx*dx + dy*dy + softening*softening);
            inv_r3 = 1/(pow(inv_r3, 2));

            ax = GravityConstant * (dx * inv_r3) * particles[j].mass;
            ay = GravityConstant * (dy * inv_r3) * particles[j].mass;

            if(startPos != j){
                //actualizamos posicion de particula "i"
                newParticles[startPos].pos_x = particles[startPos].pos_x + dt * particles[startPos].vel_x + 0.5 * pow(dt,2) * ax; 
                newParticles[startPos].pos_y = particles[startPos].pos_y + dt * particles[startPos].vel_y + 0.5 * pow(dt,2) * ay;

                //actualizamos velocidad de particula "i"
                newParticles[startPos].vel_x += dt * ax; 
                newParticles[startPos].vel_y += dt * ay;
            }
        }
    }
}

int main(int argc,char* argv[])
{
    double ax,ay,az,dx,dy,dz;
    //Crear la matriz de particulas
    particle* particles;
    //Crear la matriz resultantes
    particle* rPrParticles;
    //Crear la matriz resultante
    particle* newParticles;
    //Declaración de variable para la escritura del archivo
    FILE *fp;
    unsigned int timer;
    //Utilizar openMPI
    int processId, numProcs;

	//Verificar que la cantidad de argumentos sea la correcta
    if ((argc - 1) < R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        exit(1);
    }

    nParticles = atoi(*(argv + 1));

    if (nParticles < 0)
    {
        printf("El numero de particulas debe ser aunquesea 1\n");
        exit(1);
    }

    //3 Arreglos con los circulos usados para dibujar una particula
    sf::CircleShape p_mid(2);
    sf::CircleShape p_eff(4);
	sf::CircleShape p_eff2(8);

    //Arreglo que contiene las particulas a simular
    particles = (particle*)malloc(nParticles * sizeof(particle));
    rPrParticles = (particle*)malloc(nParticles * sizeof(particle));
    newParticles = (particle*)malloc(nParticles * sizeof(particle));

    if (particles == NULL || rPrParticles == NULL || newParticles == NULL)
    {
        printf("Error al crear las matrices, problema con malloc \n");
        exit(1);
    }

    for (unsigned int i = 0; i < nParticles; i++)
	{
		int x = getRandomInt(0, screen_width), y = getRandomInt(0, screen_height);
		particles[i].pos_x = x;
		particles[i].pos_y = y;
		particles[i].vel_x = 0.0;
		particles[i].vel_y = 0.0;
		particles[i].mass = 1000.0; 
	}

    newParticles = particles;

	sf::Clock clock;
    sf::Clock clock2;
    double meanFps;
    timer = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    MPI_Datatype mpi_particle;
    MPI_Type_contiguous(3, MPI_DOUBLE, &mpi_particle);
    MPI_Type_commit(&mpi_particle);

    if(processId == 0){
        //Cargar y renderizar pantalla
        sf::RenderWindow window(sf::VideoMode(screen_width, screen_height), "N-Body");
        //Establecemos un limite en el numero de fps
        window.setFramerateLimit(60);

        // run the program as long as the window is open
        while (window.isOpen()) {
            //Compute the frame rate
            double currentTime = clock.restart().asSeconds();
            double fps = 1.0 / currentTime;

            if(timer > 0){
                meanFps += fps;
            }

            if(clock2.getElapsedTime().asSeconds() > 5.0f){
                window.close();
            }

            // check all the window's events that were triggered since the last iteration of the loop
            sf::Event event;
            while (window.pollEvent(event))
            {
                // "close requested" event: we close the window
                if (event.type == sf::Event::Closed)
                    window.close();
            }

            // clear the window with black color
            window.clear(sf::Color::Black);
            
            // Draw particles
            for (unsigned int i = 0; i < nParticles; i++)
            {
                double green = getRandomInt(25, 75);

                p_eff.setFillColor(sf::Color(255, 75, 0, green));
                p_eff2.setFillColor(sf::Color(255, 75, 0, green / 2));

                particles[i] = newParticles[i];

                p_mid.setPosition(particles[i].pos_x, particles[i].pos_y);
                p_eff.setPosition(particles[i].pos_x, particles[i].pos_y);
                p_eff2.setPosition(particles[i].pos_x, particles[i].pos_y);

                sf::Vector2f normalised = normalise(sf::Vector2f(particles[i].pos_x, particles[i].pos_y));

                window.draw(p_eff2);
                window.draw(p_eff);
                window.draw(p_mid);
            }

            printf("\nLaunching with %i processes", numProcs);

            //Calcular las posiciones de las particulas
            calculateNewPosition(rPrParticles, particles, processId, numProcs);

            //Sincronizar procesos
            MPI_Barrier(MPI_COMM_WORLD); //IMPORTANT 
            MPI_Gather(rPrParticles, sizeof(particle), mpi_particle, newParticles, sizeof(particle), mpi_particle, 0, MPI_COMM_WORLD);

            // end the current frame
            window.display();
            timer++;
        }
    }else{
        //Calcular las posiciones de las particulas
        calculateNewPosition(rPrParticles, particles, processId, numProcs);
    }

    if (processId == 0) {
        meanFps/=timer;

        //Escribir los resultados en un csv
        fp = fopen("./resultados/meansMPI.csv", "a");
        if (fp == NULL)
        {
            printf("Error al abrir el archivo \n");
            exit(1);
        }
        fprintf(fp, "%d,%f\n", nParticles, meanFps);
        fclose(fp);

        //Liberar memoria
        free(particles);
        free(newParticles);
        free(rPrParticles);
    }

    MPI_Finalize();
    return 0;
}