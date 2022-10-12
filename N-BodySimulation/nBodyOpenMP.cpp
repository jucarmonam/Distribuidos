#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <random>
#include "omp.h"

#define R_ARGS 2
#define DIMENSIONS 3
//Verificar que el tamaño del PAD sea óptimo
#define PAD 16

/*Variable para el número de particulas*/
int Nparticles;
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

particle* particles;
particle* newParticles;

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

void calculateNewPosition(){
    //Paralelizar el algoritmo
    #pragma omp parallel num_threads(nThreads)
    {
        //Obtener el id del hilo
        int thread_id = omp_get_thread_num();
        double ax,ay,az,dx,dy,dz;

        int startPos = (thread_id < (Nparticles) % nThreads) ? ((Nparticles) / nThreads) * thread_id + thread_id : ((Nparticles) / nThreads) * thread_id + (Nparticles) % nThreads;
        int endPos = (thread_id < (Nparticles) % nThreads) ? startPos + ((Nparticles) / nThreads) : startPos + ((Nparticles) / nThreads) - 1;

        for (startPos; startPos <= endPos; startPos++){
            ax=0.0;
            ay=0.0;
            for (int j = 0; j < Nparticles; j++)
            {
                //recorremos sobre todas las Nparticles "j"
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
}

int main(int argc,char* argv[])
{
    double ax,ay,az,dx,dy,dz;

	//Verificar que la cantidad de argumentos sea la correcta
    if ((argc - 1) < R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        exit(1);
    }

    Nparticles = atoi(*(argv + 1));
    nThreads = atoi(*(argv + 2));

    /*Verificar que el número de hilos sea válido*/
    if (nThreads < 0)
    {
        printf("El número de hilos ingresado no es válido \n");
        exit(1);
    }

    if (Nparticles < 0)
    {
        printf("El numero de particulas debe ser aunquesea 1\n");
        exit(1);
    }

    //Cargar y renderizar pantalla
    sf::RenderWindow window(sf::VideoMode(screen_width, screen_height), "N-Body");

    //Establecemos un limite en el numero de fps
    window.setFramerateLimit(60);

    //3 Arreglos con los circulos usados para dibujar una particula
    sf::CircleShape p_mid(2);
    sf::CircleShape p_eff(4);
	sf::CircleShape p_eff2(8);

    //Arreglo que contiene las particulas a simular
    particles = (particle*)malloc(Nparticles * sizeof(particle));
    newParticles = (particle*)malloc(Nparticles * sizeof(particle));

    for (unsigned int i = 0; i < Nparticles; i++)
	{
		int x = getRandomInt(0, screen_width), y = getRandomInt(0, screen_height);
		particles[i].pos_x = x;
		particles[i].pos_y = y;
		particles[i].vel_x = 0.0;
		particles[i].vel_y = 0.0;
		//if(i == 0){
			//particles[i].mass = 10000;
		//}else{
		particles[i].mass = 1000; 
		//}
	}

    newParticles = particles;

	sf::Clock clock;
	double fps;
    // run the program as long as the window is open
    while (window.isOpen())
    {
		//Compute the frame rate
		double currentTime = clock.restart().asSeconds();
		double fps = 1.0 / currentTime;

		std::cout << "Fps: " << fps << std::endl;

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
		for (unsigned int i = 0; i < Nparticles; i++)
		{
			double green = getRandomInt(25, 75);

			p_eff.setFillColor(sf::Color(255, 75, 0, green));
			p_eff2.setFillColor(sf::Color(255, 75, 0, green / 2));

            particles[i] = newParticles[i];

			/*
			if(i == 0){
				p_mid.setFillColor(sf::Color(255, 75, 0));
			}else{
				p_mid.setFillColor(sf::Color(255, 255, 255));
			}
			*/

			p_mid.setPosition(particles[i].pos_x, particles[i].pos_y);
			p_eff.setPosition(particles[i].pos_x, particles[i].pos_y);
			p_eff2.setPosition(particles[i].pos_x, particles[i].pos_y);

			sf::Vector2f normalised = normalise(sf::Vector2f(particles[i].pos_x, particles[i].pos_y));

			window.draw(p_eff2);
			window.draw(p_eff);
			window.draw(p_mid);
		}

        //Calcular las posiciones de las particulas
        calculateNewPosition();

        // end the current frame
        window.display();
    }

    free(particles);
    return 0;
}