#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <random>

#define R_ARGS 3
#define DIMENSIONS 3
//Verificar que el tamaño del PAD sea óptimo
#define PAD 16

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

__global__ void calculateNewPosition(particle *particles, particle *newParticles, int nParticles, double softening, double gravityConstant, double dt, int nThreads){
    //Obtener el id del hilo
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    double ax,ay,dx,dy;

    int startPos = (thread_id < (nParticles) % nThreads) ? ((nParticles) / nThreads) * thread_id + thread_id : ((nParticles) / nThreads) * thread_id + (nParticles) % nThreads;
    int endPos = (thread_id < (nParticles) % nThreads) ? startPos + ((nParticles) / nThreads) : startPos + ((nParticles) / nThreads) - 1;

    for (; startPos <= endPos; startPos++){
        ax=0.0;
        ay=0.0;
        for (int j = 0; j < nParticles; j++)
        {
            printf("particle %d \n", startPos);
            //recorremos sobre todas las Nparticles "j"
            dx = particles[j].pos_x - particles[startPos].pos_x;        
            dy = particles[j].pos_y - particles[startPos].pos_y;

            //matrix that stores 1/r^3 for all particle pairwise particle separations 
            double inv_r3 = sqrt(dx*dx + dy*dy + softening*softening);
            inv_r3 = 1/(pow(inv_r3, 2));

            ax = gravityConstant * (dx * inv_r3) * particles[j].mass;
            ay = gravityConstant * (dy * inv_r3) * particles[j].mass;

            if(startPos != j){
                printf("PosX: %f \n",particles[startPos].pos_x);
                printf("dt: %f \n",dt);
                printf("VelX: %f \n",newParticles[startPos].vel_x);
                printf("ax: %f \n",ax);
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
    /*Variable para la constante de gravedad*/
    double gravityConstant = 1.0;
    /*Variable para el softening*/
    double softening = 0.1;
    /*Variable para el dt*/
    double dt = 0.01;
    //Variables para el tamano de la pantalla
    auto const screen_width = 1280;
    auto const screen_height = 720;
    /*Variable que mide el numero de fps*/
    double fps;
    /*Variable que mide el tiempo actual*/
    double currentTime;
    /*Variable para el número de particulas*/
    int nParticles = 0;
    /*Variable para el número de hilos*/
    int nThreads = 0;
    /*Variable para el número de bloques*/
    int nBlocks = 0;
    /*Crear la matriz de particulas*/
    particle *particles;
    /*Crear la matriz del device*/
    particle *d_Particles;
    /*Crear la matriz resultantes*/
    particle *rParticles;
    /*Crear la matriz resultante del device*/
    particle *d_rParticles;
    /*Crear variable para el sizeof int*/
    int size = sizeof(particle);
    //Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

	//Verificar que la cantidad de argumentos sea la correcta
    if ((argc - 1) < R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        exit(1);
    }

    nParticles = atoi(*(argv + 1));
    nBlocks = atoi(*(argv + 2));
    nThreads = atoi(*(argv + 3));

    /*Verificar que el número de hilos sea válido*/
    if (nThreads <= 0 || nBlocks <= 0)
    {
        printf("El número de hilos o de bloques no es válido \n");
        exit(1);
    }

    if (nParticles <= 0)
    {
        printf("El numero de particulas debe ser aunquesea 1\n");
        exit(1);
    }

    /*Crear cada matriz de particulas segun el tamaño*/
    particles = (particle *)malloc(nParticles * size);
    rParticles = (particle *)malloc(nParticles * size);

    if (particles == NULL || rParticles == NULL)
    {
        printf("Error al crear las matrices, problema con malloc \n");
        exit(1);
    }

    //Inicializamos las posiciones, velocidades y masa de las particulas
    for (unsigned int i = 0; i < nParticles; i++)
	{
		int x = getRandomInt(0, screen_width), y = getRandomInt(0, screen_height);
		particles[i].pos_x = x;
		particles[i].pos_y = y;
		particles[i].vel_x = 0.0;
		particles[i].vel_y = 0.0;
		particles[i].mass = 1000; 
	}

    rParticles = particles;

    /*Logica de la simulacion*/
    /*Crear las matrices resultantes*/
    err = cudaMalloc((void **)&d_Particles, nParticles * size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_Particles (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy inputs to device
    err = cudaMemcpy(d_Particles, particles, nParticles * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy particles from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /******************************************/

    /*Crear la matriz de particulas con los resultados*/
    err = cudaMalloc((void **)&d_rParticles, nParticles * size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_rParticles (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Cargar y renderizar pantalla
    sf::RenderWindow window(sf::VideoMode(screen_width, screen_height), "N-Body");

    //Establecemos un limite en el numero de fps
    window.setFramerateLimit(60);

    //3 Arreglos con los circulos usados para dibujar una particula
    sf::CircleShape p_mid(2);
    sf::CircleShape p_eff(4);
	sf::CircleShape p_eff2(8);

	sf::Clock clock;
    // run the program as long as the window is open
    while (window.isOpen())
    {
		//Compute the frame rate
		currentTime = clock.restart().asSeconds();
		fps = 1.0 / currentTime;

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
		for (unsigned int i = 0; i < nParticles; i++)
		{
			double green = getRandomInt(25, 75);

			p_eff.setFillColor(sf::Color(255, 75, 0, green));
			p_eff2.setFillColor(sf::Color(255, 75, 0, green / 2));

            particles[i] = rParticles[i];

            /*
            std::cout << "Particle: " << i << std::endl;
            std::cout << "PosX: " << particles[i].pos_x << std::endl;
            std::cout << "PosY: " << particles[i].pos_y << std::endl;
            std::cout << "VelX: " << particles[i].vel_x << std::endl;
            std::cout << "VelY: " << particles[i].vel_y << std::endl;
            */
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

        // Copy inputs to device
        err = cudaMemcpy(d_Particles, particles, nParticles * size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy particles from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //Calcular las posiciones de las particulas
        /*Paralelizar el algoritmo*/
        calculateNewPosition<<<nBlocks, nThreads>>>(d_Particles, d_rParticles, nParticles, softening, gravityConstant, dt, nBlocks * nThreads);

        cudaDeviceSynchronize();

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch calculateNewPosition kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy result back to host
        err = cudaMemcpy(rParticles, d_rParticles, nParticles * size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy rParticles from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // end the current frame
        window.display();
    }

    /*Liberar memoria*/
    free(particles);
    free(rParticles);
    cudaFree(d_Particles);
    cudaFree(d_rParticles);
    return 0;
}