#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <random>

#define R_ARGS 1
#define DIMENSIONS 3
#define DT 0.01

/*Variable para el número de particulas*/
int Nparticles;
float GravityConstant;

float softening = 0.1;

struct particle
{
	float pos_x = 0, pos_y = 0, vel_x = 0, vel_y = 0, mass = 0;
};

float dist(sf::Vector2f dif)
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
	float distance = dist(dif);
	return sf::Vector2f(dif.x / distance, dif.y / distance);
}

particle* calculateNewPosition(particle* particles){
	float ax,ay,az,dx,dy,dz;
	unsigned int i,j;
	for (i = 0; i < Nparticles; i++){
		ax=0.0;
      	ay=0.0;
      	//az=0.0;
		for (j = 0; j < Nparticles; j++)
		{
			//recorremos sobre todas las Nparticles "j"
			dx = particles[j].pos_x - particles[i].pos_x;        
			dy = particles[j].pos_y - particles[i].pos_y;  

			//dz = *(positions + (j * DIMENSIONS + 2)) - *(positions + (i * DIMENSIONS + 2));

			//matrix that stores 1/r^3 for all particle pairwise particle separations 
			float inv_r3 = sqrt(dx*dx + dy*dy + softening*softening);
			inv_r3 = 1/(pow(inv_r3, 2));

			ax = GravityConstant * (dx * inv_r3) * particles[j].mass;
			ay = GravityConstant * (dy * inv_r3) * particles[j].mass;
			//az = GravityConstant * (dz * inv_r3) @ mass

			if(i != j){
				//actualizamos posicion de particula "i"
				particles[i].pos_x = particles[i].pos_x + DT * particles[i].vel_x + 0.5 * pow(DT,2) * ax; 
				particles[i].pos_y = particles[i].pos_y + DT * particles[i].vel_y + 0.5 * pow(DT,2) * ay;
				//particles[j].pos_z = *(positions + (i * DIMENSIONS + 2)) + dt * *(speeds + (i * DIMENSIONS + 2)) + 0.5*dt*dt*az;

				//actualizamos velocidad de particula "i"
				particles[i].vel_x += DT * ax; 
				particles[i].vel_y += DT * ay;
				//particles[i].vel_x += DT * az;
			}
		}

	}

	return particles;
}

int main(int argc,char* argv[])
{
    float ax,ay,az,dx,dy,dz;

	//Verificar que la cantidad de argumentos sea la correcta
    if ((argc - 1) < R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        exit(1);
    }

    Nparticles = atoi(*(argv + 1));

    sf::RenderWindow window(sf::VideoMode(1920, 1080), "N-Body");

    std::cout << "Initializing..." << std::endl;

    window.setFramerateLimit(20);

    auto const screen_width = 1920;
	auto const screen_height = 1000;

	GravityConstant = 1.0;

    unsigned int timer;

    sf::Vector2f mpos;

    sf::CircleShape p_mid(2);
    //sf::CircleShape p_eff(4);
	//sf::CircleShape p_eff2(8);

    particle* particles = (particle*)malloc(Nparticles * sizeof(particle));

    for (unsigned int i = 0; i < Nparticles; i++)
	{
		int x = getRandomInt(0, screen_width), y = getRandomInt(0, screen_height);
		particles[i].pos_x = x;
		particles[i].pos_y = y;
		particles[i].mass = 10;
	}

    timer = 400;

	float fps;
    // run the program as long as the window is open
    while (window.isOpen())
    {
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
			//float green = getRandomInt(25, 75);

			//p_eff.setFillColor(sf::Color(255, 75, 0, green));
			//p_eff2.setFillColor(sf::Color(255, 75, 0, green / 2));

			p_mid.setPosition(particles[i].pos_x, particles[i].pos_y);
			//p_eff.setPosition(particles[i].pos_x, particles[i].pos_y);
			//p_eff2.setPosition(particles[i].pos_x, particles[i].pos_y);

			//particles[i].pos_y += particles[i].vel_y;
			//particles[i].pos_x += particles[i].vel_x;

			//sf::Vector2f dif = sf::Vector2f(particles[i].pos_x, particles[i].pos_y) - sf::Vector2f(mpos);
			sf::Vector2f normalised = normalise(sf::Vector2f(particles[i].pos_x, particles[i].pos_y));

			//particles[i].vel_x *= 0.999;
			//particles[i].vel_y *= 0.999;
			//particles[i].vel_x -= normalised.x / 20.0f;
			//particles[i].vel_y -= normalised.y / 20.0f;

			//window.draw(p_eff2);
			//window.draw(p_eff);
			window.draw(p_mid);
		}

		particles = calculateNewPosition(particles);

        // end the current frame
        window.display();
        timer++;
    }

    return 0;
}