#include "net2.cpp"
#include <algorithm>
#include <unistd.h>

int GENERATION_CUTOFF = 4;
int GENERATION_SIZE = 12; //generation_cutoff*(generation_cutoff-1)
int MUTATION_RATE = 10;

class cmpob
{
	public:
	network n;
	float fitness;

	cmpob()
	{
	}
};

bool fitness_sort(cmpob a, cmpob b)
{
	return (a.fitness < b.fitness);
}

network crossover(network x,network y)
{
	for(int i = 1; i < x.rows.size(); i++)
	{
		for(int j = 0; j < x.rows[i].contents.size(); j++)
		{
			if(x.rows[i].contents[j].bias == 1) continue;
			for(int k = 0; k < x.rows[i].contents[j].weights.size(); k++)
			{
				if(floor(rnd()*MUTATION_RATE)==0) x.rows[i].contents[j].weights[k /*uhhh -- remember how this works here*/] = rnd(); //mutation
				else
				{
					if(round(rnd())==0) //x-dominant
						continue;
					else //y-dominant
						x.rows[i].contents[j].weights[k/*uhhh*/] = y.rows[i].contents[j].weights[k];
				}
			}
		}
	}
	return x;
}

std::string _bin(int x)
{
	if(x==0) return "0";
else	if(x==1) return "1";

	std::string out;

	int largest = floor(log(x)/log(2));

	for(int i = largest; i >= 0; i--)
	{
		float power = pow(2,i);

		if(x - power >= 0)
		{
			out.append(std::string("1"));
			x -= power;
		}
		else
		{
			out.append(std::string("0"));
		}
	}

	return out;
}

std::string left_pad(std::string s, int len, std::string padding)
{
	while(s.length() < len)
	{
		s = padding + s;
	}
	return s;
}

std::vector<float> str_split_float(std::string s)
{
	std::vector<float> result;
	for(int i = 0; i < s.length(); i++)
	{
		result.push_back
		(
			(float)(s[i] - '0')
		);
	}
	return result;
}

std::vector<cmpob> array_reverse(std::vector<cmpob> li)
{
	std::vector<cmpob> out;
	for(int i = li.size()-1; i >= 0; i--)
		out.push_back(li[i]);

	return out;
}
float assess_fitness(std::vector<network> &generation)
{
	float tests[][2] =
	{
		{0.0,0.0},
		{0.0,1.0},
		{1.0,0.0},
		{1.0,1.0}
	};
	float targets[][1] =
	{
		{0.0},
		{1.0},
		{1.0},
		{0.0}
	};

	float error[generation.size()];
	
	for(int i = 0; i < generation.size(); i++) error[i] = 0;
	
	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < generation.size(); j++)
		{
			generation[j].setSensors(tests[i]);
			generation[j].run();
			error[j] += (1.0 - generation[j].correct_serial(to_vector(targets[i],4),.05,.01))/generation.size();
		}
	}
	
	std::vector<cmpob> sortable;
	
	float average_error = 0;
	
	for(int i = 0; i < generation.size(); i++)
	{
		cmpob c;
		c.n = generation[i];
		c.fitness = error[i];

		sortable.push_back(c);
		average_error += error[i] / ((float)GENERATION_SIZE);
	}

	std::sort(sortable.begin(), sortable.end(), fitness_sort);

	sortable = array_reverse(sortable);
	
	//echo "Network Fitness:\n";
	//foreach($sortable as $i=>$v)
	//	echo "Network $i: ".$sortable[$i][1]."\n"
		;

	std::vector<network> new_generation;
	for(int i = 0; i < GENERATION_CUTOFF; i++)
	{
		new_generation.push_back(sortable[i].n);
	}
	generation = new_generation;
	
	return average_error;
}

float evolve(std::vector<network> &generation)
{
	float generational_error = assess_fitness(generation);
	std::vector<network> new_generation;
	
	for(int i = 0; i < generation.size(); i++)
		for(int j = 0; j < generation.size(); j++)
			if(i!=j && i < j)
			{
				network newnet = crossover(generation[i],generation[j]);
				new_generation.push_back(newnet);
				new_generation.push_back(generation[i]);
			}
	
	generation = new_generation;

	return generational_error;
}

std::string implode(std::string delimiter, std::vector<std::string> array){
	std::string out = "";
	for(int i = 0; i < array.size(); i++)
	{
		out.append(array[i]);
		if(i<array.size()-1) out.append(delimiter);
	}
	return out;
}


int main()
{

	float tests[][2] =
	{
		{0.0,0.0},
		{0.0,1.0},
		{1.0,0.0},
		{1.0,1.0}
	};
	float targets[][1] =
	{
		{0.0},
		{1.0},
		{1.0},
		{0.0}
	};

	std::vector<network> generation;

	for (int i = 0; i < GENERATION_SIZE; i++)
	{
		network n(2,1,2,1,1);
		generation.push_back(n);
	}

	for(int i = 0; i < 100000; i++)
	{
		float o = evolve(generation);
		std::cout << "Generation " << i << " Average Fitness: " << (1-o) << "\n";


		for(int j = 0; j < 4; j++)
		{
			std::vector<float> sensors = to_vector(tests[j],2);
			std::vector<float> target = to_vector(targets[j],1);

			for(int k = 0; k < generation.size(); k++)
			{
				std::cout << "XOR(" << sensors[0] << "," << sensors[1] << ") = ";

				generation[k].setSensors(sensors);
				generation[k].run();
				std::cout << generation[k].rows[2].contents[0].value << " (Neuron " << k << ") - Target " << target[0] << "\n";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
		//usleep(10000);
		system("clear");
	}
	return 0;
}
