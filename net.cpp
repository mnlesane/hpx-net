#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

//TODO: find std/boost equivalents of reinvented wheels

double rnd()
{
	return ( (double)rand() * ( 1 - 0 ) ) / (double)RAND_MAX + 0;
}
float f(float x)
{
	return tanh(x);
}
float df(float x)
{
	return 1.0-pow(x,2.0);
}
float error(float target, float result)
{
	return 0.5*pow(target-result,2.0);
}
float derror(float target,float result)
{
	return result-target;
}
int int_in_vector(std::vector<int> v, int f)
{
	for(int i = 0; i < v.size(); i++)
	{
		//std::cout << f << " in " << v[i] << "?\n";
		if(v[i]==f) return i;
	}
	return -1;
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

void init()
{
	unsigned int seed;
	FILE* urandom = fopen("/dev/urandom", "r");
	fread(&seed, sizeof(int), 1, urandom);
	fclose(urandom);
	srand(seed);
}

class neuron
{
	public:

	std::vector<int> activation_ids;
	std::vector<float> roots;

	std::vector<float> weights;
	std::vector<float> last_change;

	float value, row, bias;

	int id;

	neuron(float row, float bias, int id)
	{
		this->value = 0.0;
		this->row = row;
		this->bias = bias;
		if(bias) this->value = 1.0;
		this->id = id;
	}
};

float productsum(std::vector<float> roots, std::vector<float> weights)
{
	float out = 0;
	for(int i = 0; i < roots.size(); i++)
	{
		out += roots[i]*weights[i];
		//std::cout << roots[i] << " x " << weights[i] << "\n";
	}
	return out;
}

class network
{
	public:
	std::vector<neuron> net;

	void setSensors(float vals[])
	{
		for(int i = 0; i < sizeof(vals)/sizeof(float); i++)
		{
			net[i].value = vals[i];
		}
	}
	void setSensors(std::vector<float> vals)
	{
		for(int i = 0; i < vals.size(); i++)
		{
			net[i].value = vals[i];
		}
	}
	void run() //Forward Propagation
	{

		float ids[net.size()]; //substitute this with a future array/vector later

		for(int it = 0; it <= net[net.size()-1].row; it++)
		{
			std::vector<float> roots;
			for(int i = 0; i < net.size(); i++) //Learner node
			{
				if(net[i].row==it-1)
				{
					roots.push_back(net[i].value);	
					continue;
				}			
				if(net[i].bias==1||net[i].row==0||(net[i].row!=it)) continue;
				ids[i] = productsum(roots,net[i].weights); //future goes here
			}
			for(int i = 0; i < net.size(); i++)
			{
				if(net[i].bias==1||net[i].row==0||net[i].row!=it) continue;
				net[i].value = ids[i]; //wait goes here
				net[i].value = f(net[i].value);
			}
		}
	}

	std::vector<float> reverse(std::vector<float> vals)
	{
		std::vector<float> result;
		for(int i = vals.size()-1; i >= 0; i--)
		{
			result.push_back(vals[i]);
		}
		return result;
	}

	float correct_serial(std::vector<float> v,float m,float n)
	{
		std::vector<float> vi = v;
		v = this->reverse(v);
		float deltas[net.size()];

		float error;

		for(int i = net.size()-1; i >= 0; i--)
		{
			if(net[i].row == 0 || net[i].bias == 1) continue;
			if(net[i].row == net[net.size()-1].row) //output deltas
			{
				float target = v[0];
				v.erase(v.begin(),v.begin()+1);

				error = target - net[i].value;
				float dfunc = df(net[i].value);
				deltas[i] = error*dfunc;

				for(int j = 0; j < net[i].weights.size(); j++) //previous layer
				{
					int pos = net[i].activation_ids[j];
					float change = deltas[i]*net[pos].value;
					float change2 = m*change + n*net[i].last_change[j];
					net[i].weights[j] += change2;
					net[i].last_change[j] = change;
				}
			}	
			else //hidden deltas
			{
				error = 0;
				for(int j = 0; j < net.size(); j++)
				{
					int pos = int_in_vector(net[j].activation_ids,i);
					if(pos >= 0)
					{
						error += deltas[j]*net[j].weights[pos];
					}
				}
				deltas[i] = error*df(net[i].value);

				for(int j = 0; j < net[i].weights.size(); j++)
				{
					neuron val = net[j];
					float change = deltas[i]*net[j].value;
					int pos = net[i].activation_ids[j];

					float change2 = m*change + n*net[i].last_change[j];
					net[i].weights[j] += change2;
					net[i].last_change[j] = change;
				}
			}
		}
		error = 0;
		float out_index = 0;

		for(int i = 0; i < net.size(); i++)
		{
			if(net[i].row==net[net.size()-1].row)
			{
				error += 0.5*pow(vi[out_index]-net[i].value,2);
				out_index++;
			}
		}
		return error;
	}
	network(int in, int hidden_rows, int hidden_cols, int out, int bias = 0)
	{
		int rowcount = 0;
		std::vector<neuron> net;
		int netcount = 0;

		for(int i = 0; i < in; i++)
		{
			net.push_back(neuron(rowcount,0,netcount));
			netcount++;
		}
		net.push_back(neuron(0,1,netcount));
		rowcount++;
		netcount++;

		for(int i = 0; i < hidden_rows; i++) //"Rows" in Hidden Layers
		{
			for(int j = 0; j < hidden_cols; j++) //"Columns" in Hidden Layers
			{
				//Nodes for hidden layers
				net.push_back(neuron(rowcount,0,netcount));
				for(int k = 0; k < net.size(); k++)
				{
					if(net[k].row == rowcount-1)
					{
						net[netcount].activation_ids.push_back(k);
//						net[netcount].weights.push_back(0);//rnd()-0.5);
						net[netcount].weights.push_back(rnd()-0.5);
						net[netcount].last_change.push_back(0);
					}
				}
				netcount++;
			}

			//Bias node for each hidden layer
			if(bias == 1)
			{
				net.push_back(neuron(rowcount,1,netcount));
				netcount++;
			}

			rowcount++;
		}

		//Output node/layer
		for(int i = 0; i < out; i++)
		{
			net.push_back(neuron(rowcount,0,netcount));
			for(int k = 0; k < net.size(); k++)
			{
				if(net[k].row == rowcount-1)
				{
					net[netcount].activation_ids.push_back(k);
//					net[netcount].weights.push_back(0);//rnd()-0.5);
					net[netcount].weights.push_back(rnd()-0.5);
					net[netcount].last_change.push_back(0);
				}
			}
			netcount++;
		}
		this->net = net;
	}
/*
*/
};

std::vector<float> to_vector(float x[],int s)
{
	std::vector<float> out;
	for(int i = 0; i < s; i++) out.push_back(x[i]);
	return out;
}

int main()
{
	network n(2,1,2,1,1);

//XOR
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

	for(int i = 0; i < 4000; i++)
	{
		std::cout << i << " ";
		int s = i%(sizeof(tests)/sizeof(tests[0]));

		std::vector<float> sensor = to_vector(tests[s],sizeof(tests[s])/sizeof(float));
		std::vector<float> target = to_vector(targets[s],sizeof(targets[s])/sizeof(float));
		n.setSensors(sensor);
		n.run();

		std::cout << "(";
		for(int it = 0; it < sizeof(tests[0])/sizeof(tests[0][0]); it++)
		{
			std::cout << sensor[it];
			if(it < sizeof(tests[0])/sizeof(tests[0][0])-1) std::cout << " ";
		}

		std::cout << ") = ";

		int valid = 1;

		for(int it = 0; it < n.net.size(); it++)
		{
			int counter = 0;
			if(n.net[it].row == n.net[n.net.size()-1].row)
			{
				float r = n.net[it].value;
				if(r < 0) r = 0;
				r = round(r);
				std::cout << r << " ";
				if(round(n.net[it].value) != targets[s][counter]) valid = 0;
				else counter++;
			}
		}

		std::cout << "(";
		for(int it = 0; it < sizeof(targets[0])/sizeof(targets[0][0]); it++)
		{
			std::cout << target[it];
			if(it < sizeof(targets[0])/sizeof(targets[0][0])-1) std::cout << " ";
		}
		std::cout << ")";
		if(valid) std::cout << "\033[32mCorrect!\033[0m  \t";
		else std::cout << "\033[31mIncorrect!\033[0m\t";

		float error = n.correct_serial(target,0.05,0.01);
		std::cout << error;
		std::cout << "\n";
	}
	return 0;
}
