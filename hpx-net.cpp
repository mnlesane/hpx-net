#include <iostream>
#include <cstdio>

#include <cstdlib>
#include <vector>
#include <cmath>

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/unwrapped.hpp>

float productsum(std::vector<float>,std::vector<float>);

HPX_PLAIN_ACTION(productsum, ps_action);

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
void init()
{
	unsigned int seed;
	FILE* urandom = fopen("/dev/urandom", "r");
	fread(&seed, sizeof(int), 1, urandom);
	fclose(urandom);
	srand(seed);
}

float productsum(std::vector<float> roots, std::vector<float> weights)
{
	float out = 0;
	for(int i = 0; i < (int)roots.size(); i++)
		out += roots[i]*weights[i];
	return out;
}

hpx::lcos::future<float> future_productsum(std::vector<hpx::lcos::future<float>> roots, std::vector<float> weights)
{
	hpx::lcos::future<float> out = hpx::lcos::make_ready_future((float)0.0);
        for(int i = 0; i < (int)roots.size(); i++)
	{
		hpx::lcos::future<float> add = hpx::lcos::local::dataflow
		(
			hpx::util::unwrapped
			( [] (float a, float b)
			{
				return a*b;
			}
			),roots[i],hpx::lcos::make_ready_future((float)weights[i])
		);
		out = hpx::lcos::local::dataflow
		(
			hpx::util::unwrapped
			( [] (float a, float b)
			{
				return a+b;

			}
			),out,add
		);
	}
	return out;
}

class neuron
{
	public:

	std::vector<float> weights;
	std::vector<float> last_change;
	hpx::lcos::future<float> new_value;

	hpx::lcos::future<float> get_f_future();

	float value, bias, delta;

	neuron(float,int,int);

	void run(std::vector<neuron> roots);
	float get_value();
};
	hpx::lcos::future<float> neuron::get_f_future()
	{
		return hpx::lcos::local::dataflow
		(
			hpx::util::unwrapped
			( [] (float a)
			{
				return f(a);
			}
			),this->new_value
		);
	}

std::vector<float> extract_roots(std::vector<neuron> contents)
{

	std::vector<float> result;
	for(int i = 0; i < (int)contents.size(); i++)
		result.push_back(contents[i].value);
	return result;
}
std::vector<float> future_get_roots(std::vector<neuron> contents)
{
	std::vector<float> result;
	for(int i = 0; i < (int)contents.size(); i++)
		result.push_back(contents[i].get_value());
	return result;
}
std::vector<hpx::lcos::future<float>> extract_future_roots(std::vector<neuron> contents)
{
  std::vector<hpx::lcos::future<float>> out;
  for(int i = 0; i < (int)contents.size(); i++)
  {
    out.push_back(contents[i].get_f_future());
  }
  return out;
}

	neuron::neuron(float bias, int count_activations, int random)
	{
		this->value = 0.0;
		this->bias = bias;
		if(bias)
		{
			this->value = 1.0;
			this->new_value = hpx::lcos::make_ready_future((float)1.0);
		}
		//Activation-related data
		for(int k = 0; k < count_activations; k++)
		{
			this->weights.push_back(rnd()-0.5);
			this->last_change.push_back(0);
		}
	}
	void neuron::run(std::vector<neuron> roots)
	{
		if(this->bias) return;
		this->new_value = hpx::lcos::make_ready_future(productsum(future_get_roots(roots),this->weights));
							       //this->new_value = future_productsum(extract_future_roots(roots),this->weights);
	}
	float neuron::get_value()
	{
	    if(!this->bias)
	      this->value = f(this->new_value.get());
	    return this->value;
	}

class neuron_row
{
	public:

	std::vector<neuron> contents;
	int out;

	neuron_row(std::vector<neuron> init,int out)
	{
		this->contents = init;
		this->out = out;
	}

	neuron_row(int neurons, int out, int bias, int count_activations, int random)
	{
		std::vector<neuron> init;
		for(int i = 0; i < neurons; i++)
			init.push_back(neuron(0,count_activations,random));
	if(bias)	init.push_back(neuron(1,count_activations,random));
		this->contents = init;
		this->out = out;
	}

	void run(std::vector<neuron> roots)
	{
		for(int i = 0; i < (int)this->contents.size(); i++)
			this->contents[i].run(roots);
	}

	void add(neuron x)
	{
		contents.push_back(x);
	}

	int size()
	{
		return this->contents.size();
	}
	void correct(std::vector<float> v, float m, float n,neuron_row prev,neuron_row next)
	{
		float error;
		for(int j = 0; j < (int)this->size(); j++)
	      	{
       			if(this->contents[j].bias == 1) continue;
	       		if((int)this->out) //output deltas
	       		{
	       			float target = v[j];
       				error = target - this->contents[j].get_value();
       				float dfunc = df(this->contents[j].get_value());
       				this->contents[j].delta = error*dfunc;
       			}
       			else //hidden deltas
       			{
       				error = 0;
       				for(int k = 0; k < (int)next.size(); k++)
       					if(next.contents[k].bias) continue;
       					else error += next.contents[k].delta * next.contents[k].weights[j];
       				this->contents[j].delta = error*df(this->contents[j].get_value());
       			}
			for(int k = 0; k < (int)prev.size(); k++) //previous layer
			{
       				float change = this->contents[j].delta * prev.contents[k].get_value();
       				float change2 = m*change + n*this->contents[j].last_change[k];
				this->contents[j].weights[k] += change2;
				this->contents[j].last_change[k] = change;
			}
       		}
	}
};

class network
{
	public:
	std::vector<neuron_row> rows;
	std::vector<hpx::lcos::future<neuron_row>> future_rows;

	void setSensors(float vals[])
	{
		for(int i = 0; i < (int)(sizeof(vals)/sizeof(float)); i++)
		{
			this->rows[0].contents[i].value = vals[i];
			this->rows[0].contents[i].new_value = hpx::lcos::make_ready_future(vals[i]);
		}
	}
	void setSensors(std::vector<float> vals)
	{
		for(int i = 0; i < (int)vals.size(); i++)
		{
			this->rows[0].contents[i].value = vals[i];
			this->rows[0].contents[i].new_value = hpx::lcos::make_ready_future(vals[i]);
		}
	}
	void run() //Forward Propagation
	{
		for(int i = 1; i < (int)this->rows.size(); i++)
			this->rows[i].run(this->rows[i-1].contents);
	}
	std::vector<float> reverse(std::vector<float> vals)
	{
		std::vector<float> result;
		for(int i = (int)vals.size()-1; i >= 0; i--)
			result.push_back(vals[i]);
		return result;
	}
	float correct(std::vector<float> v, float m /*learning_rate*/, float n /*momentum*/)
	{
		std::vector<float> vi = v;
		//v = this->reverse(v);

		float error;

		for(int i = (int)this->rows.size()-1; i >= 1; i--)
		{
			int s = (int)this->rows.size();
			if(s-1 == i)
				this->rows[i].correct(v,m,n,this->rows[i-1],this->rows[i-1]);
			else	this->rows[i].correct(v,m,n,this->rows[i-1],this->rows[i+1]);
		}
		error = 0;
		float out_index = 0;

		for(int i = 0; i < (int)this->rows[this->rows.size()-1].size(); i++)
		{
			error += 0.5*pow(vi[out_index]-this->rows[this->rows.size()-1].contents[i].get_value(),2);
			out_index++;
		}
		return error;
	}
  float correct_serial(std::vector<float> v, float m /*learning_rate*/, float n /*momentum*/)
  {
    std::vector<float> vi = v;
    //v = this->reverse(v);

    float error;

    for(int i = (int)this->rows.size()-1; i >= 1; i--)
      {
	for(int j = 0; j < (int)this->rows[i].size(); j++)
	  {
	    if(this->rows[i].contents[j].bias == 1) continue;
	    if(i == (int)this->rows.size()-1) //output deltas
	      {
		float target = v[0];
		v.erase(v.begin(),v.begin()+1);

		error = target - this->rows[i].contents[j].get_value();

		float dfunc = df(this->rows[i].contents[j].get_value());
		this->rows[i].contents[j].delta = error*dfunc;

		for(int k = 0; k < (int)this->rows[i-1].size(); k++) //previous layer
		  {
		    float change = this->rows[i].contents[j].delta*this->rows[i-1].contents[k].get_value();
		    float change2 = m*change + n*this->rows[i].contents[j].last_change[k];

		    this->rows[i].contents[j].weights[k] += change2;
		    this->rows[i].contents[j].last_change[k] = change;
		  }
	      }
	    else //hidden deltas
	      {
		error = 0;
		for(int k = 0; k < (int)this->rows[i+1].size(); k++)
		  {
		    error += this->rows[i+1].contents[k].delta * this->rows[i+1].contents[k].weights[j];
		  }
		this->rows[i].contents[j].delta = error*df(this->rows[i].contents[j].get_value());

		for(int k = 0; k < (int)this->rows[i].contents[j].weights.size(); k++) //previous layer
		  {
		    float change = this->rows[i].contents[j].delta * this->rows[i-1].contents[k].get_value();
		    float change2 = m*change + n*this->rows[i].contents[j].last_change[k];

		    this->rows[i].contents[j].weights[k] += change2;
		    this->rows[i].contents[j].last_change[k] = change;
		  }
	      }
	  }
      }
    error = 0;
    float out_index = 0;

    for(int i = 0; i < (int)this->rows[this->rows.size()-1].size(); i++)
      {
	error += 0.5*pow(vi[out_index]-this->rows[this->rows.size()-1].contents[i].get_value(),2);
	out_index++;
      }
    return error;
  }
	//TODO: Create network in parallel 
	void init(int in, int hidden_rows, int hidden_cols, int out, int bias = 0)
	{
		std::vector<neuron_row> row;

		//Input Layer
		row.push_back(neuron_row(in,0,1,0,1));

		//Hiden Layers
		for(int i = 0; i < hidden_rows; i++)
			row.push_back(neuron_row(hidden_cols,0,1,row[i].size(),1));

		//Output Layer
		row.push_back(neuron_row(out,1,0,hidden_cols,1));
		this->rows = row;
	}
	network(int in, int hidden_rows, int hidden_cols, int out, int bias = 0)
	{
		this->init(in,hidden_rows,hidden_cols,out,bias);
	}
	void profile()
	{
		std::cout << "\n";
		for(int i = 0; i < (int)this->rows.size(); i++)
		{
			std::cout << "ROW " << i << "\n";
			for(int j = 0; j < (int)this->rows[i].contents.size(); j++)
			{
				std::cout << "\tCOL " << j << " -> VAL " << this->rows[i].contents[j].get_value() << "\n";
				for(int k = 0; k < (int)this->rows[i].contents[j].weights.size(); k++)
				{
					std::cout << "\t\tWEIGHT " << k << ": " << this->rows[i].contents[j].weights[k] << "\n";
				}
			}
		}
	}
};

std::vector<float> to_vector(float x[],int s)
{
	std::vector<float> out;
	for(int i = 0; i < s; i++) out.push_back(x[i]);
	return out;
}

int hpx_main()
{
	int in, hidden_rows, hidden_cols, out, its;
	/*
	std::cin
		>> in
		>> hidden_rows
		>> hidden_cols
		>> out
		>> its;

	network n(in,hidden_rows,hidden_cols,out,1);
	*/
	its = 5000;
	network n(2,1,2,1,5000);
//XOR
		int problem_count = 4;
		int problem_correct = 0;

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

	for(int i = 0; i < its; i++)
	{
		std::cout << i << " ";
		int s = i%(sizeof(tests)/sizeof(tests[0]));

		std::vector<float> sensor = to_vector(tests[s],sizeof(tests[s])/sizeof(float));
		std::vector<float> target = to_vector(targets[s],sizeof(targets[s])/sizeof(float));
		n.setSensors(sensor);
		n.run();

		std::cout << "(";
		for(int it = 0; it < (int)(sizeof(tests[0])/sizeof(tests[0][0])); it++)
		{
			std::cout << sensor[it];
			if(it < (int)(sizeof(tests[0])/sizeof(tests[0][0]))-1) std::cout << " ";
		}

		std::cout << ") = ";

		int valid = 1;

		for(int it = 0; it < (int)n.rows[n.rows.size()-1].contents.size(); it++)
		{
			int counter = 0;
			float r = n.rows[n.rows.size()-1].contents[it].get_value();
			if(r < 0) r = 0;
			r = round(r);
			std::cout << r << " ";
			if(round(n.rows[n.rows.size()-1].contents[it].get_value()) != targets[s][counter]) valid = 0;
			else counter++;
		}

		std::cout << "(";
		for(int it = 0; it < (int)(sizeof(targets[0])/sizeof(targets[0][0])); it++)
		{
			std::cout << target[it];
			if(it < (int)(sizeof(targets[0])/sizeof(targets[0][0]))-1) std::cout << " ";
		}
		std::cout << ")";
		if(valid)
		  {
		    problem_correct++;
		    std::cout << "\033[32mCorrect!\033[0m  \t";
		  }
		else
		  {
		    problem_correct = 0;
		    std::cout << "\033[31mIncorrect!\033[0m\t";
		  }

		float error =
		  //n.correct(target,0.05,0.01);
		  n.correct_serial(target,0.05,0.01);
		std::cout << error;
		std::cout << "\n";

	      	if(problem_count == problem_correct) break;
	}
	return hpx::finalize();
}

int main(int argc, char* argv[])
{
  for(int i = 0; i < 500; i++) hpx::init(argc,argv);
}
