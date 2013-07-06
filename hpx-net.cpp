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

class neuron_row;

int FORWARD_THRESHOLD,
    BACKPROP_THRESHOLD;

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

class neuron
{
	public:

	std::vector<float> weights;
	std::vector<float> last_change;

	float out;

	hpx::lcos::future<float> get_f_future();

	hpx::lcos::future<float> new_value;
	hpx::lcos::future<float> new_delta;
	hpx::lcos::future<float> new_error;

        hpx::lcos::future<hpx::lcos::future<float>> psum;

	float get_value();
	float get_delta();
	float get_error();

	float value, bias, delta, error;

	neuron(float,int,int);

	void run(std::vector<neuron> roots,int);

	void correct(float,int,float,float,neuron_row,neuron_row,int serial);
	void finalize_correct(neuron_row,int,float,float);
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

std::vector<hpx::lcos::future<float>> extract_future_roots(std::vector<neuron> contents)
{
  std::vector<hpx::lcos::future<float>> out;
  for(int i = 0; i < (int)contents.size(); i++)
  {
    out.push_back(contents[i].get_f_future());
  }
  return out;
}

float future_productsum(std::vector<neuron> prev, std::vector<float> weights)
{
	hpx::lcos::future<std::vector<hpx::lcos::future<float>>> future_roots = hpx::async(&extract_future_roots,prev);
	
	hpx::lcos::future<float> out = hpx::lcos::local::dataflow
	(
		hpx::util::unwrapped
		( [] (std::vector<hpx::lcos::future<float>> roots,
		      std::vector<float> weights)
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
		),
		future_roots,
		hpx::lcos::make_ready_future(weights)
	);
	return out.get();
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

	neuron::neuron(float bias, int count_activations, int random)
	{
		this->value = 0.0;
		this->bias = bias;
		if(bias)
		{
			this->value = 1.0;
			this->new_value = hpx::lcos::make_ready_future((float)1.0);
		}  else this->new_value = hpx::lcos::make_ready_future((float)0.0);
    	                this->new_error = hpx::lcos::make_ready_future((float)0.0);
		        this->new_delta = hpx::lcos::make_ready_future((float)0.0);
		//Activation-related data
		for(int k = 0; k < count_activations; k++)
		{
			this->weights.push_back(rnd()-0.5);
			this->last_change.push_back(0);
		}
	}
	float neuron::get_delta()
	{
		this->delta = this->new_delta.get();
		return this->delta;
	}
	float neuron::get_error()
	{
		this->error = this->new_error.get();
		return this->error;
	}
	void neuron::run(std::vector<neuron> roots, int serial)
	{
		if (this->bias) return;
		if (serial) this->new_value = hpx::lcos::make_ready_future(productsum(future_get_roots(roots),this->weights));
	        else	    this->new_value = hpx::async(&future_productsum,roots,this->weights);
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
		for(int i = 0; i < neurons; i++){
			init.push_back(neuron(0,count_activations,random));
			init[i].out = out;
		}
	if(bias)	init.push_back(neuron(1,count_activations,random));
		this->contents = init;
//		std::cout << this->contents.size() << " ";
		this->out = out;
	}

	void run(std::vector<neuron> roots,int serial)
	{
		for(int i = 0; i < (int)this->contents.size(); i++)
			this->contents[i].run(roots,serial);
	}

	void add(neuron x)
	{
		contents.push_back(x);
	}

	int size()
	{
		return this->contents.size();
	}
	void correct(std::vector<float> v, float m, float n,neuron_row prev,neuron_row next,int serial)
	{
		for(int j = 0; j < (int)this->size(); j++)
	      		this->contents[j].correct((this->out)?v[j]:0,j,m,n,prev,next,serial);
	}
};

float calc_hidden_error(neuron_row next,int j)
{
	float error = 0;
	for (int k = 0; k < (int)next.size(); k++)
		if (next.contents[k].bias) continue;
		else error += next.contents[k].get_delta() * next.contents[k].weights[j];
	return error;
}
hpx::lcos::future<float> future_hidden_error(neuron_row next,int j)
{
	hpx::lcos::future<float> result = hpx::async(&calc_hidden_error,next,j);
	return result;
}
void neuron::correct(float target, int j /* index */, float m, float n, neuron_row prev, neuron_row next, int serial)
{
	if (this->bias) return;
	if (this->out)  this->new_error = hpx::lcos::make_ready_future(target - this->get_value());
        if(serial)
	{
			if (!this->out) this->new_error = hpx::lcos::make_ready_future(calc_hidden_error(next,j));
			this->new_delta = hpx::lcos::make_ready_future(this->get_error() * df(this->get_value()));
	}
	else
	{
			if (!this->out) this->new_error = future_hidden_error(next,j);
	                this->new_delta =
			hpx::lcos::local::dataflow
			(
				hpx::util::unwrapped
				( [] (float error, float value)
				{
					return error*df(value);
				}
				),this->new_error,hpx::lcos::make_ready_future(this->get_value())
			);
	}
	for(int k = 0; k < (int)prev.size(); k++)
		this->finalize_correct(prev,k,m,n);
}
void neuron::finalize_correct(neuron_row prev,int k,float m,float n)
{
	float change = this->get_delta() * prev.contents[k].get_value();
	float change2 = m*change + n*this->last_change[k];
	this->weights[k] += change2;
	this->last_change[k] = change;
}
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
	void run(int serial) //Forward Pass
	{
		for(int i = 1; i < (int)this->rows.size(); i++)
			if(i < FORWARD_THRESHOLD)
				this->rows[i].run(this->rows[i-1].contents,serial);
			else    this->rows[i].run(this->rows[i-1].contents,1);
	}
	std::vector<float> reverse(std::vector<float> vals)
	{
		std::vector<float> result;
		for(int i = (int)vals.size()-1; i >= 0; i--)
			result.push_back(vals[i]);
		return result;
	}
	float correct(std::vector<float> v, float m /*learning_rate*/, float n /*momentum*/, int serial)
	{
		std::vector<float> vi = v;
		//v = this->reverse(v);

		float error;

		for(int i = (int)this->rows.size()-1; i >= 1; i--)
		{
			int s = (int)this->rows.size();
			if(i > BACKPROP_THRESHOLD) serial = 1;
			if(s-1 == i)
				this->rows[i].correct(v,m,n,this->rows[i-1],this->rows[i-1],serial);
			else	this->rows[i].correct(v,m,n,this->rows[i-1],this->rows[i+1],serial);
		}
		error = 0;
		for(int i = 0; i < (int)this->rows[this->rows.size()-1].size(); i++)
			error += 0.5*pow(vi[i]-this->rows[this->rows.size()-1].contents[i].get_value(),2);
		return error;
	}

	//TODO: Create network in parallel 
	void init(int in, int hidden_rows, int hidden_cols, int out, int bias = 0)
	{
		int random = 0;
		std::vector<neuron_row> row;

		//Input Layer
		row.push_back(neuron_row(in,0,1,0,random));

		//Hiden Layers
		for(int i = 0; i < hidden_rows; i++)
			row.push_back(neuron_row(hidden_cols,0,1,row[i].size(),random));

		//Output Layer
		row.push_back(neuron_row(out,1,0,hidden_cols+1,random));
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
				std::cout << "\tCOL " << j << " -> ";
				std::cout << "VAL " << this->rows[i].contents[j].get_value() << "\n";
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

int main_main(int in, int hidden_rows, int hidden_cols, int out, int its, int serial, hpx::util::high_resolution_timer t)
{
        std::cout << "Initializing simulation... ";

	float toffset = t.elapsed();

	network n(in,hidden_rows,hidden_cols,out,1);

	std::cout << "Done. "
		  << "(" << (t.elapsed()-toffset) << " s)\n";

	int problem_count = 4;
	int problem_correct = 0;

	int display_output = 0;
	int success = 0;

	int i = 0;
	for(i = 0; i < its; i++)
	{
	        if(display_output) std::cout << i << " ";
		int s = i%(sizeof(tests)/sizeof(tests[0]));

		std::vector<float> sensor = to_vector(tests[s],sizeof(tests[s])/sizeof(float));
		std::vector<float> target = to_vector(targets[s],sizeof(targets[s])/sizeof(float));
		n.setSensors(sensor);
		std::cout << "Forward Pass... ";

		toffset = t.elapsed();
		n.run(serial);
		std::cout << "Done. (" << (t.elapsed()-toffset) << " s)\n";

		if(display_output)
		{
		    std::cout << "(";
		    for(int it = 0; it < (int)(sizeof(tests[0])/sizeof(tests[0][0])); it++)
		      {
			std::cout << sensor[it];
			if(it < (int)(sizeof(tests[0])/sizeof(tests[0][0]))-1) std::cout << " ";
		      }
		    std::cout << ") = ";
		}

		int valid = 1;

		for(int it = 0; it < (int)n.rows[n.rows.size()-1].contents.size(); it++)
		{
			int counter = 0;
			float r = n.rows[n.rows.size()-1].contents[it].get_value();
			if(r < 0) r = 0;
			r = round(r);
			if(display_output) std::cout << r << " ";
			if(round(n.rows[n.rows.size()-1].contents[it].get_value()) != targets[s][counter]) valid = 0;
			else counter++;
		}

		if(display_output)
		    std::cout << "(";

		if(display_output)
		for(int it = 0; it < (int)(sizeof(targets[0])/sizeof(targets[0][0])); it++)
		  {
		    std::cout << target[it];
		    if(it < (int)(sizeof(targets[0])/sizeof(targets[0][0]))-1) std::cout << " ";
		  }
		if(display_output)
	          std::cout << ")";

		  if(valid)
		    {
		      problem_correct++;
		      if(display_output) std::cout << "\033[32mCorrect!\033[0m  \t";
		    }
		  else
		    {
		      problem_correct = 0;
		      if(display_output) std::cout << "\033[31mIncorrect!\033[0m\t";
		    }

		if(!display_output) std::cout << "Backpropagating... ";

	        toffset = t.elapsed();
		float error = n.correct(target,0.05,0.01,serial);

		if(!display_output) std::cout << "Done. (" << (t.elapsed()-toffset) << " seconds)\n";

		if(display_output)
		  {
		    std::cout << error;
		    std::cout << "\n";
		  }

	      	if(problem_count == problem_correct)
		{
			success = 1;
			//break;
		}
	}
	std::cerr << i << " iterations.";
	if(success) std::cerr << "  Successful convergence.";
	std::cerr << "\n";
  return 0;
}

int hpx_main()
{
  int a,b,c,d,e,f;
  std::cin >> a >> b >> c >> d >> e >> f >> FORWARD_THRESHOLD >> BACKPROP_THRESHOLD;
  hpx::util::high_resolution_timer t;
  for(int i = 0; i < 1; i++)
    main_main(a,b,c,d,e,f,t);
  std::cout << "Total Time: " << t.elapsed() << " seconds.\n";
  return hpx::finalize();
}

int main(int argc, char* argv[])
{
  init();
  hpx::init(argc,argv);
}
