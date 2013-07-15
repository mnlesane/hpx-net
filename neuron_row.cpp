/* Copyright (c) 2013 Michael LeSane
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*/

//initializes a neuron row given a vector of neurons.
neuron_row::neuron_row(std::vector<neuron> init,int out)
{
	this->contents = init;
	this->out = out;
}

//initializes a neuron row given various parameters.
neuron_row::neuron_row(int neurons, int out, int bias, int count_activations, int random)
{
	std::vector<neuron> init;
	for(int i = 0; i < neurons; i++){
		init.push_back(neuron(0,count_activations,random));
		init[i].out = out;
	}
if(bias)	init.push_back(neuron(1,count_activations,random));
	this->contents = init;
	this->out = out;
}

//forward pass
std::vector<neuron> run_new_contents
(
	hpx::lcos::future<std::vector<neuron>> prev_row,
	std::vector<neuron> current_row,
	int serial
)
{
	return hpx::lcos::local::dataflow
	(
		hpx::util::unwrapped
		( [] (std::vector<neuron> roots, std::vector<neuron> current, int serial)
		{
			for(int i = 0; i < (int)current.size(); i++)
				current[i].run(roots,serial);
			return current;
		}
		),prev_row,hpx::lcos::make_ready_future(current_row),hpx::lcos::make_ready_future((int)serial)
	).get();
}

void neuron_row::run(neuron_row prev,int serial)
{
	//New Implementation
	this->new_contents = hpx::async(&run_new_contents,prev.new_contents,this->contents,serial);
	return;

	//Old Implementation
	std::vector<neuron> roots = prev.contents;
	for(int i = 0; i < (int)this->contents.size(); i++)
		this->contents[i].run(roots,serial);
}

//adds a neuron to the row.  weights of next row not affected, please eventually fix this.
void neuron_row::add(neuron x)
{
	contents.push_back(x);
}

//returns the number of neurons in the row.
int neuron_row::size()
{
	return this->contents.size();
}

//backpropagation
void neuron_row::correct(std::vector<float> v, float m, float n,neuron_row prev,neuron_row next,int serial)
{
	for(int j = 0; j < (int)this->size(); j++)
      		this->contents[j].correct((this->out)?v[j]:0,j,m,n,prev,next,serial);
}
