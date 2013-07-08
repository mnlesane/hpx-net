/* Copyright (c) 2013 Michael LeSane
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*/

neuron_row::neuron_row(std::vector<neuron> init,int out)
{
	this->contents = init;
	this->out = out;
}

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
void neuron_row::run(std::vector<neuron> roots,int serial)
{
	for(int i = 0; i < (int)this->contents.size(); i++)
		this->contents[i].run(roots,serial);
}
void neuron_row::add(neuron x)
{
	contents.push_back(x);
}
int neuron_row::size()
{
	return this->contents.size();
}
void neuron_row::correct(std::vector<float> v, float m, float n,neuron_row prev,neuron_row next,int serial)
{
	for(int j = 0; j < (int)this->size(); j++)
      		this->contents[j].correct((this->out)?v[j]:0,j,m,n,prev,next,serial);
}
