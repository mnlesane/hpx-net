/* Copyright (c) 2013 Michael LeSane
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*/

class neuron_row
{
	public:

	std::vector<neuron> contents;
	hpx::lcos::future<std::vector<neuron>> new_contents;

	int out;

	neuron_row(std::vector<neuron>,int);
	neuron_row(int,int,int,int,int);

	void run(neuron_row,int);
	void add(neuron);

	int size();

	void correct(std::vector<float>,float,float,neuron_row,neuron_row,int);
};

