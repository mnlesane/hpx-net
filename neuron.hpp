/* Copyright (c) 2013 Michael LeSane
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*/

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

