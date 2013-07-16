/* Copyright (c) 2013 Michael LeSane
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*/

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

//initialization
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

//waits on and returns delta from backpropagation.
float neuron::get_delta()
{
	this->delta = this->new_delta.get();
	return this->delta;
}

//waits on and returns hidden error.
float neuron::get_error()
{
	this->error = this->new_error.get();
	return this->error;
}

//forward pass
void neuron::run(std::vector<neuron> roots, int serial)
{
	serial = 1;
	/*
	Executing rows in parallel and neurons for each row in serial
	seems to sometimes give parallel execution a slight advantage over
	serial execution in certain simulations, if not equal/similar performance.

	To do this, set serial to 1 above.  Otherwise, remove the line.

	Larger networks (e.g., 100,000 neurons) tend to end in segfaults,
	and otherwise have excessively lengthy execution times,
	so scalability of the "finest-grain implementation" thus far
	has not been be properly assessed at this point.
	*/

	if (this->bias) return;
	if (serial) this->new_value = hpx::lcos::make_ready_future(productsum(future_get_roots(roots),this->weights));
        else	    this->new_value = hpx::async(&future_productsum,roots,this->weights);
}

//waits on and returns activation future.
float neuron::get_value()
{
    if(!this->bias)
      this->value = f(this->new_value.get());
    return this->value;
}

//backpropagation
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

//backpropagation finalization. momentum and weights adjusted.
void neuron::finalize_correct(neuron_row prev,int k,float m,float n)
{
	float change = this->get_delta() * prev.contents[k].get_value();
	float change2 = m*change + n*this->last_change[k];
	this->weights[k] += change2;
	this->last_change[k] = change;
}
