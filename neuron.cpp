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