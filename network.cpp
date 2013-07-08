/* Copyright (c) 2013 Michael LeSane
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*/

//sets sensor activation values to those of values given.
void network::setSensors(float vals[])
{
	for(int i = 0; i < (int)(sizeof(vals)/sizeof(float)); i++)
	{
		this->rows[0].contents[i].value = vals[i];
		this->rows[0].contents[i].new_value = hpx::lcos::make_ready_future(vals[i]);
	}
}
void network::setSensors(std::vector<float> vals)
{
	for(int i = 0; i < (int)vals.size(); i++)
	{
		this->rows[0].contents[i].value = vals[i];
		this->rows[0].contents[i].new_value = hpx::lcos::make_ready_future(vals[i]);
	}
}
//Forward Pass
void network::run(int serial)
{
	for(int i = 1; i < (int)this->rows.size(); i++)
		if(i < FORWARD_THRESHOLD)
			this->rows[i].run(this->rows[i-1].contents,serial);
		else    this->rows[i].run(this->rows[i-1].contents,1);
}

//reverses a vector.
std::vector<float> network::reverse(std::vector<float> vals)
{
	std::vector<float> result;
	for(int i = (int)vals.size()-1; i >= 0; i--)
		result.push_back(vals[i]);
	return result;
}

//backpropagation
float network::correct(std::vector<float> v, float m /*learning_rate*/, float n /*momentum*/, int serial)
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
//network initialization.
void network::init(int in, int hidden_rows, int hidden_cols, int out, int bias = 0)
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
//constructur, calls initialization method.
network::network(int in, int hidden_rows, int hidden_cols, int out, int bias = 0)
{
	this->init(in,hidden_rows,hidden_cols,out,bias);
}
//displays network contents.
void network::profile()
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
