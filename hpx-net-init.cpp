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

int FORWARD_THRESHOLD,
    BACKPROP_THRESHOLD;

float productsum(std::vector<float>,std::vector<float>);

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
std::vector<hpx::lcos::future<float>> extract_future_roots(std::vector<neuron> contents)
{
	std::vector<hpx::lcos::future<float>> out;
	for(int i = 0; i < (int)contents.size(); i++)
		out.push_back(contents[i].get_f_future());
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
std::vector<float> to_vector(float x[],int s)
{
	std::vector<float> out;
	for(int i = 0; i < s; i++) out.push_back(x[i]);
	return out;
}

