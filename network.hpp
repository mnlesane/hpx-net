class network
{
	public:
	std::vector<neuron_row> rows;
	std::vector<hpx::lcos::future<neuron_row>> future_rows;

	void setSensors(float vals[]);

	void setSensors(std::vector<float>);

	void run(int); //Forward Pass

	std::vector<float> reverse(std::vector<float>);

	float correct(std::vector<float>,float,float,int);

	//TODO: Create network in parallel 
	void init(int,int,int,int,int);

	network(int,int,int,int,int);

	void profile();
};

