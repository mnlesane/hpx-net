#include <iostream>
#include <string>

struct s
{
	int x;
	float y;
	std::string z;
	s* new_state;

	s()
	{
		this->x = 9;
		this->y = 9.9;
		this->z = "nine";
	}
	void readjust()
	{
		this->x = 8;
		this->y = 8.8;
		this->z = "eight";
		this->new_state = new s;
	}
	void bigswitch()
	{
		this = this->new_state;
	}
};

int main()
{
};
