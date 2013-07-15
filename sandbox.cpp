#include <iostream>
#include <cstdio>

#include <cstdlib>
#include <vector>
#include <cmath>

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/include/components.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/serialization.hpp>

struct s
{
	hpx::lcos::future<float> next;
	s()
	{
		this->next = hpx::lcos::make_ready_future((float)9.9);
	}
	void f()
	{
		this->next = hpx::lcos::make_ready_future((float)8.8);
	}
	float g()
	{
		return this->next.get();
	}
};

int hpx_main()
{
	s ob;
	std::cout << ob.g() << "\n";
	hpx::lcos::future<float> state = ob.next;
	ob.f();
	std::cout << ob.g() << "\n";
	std::cout << state.get() << "\n";
	return hpx::finalize();
}
int main(int argc, char* argv[])
{
	hpx::init(argc,argv);
}
