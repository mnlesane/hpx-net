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

struct HPX_COMPONENT_EXPORT fibmgr : hpx::components::simple_component<fibmgr>
{
	public: int x;
	fibmgr(int x)
	{
		this->x = x;
	}
	int fib()
	{
		int x = this->x;
		hpx::naming::id_type const LOCAL = hpx::find_here();

		if(x==1) return 1;
		if(x==0) return 0;
		hpx::lcos::future<int> a = hpx::async(fib_action,x-1);
		hpx::lcos::future<int> b = hpx::async(fib_action,x-2);
		return a.get()+b.get();
	}
	HPX_DEFINE_COMPONENT_ACTION(fibmgr,fib,fib_action);
};
	HPX_REGISTER_ACTION_DECLARATION(fibmgr::fib_action);

int hpx_main()
{
	fibmgr f(24);
	int x;
	std::cin >> x;
	std::cout << hpx::async(fibmgr::fib_action,hpx::find_here(),24).get() << "\n";
	return hpx::finalize();
}
int main(int argc, char* argv[])
{
	hpx::init(argc,argv);
}
