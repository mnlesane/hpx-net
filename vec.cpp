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

std::vector<double> vector(int n)
{
  std::vector<double> out;
  for(int i = 0; i < n; i++)
  {
    double r = (double)(rand())/((double)(rand()+1));
    out.push_back(r);
  }
  return out;
}

double dotproduct(std::vector<double> a, std::vector<double> b)
{
  double out = 0;
  if(a.size() != b.size()) ;// error
  for(int i = 0; i < (int)a.size(); i++)
    out += a[i] * b[i];
  return out;
}

double parallel_dotproduct(std::vector<double> vector1, std::vector<double> vector2, int start, int stop, int thresh)
{
	double out = 0;
	if(stop-start <= thresh)
		if(stop-start > 0)
			for(int i = start; i <= stop; i++)
				out += vector1[i] * vector2[i];
				//out += 1;
		else		out = vector1[start] * vector2[start];
				//out = 1;
	else
	{
		hpx::lcos::future<double> left = hpx::async(&parallel_dotproduct,vector1,vector2,start,(int)((start+stop)/2),thresh);
		hpx::lcos::future<double> right = hpx::async(&parallel_dotproduct,vector1,vector2,(int)((start+stop)/2)+1,stop,thresh);
		out = left.get() + right.get();
	}
	return out;
}


int hpx_main()
{
  hpx::util::high_resolution_timer t;

  //Length of vectors
  double len;
  std::cin >> len;

  //Minimum threshold
  double minthresh = len/16;

  //Initialization of vectors
  std::cout << "Initializing Vector 1...\n";
  std::vector<double> a = vector(len);
  std::cout << "Initializing Vector 2...\n";
  std::vector<double> b = vector(len);

  double n;
  //Serial Test
  {
	double x = t.elapsed();
	double c = dotproduct(a,b);
	double y = t.elapsed();
	n = y-x;
	std::cout << "Result: " << c << "\n";
	std::cout << "Serial: " << n << "\n\n";
  }

  //Parallel Tests
  for(double thresh = len*2; thresh >= ceil(minthresh); thresh /= 2)
  {
	double x = t.elapsed();
	double c = parallel_dotproduct(a,b,0,a.size()-1,thresh);
	double y = t.elapsed();
	double m = y-x;
	std::cout << "Result: " << c << "\n";
	std::cout << "Parallel: " << m << " (t=" << thresh << ")\n";
	double q = n/(m+0.000001);
	std::cout << "Ratio: " << q << "\n\n";
  }
  return hpx::finalize();
}

int main()
{
  return hpx::init();
  return 0;
}
