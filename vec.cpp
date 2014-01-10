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

std::vector<float> vector(int n)
{
  std::vector<float> out;
  for(int i = 0; i < n; i++)
  {
    float r = (float)(rand())/((float)(rand()+1));
    out.push_back(r);
  }
  return out;
}

float dotproduct(std::vector<float> a, std::vector<float> b)
{
  float out = 0;
  if(a.size() != b.size()) ;// error
  for(int i = 0; i < (int)a.size(); i++)
    out += a[i] * b[i];
  return out;
}

float parallel_dotproduct(std::vector<float> vector1, std::vector<float> vector2, int start, int stop, int thresh)
{
	float out = 0;
	if(stop-start <= thresh)
		if(stop-start > 0)
			for(int i = start; i <= stop; i++)
				out += vector1[i] * vector2[i];
		else		out = vector1[start] * vector2[start];
	else
	{
		hpx::lcos::future<float> left = hpx::async(&parallel_dotproduct,vector1,vector2,start,(int)((start+stop)/2),thresh);
		hpx::lcos::future<float> right = hpx::async(&parallel_dotproduct,vector1,vector2,(int)((start+stop)/2)+1,stop,thresh);
		out = left.get() + right.get();
	}
	return out;
}


int hpx_main()
{
  hpx::util::high_resolution_timer t;

  //Length of vectors
  float len;
  std::cin >> len;

  //Minimum threshold
  float minthresh = len/16;

  //Initialization of vectors
  std::cout << "Initializing Vector 1...\n";
  std::vector<float> a = vector(len);
  std::cout << "Initializing Vector 2...\n";
  std::vector<float> b = vector(len);

  float n;
  //Serial Test
  {
	float x = t.elapsed();
	float c = dotproduct(a,b);
	float y = t.elapsed();
	n = y-x;
	std::cout << "Result: " << c << "\n";
	std::cout << "Serial: " << n << "\n\n";
  }

  //Parallel Tests
  for(float thresh = len*2; thresh >= ceil(minthresh); thresh /= 2)
  {
	float x = t.elapsed();
	float c = parallel_dotproduct(a,b,0,a.size()-1,thresh);
	float y = t.elapsed();
	float m = y-x;
	std::cout << "Result: " << c << "\n";
	std::cout << "Parallel: " << m << " (t=" << thresh << ")\n";
	float q = n/(m+0.000001);
	std::cout << "Ratio: " << q << "\n\n";
  }
  return hpx::finalize();
}

int main()
{
  return hpx::init();
  return 0;
}
