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
    std::cout << r << "\n";
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

int hpx_main()
{
  std::vector<float> x = vector(9);
  return hpx::finalize();
}

int main()
{
  return hpx::init();
  return 0;
}
