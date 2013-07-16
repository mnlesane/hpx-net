/* Copyright (c) 2013 Michael LeSane
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*/

//TODO: find std/boost equivalents of reinvented wheels
#include "hpx-net-init.hpp"
#include "neuron.hpp"
#include "neuron_row.hpp"
#include "network.hpp"

#include "hpx-net-init.cpp"
#include "neuron.cpp"
#include "neuron_row.cpp"
#include "network.cpp"

//Initializes, trains, and backpropagates network on XOR problem based on inputs, while timing performance
int main_main(int in, int hidden_rows, int hidden_cols, int out, int its, int serial, hpx::util::high_resolution_timer t)
{
        std::cout << "Initializing simulation... ";

	float toffset = t.elapsed();

	network n(in,hidden_rows,hidden_cols,out,1);

	std::cout << "Done. "
		  << "(" << (t.elapsed()-toffset) << " s)\n";

	int problem_count = 4;
	int problem_correct = 0;

	int display_output = 1;
	int success = 0;

	int i = 0;
	for(i = 0; i < its; i++)
	{
	        if(display_output) std::cout << i << " ";
		int s = i%(sizeof(tests)/sizeof(tests[0]));

		std::vector<float> sensor = to_vector(tests[s],sizeof(tests[s])/sizeof(float));
		std::vector<float> target = to_vector(targets[s],sizeof(targets[s])/sizeof(float));
		n.setSensors(sensor);
		std::cout << "Forward Pass... ";

		toffset = t.elapsed();
		n.run(serial);
		std::cout << "Done. (" << (t.elapsed()-toffset) << " s)\n";

		if(display_output)
		{
		    std::cout << "(";
		    for(int it = 0; it < (int)(sizeof(tests[0])/sizeof(tests[0][0])); it++)
		      {
			std::cout << sensor[it];
			if(it < (int)(sizeof(tests[0])/sizeof(tests[0][0]))-1) std::cout << " ";
		      }
		    std::cout << ") = ";
		}

std::cout << "Waiting on results... ";
//toffset = t.elapsed();

int valid = 1;
		for(int it = 0; it < (int)n.rows[n.rows.size()-1].contents.size(); it++)
		{
			if(!serial) n.rows[n.rows.size()-1].finalize_run();
			int counter = 0;
			float r = n.rows[n.rows.size()-1].contents[it].get_value();
			if(r < 0) r = 0;
			r = round(r);
			if(display_output) std::cout << r << " ";
			if(round(n.rows[n.rows.size()-1].contents[it].get_value()) != targets[s][counter]) valid = 0;
			else counter++;
		}
std::cout << "Done. (" << t.elapsed() - toffset << " s total)\n";

		if(display_output)
		    std::cout << "(";

		if(display_output)
		for(int it = 0; it < (int)(sizeof(targets[0])/sizeof(targets[0][0])); it++)
		  {
		    std::cout << target[it];
		    if(it < (int)(sizeof(targets[0])/sizeof(targets[0][0]))-1) std::cout << " ";
		  }
		if(display_output)
	          std::cout << ")";

		  if(valid)
		    {
		      problem_correct++;
		      if(display_output) std::cout << "\033[32mCorrect!\033[0m  \t";
		    }
		  else
		    {
		      problem_correct = 0;
		      if(display_output) std::cout << "\033[31mIncorrect!\033[0m\t";
		    }

		if(!display_output) std::cout << "Backpropagating... ";

	        toffset = t.elapsed();
		float error = 0;
		      //error = n.correct(target,0.05,0.01,serial);

		if(!display_output) std::cout << "Done. (" << (t.elapsed()-toffset) << " seconds)\n";

		if(display_output)
		  {
		    std::cout << error;
		    std::cout << "\n";
		  }

	      	if(problem_count == problem_correct)
		{
			success = 1;
			//break;
		}
	}
	std::cerr << i << " iterations.";
	if(success) std::cerr << "  Successful convergence.";
	std::cerr << "\n";
  return 0;
}

//Prompts for inputs and launches simulation.
int hpx_main()
{
  int a,b,c,d,e,f;
  std::cin >> a >> b >> c >> d >> e >> f >> FORWARD_THRESHOLD >> BACKPROP_THRESHOLD;
  hpx::util::high_resolution_timer t;
  for(int i = 0; i < 1; i++)
    main_main(a,b,c,d,e,f,t);
  std::cout << "Total Time: " << t.elapsed() << " seconds.\n";
  return hpx::finalize();
}

//Initialization
int main(int argc, char* argv[])
{
  init();
  hpx::init(argc,argv);
}
