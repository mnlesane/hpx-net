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

HPX_REGISTER_COMPONENT_MODULE();

namespace app
{
    struct	HPX_COMPONENT_EXPORT
		some_component
      		: hpx::components::managed_component_base<some_component>
    {
	int val;
	some_component(): val(0)
	{}

        void increment()
        {
            this->val++;
        }

        // This will define the action type 'some_member_action' which
        // represents the member function 'some_member_function' of the
        // obect type 'some_component'.
        HPX_DEFINE_COMPONENT_ACTION(some_component,increment,increment_action);
    };
}

// Note: The second arguments to the macro below have to be systemwide-unique
//       C++ identifiers
HPX_REGISTER_ACTION_DECLARATION(app::some_component::increment_action);

