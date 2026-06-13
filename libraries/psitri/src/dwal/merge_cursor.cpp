#include <psitri/dwal/merge_cursor_impl.hpp>

namespace psitri::dwal
{
   template class basic_merge_cursor<psitri_engine<std_lock_policy>>;
}  // namespace psitri::dwal
