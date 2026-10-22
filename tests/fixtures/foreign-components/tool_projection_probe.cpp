// CPU probe for the actual generated private adapter, never a ROS/GPU runtime.
#include "tool_projection.hpp"
#include <iomanip>
#include <iostream>
namespace adapter = invar_tool_projection_v1;
struct Wire { double xyz_m[3]; double quaternion_xyzw[4]; };
struct Request { Wire target_base_from_tool, tool_from_cube, cube_base_pose; };
bool read(double *p,double *q) {
  for(int i=0;i<3;++i) if(!(std::cin>>p[i])) return false;
  for(int i=0;i<4;++i) if(!(std::cin>>q[i])) return false;
  return true;
}
void print(const double *p,const double *q) {
  for(int i=0;i<3;++i) std::cout<<p[i]<<' ';
  for(int i=0;i<4;++i) std::cout<<q[i]<<' ';
}
int main() {
  std::cout<<std::setprecision(17);
  adapter::Pose raw{};
  while(read(raw.p,raw.q)) {
    Request request{};
    if(!read(request.target_base_from_tool.xyz_m,request.target_base_from_tool.quaternion_xyzw) ||
       !read(request.tool_from_cube.xyz_m,request.tool_from_cube.quaternion_xyzw) ||
       !read(request.cube_base_pose.xyz_m,request.cube_base_pose.quaternion_xyzw)) return 1;
    if(!adapter::fk_to_cell(raw.p,raw.q) || !adapter::request_to_reference(request)) return 2;
    print(raw.p,raw.q);
    print(request.target_base_from_tool.xyz_m,request.target_base_from_tool.quaternion_xyzw);
    print(request.tool_from_cube.xyz_m,request.tool_from_cube.quaternion_xyzw);
    print(request.cube_base_pose.xyz_m,request.cube_base_pose.quaternion_xyzw);
    std::cout<<'\n';
  }
  return std::cin.eof()?0:1;
}
