// CPU boundary stand-ins only. No real ROS scheduler, transport or simulation.
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace geometry_msgs::msg {
struct Pose {struct Position {double x=0,y=0,z=0;}position;struct Orientation{double x=0,y=0,z=0,w=1;}orientation;using ConstSharedPtr=std::shared_ptr<const Pose>;};
struct PoseStamped {struct Header{std::string frame_id;struct Stamp{std::int32_t sec=0;std::uint32_t nanosec=0;}stamp;}header;Pose pose;};
}
namespace std_msgs::msg {struct Bool{bool data=false;};struct Int32MultiArray{std::vector<std::int32_t> data;};struct String{std::string data;};}
namespace std_srvs::srv {struct SetBool{struct Request{bool data=false;};struct Response{bool success=true;std::string message;};};}
namespace rcl_interfaces::msg {struct ParameterDescriptor{bool read_only=false;};}
namespace invar_cell_interfaces::msg {
struct CellSnapshot@JOINTS@V1{
 using ConstSharedPtr=std::shared_ptr<const CellSnapshot@JOINTS@V1>;
 std::uint64_t simulation_ns=0,step_index=0;
 std::array<std::uint8_t,32> source_ir_sha256{},joint_order_sha256{};
 std::array<double,@JOINTS@> q_rad{},dq_rad_s{};
 std::array<double,7> world_from_tool{},world_from_cube{};
 std::uint32_t attached=0;std::vector<std::uint8_t> geom_classes;std::vector<std::int32_t> contact_geom_ids;
 std::array<std::uint8_t,32> robot_joint_order_sha256{};
 std::vector<double> robot_q_si,robot_dq_si_per_s;
};
struct PlannerRequest@JOINTS@V2{
 std::array<double,@JOINTS@> start_q_rad{},goal_q_rad{};
 std::array<double,7> target_base_from_tool{},cube_base_pose{},tool_from_cube{};
 std::uint32_t phase=0;
};
struct PlannerWaypoint@JOINTS@V1{double time_from_start_s=0;std::array<double,@JOINTS@> q_rad{},dq_rad_s{};};
struct PlannerTrajectory@JOINTS@V1{using ConstSharedPtr=std::shared_ptr<const PlannerTrajectory@JOINTS@V1>;double dt_s=0;std::vector<PlannerWaypoint@JOINTS@V1> waypoints;};
}
namespace stub {
inline std::map<std::string,std::string> params;
inline bool stopped=false,service_success=true;
inline std::size_t subscribers=1;
inline std::vector<bool> attachment_calls;
inline std::function<void()> spin_hook;
template<class M>inline std::function<void(const M&)> published;
template<class M>inline std::function<void(typename M::ConstSharedPtr)> received;
template<class S>inline std::function<void(std::shared_ptr<typename S::Request>,std::shared_ptr<typename S::Response>)> service;
template<class M>void send(const M &m){received<M>(std::make_shared<const M>(m));}
}
namespace rclcpp {
struct KeepLast{explicit KeepLast(int){}};
struct QoS{explicit QoS(int){}explicit QoS(KeepLast){}QoS &reliable(){return *this;}};
struct TimerBase{using SharedPtr=std::shared_ptr<TimerBase>;};
template<class M>struct Publisher{
 using SharedPtr=std::shared_ptr<Publisher>;
 void publish(const M&m){if(stub::published<M>)stub::published<M>(m);}
};
template<class M>struct Subscription{using SharedPtr=std::shared_ptr<Subscription>;};
template<class S>struct Service{using SharedPtr=std::shared_ptr<Service>;};
template<class S>struct Client{
 using SharedPtr=std::shared_ptr<Client>;using SharedFuture=std::shared_future<std::shared_ptr<typename S::Response>>;
 bool service_is_ready(){return true;}
 template<class F>void async_send_request(std::shared_ptr<typename S::Request> r,F callback){
  stub::attachment_calls.push_back(r->data);std::promise<std::shared_ptr<typename S::Response>> p;auto response=std::make_shared<typename S::Response>();response->success=stub::service_success;p.set_value(response);callback(p.get_future().share());
 }
};
class Node {
 public:using SharedPtr=std::shared_ptr<Node>;explicit Node(const char*){}
 template<class T>T declare_parameter(const std::string &name,const T &fallback,const rcl_interfaces::msg::ParameterDescriptor&){auto i=stub::params.find(name);if(i==stub::params.end())return fallback;return i->second;}
 template<class M>typename Publisher<M>::SharedPtr create_publisher(const std::string&,int){return std::make_shared<Publisher<M>>();}
 template<class M>typename Publisher<M>::SharedPtr create_publisher(const std::string&,const QoS&){return std::make_shared<Publisher<M>>();}
 template<class M,class F>typename Subscription<M>::SharedPtr create_subscription(const std::string&,const QoS&,F callback){stub::received<M> = callback;return std::make_shared<Subscription<M>>();}
 template<class S>typename Client<S>::SharedPtr create_client(const std::string&){return std::make_shared<Client<S>>();}
 template<class S,class F>typename Service<S>::SharedPtr create_service(const std::string&,F callback){stub::service<S> = callback;return std::make_shared<Service<S>>();}
 template<class Duration,class F>TimerBase::SharedPtr create_wall_timer(Duration,F){return std::make_shared<TimerBase>();}
 struct EndpointInfo{std::string name,ns;const std::string &node_name()const{return name;}const std::string &node_namespace()const{return ns;}};
 // A recorder subscribes to every topic; only the planner node's own subscription counts.
 std::vector<EndpointInfo> get_subscriptions_info_by_topic(const std::string &topic)const{std::vector<EndpointInfo> v{{"rosbag2_recorder","/"}};if(topic=="/sequencer/request"&&stub::subscribers)v.push_back({"planner","/planner"});return v;}
};
inline void init(int,char**){}inline void shutdown(){stub::stopped=true;}inline void spin(std::shared_ptr<Node>){if(stub::spin_hook)stub::spin_hook();}
}
