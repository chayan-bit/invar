#include <atomic>
#include <cstdint>
#include <memory>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <invar_cell_interfaces/msg/cell_snapshot@JOINTS@_v1.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <mujoco/mujoco.h>
#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace invar_cell_bringup {
class AttachmentPlugin final : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase {
 public:
  bool init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data) override {
    weld_id_ = mj_name2id(model, mjOBJ_EQUALITY, "@OBJECT@_attach");
    tool_id_ = mj_name2id(model, mjOBJ_SITE, "invar_tool");
    object_id_ = mj_name2id(model, mjOBJ_BODY, "@OBJECT@");
    root_id_ = mj_name2id(model, mjOBJ_BODY, "@ROBOT_ROOT@");
    table_id_ = mj_name2id(model, mjOBJ_BODY, "table");
    if (weld_id_ < 0 || tool_id_ < 0 || object_id_ < 0 || root_id_ < 0 || table_id_ < 0 || model->ngeom > 4096 || !private_forward_supported(model)) return false;
    constexpr std::array<const char *, @JOINTS@> names = {@JOINT_NAMES@};
    for (std::size_t i=0;i<names.size();++i) {
      const int id=mj_name2id(model,mjOBJ_JOINT,names[i]);
      if (id<0 || model->jnt_type[id]!=mjJNT_HINGE) return false;
      q_ids_[i]=model->jnt_qposadr[id]; v_ids_[i]=model->jnt_dofadr[id];
    }
    constexpr std::array<const char *,@ROBOT_JOINTS@> robot_names={@ROBOT_JOINT_NAMES@};
    constexpr std::array<int,@ROBOT_JOINTS@> robot_types={@ROBOT_JOINT_TYPES@};
    for(std::size_t i=0;i<robot_names.size();++i) {
      const int id=mj_name2id(model,mjOBJ_JOINT,robot_names[i]);
      if(id<0 || model->jnt_type[id]!=robot_types[i])return false;
      robot_q_ids_[i]=model->jnt_qposadr[id];robot_v_ids_[i]=model->jnt_dofadr[id];
    }
    classes_.resize(model->ngeom);
    for (int g=0;g<model->ngeom;++g) {
      int body=model->geom_bodyid[g];
      classes_[g]=body==table_id_ ? 2 : body==object_id_ ? 3 : 0;
      for (int b=body;b>0;b=model->body_parentid[b]) {
        if (b==root_id_) {classes_[g]=1;break;}
      }
    }
    model_=model;frame_.reset(mj_makeData(model));
    if (!frame_) return false;
    desired_.store(false);
    data->eq_active[weld_id_] = 0;
    state_ = node->create_publisher<std_msgs::msg::Bool>("/invar/attachment_active", 10);
    tool_ = node->create_publisher<geometry_msgs::msg::PoseStamped>("/invar/tool_pose", 10);
    object_ = node->create_publisher<geometry_msgs::msg::PoseStamped>("/invar/object_pose", 10);
    contacts_ = node->create_publisher<std_msgs::msg::Int32MultiArray>("/invar/contact_geom_ids", 10);
    snapshot_ = node->create_publisher<Snapshot>("/invar/cell_snapshot",rclcpp::QoS(rclcpp::KeepLast(256)).reliable());
    service_ = node->create_service<std_srvs::srv::SetBool>(
      "/invar/set_attachment",
      [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
             std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
        desired_.store(request->data);
        response->success = true;
        response->message = "queued for next physics step";
      });
    return true;
  }
  void update(const mjModel*, mjData* data) override {
    if (snapshot_fault_.load(std::memory_order_acquire))
      throw std::runtime_error("S1406: coherent snapshot capture/queue failure");
    auto r=read_.load(std::memory_order_relaxed);
    const auto w=write_.load(std::memory_order_acquire);
    while (r<w) {
      const auto &p=packets_[r%packets_.size()];
      Snapshot snapshot;
      snapshot.simulation_ns=p.simulation_ns; snapshot.step_index=p.step_index;
      snapshot.source_ir_sha256={@IR_BYTES@}; snapshot.joint_order_sha256={@ORDER_BYTES@};
      snapshot.q_rad=p.q; snapshot.dq_rad_s=p.dq;
      snapshot.world_from_tool=p.tool; snapshot.world_from_cube=p.cube; snapshot.attached=p.attached;
      snapshot.geom_classes.assign(classes_.begin(),classes_.end());
      snapshot.contact_geom_ids.assign(p.contacts.begin(),p.contacts.begin()+p.contact_count);
      snapshot.robot_joint_order_sha256={@ROBOT_ORDER_BYTES@};
      snapshot.robot_q_si.assign(p.robot_q.begin(),p.robot_q.end());
      snapshot.robot_dq_si_per_s.assign(p.robot_dq.begin(),p.robot_dq.end());
      snapshot_->publish(snapshot);
      read_.store(++r,std::memory_order_release);
    }
    std_msgs::msg::Bool state;
    state.data = data->eq_active[weld_id_] != 0;
    state_->publish(state);
    if (!std::isfinite(data->time) || data->time<0 || data->time>=18446744073.0)
      throw std::runtime_error("S1406: invalid legacy telemetry time");
    const auto nanoseconds = static_cast<std::uint64_t>(data->time * 1000000000.0);
    geometry_msgs::msg::PoseStamped tool;
    tool.header.frame_id = "world";
    tool.header.stamp.sec = static_cast<std::int32_t>(nanoseconds / 1000000000);
    tool.header.stamp.nanosec = static_cast<std::uint32_t>(nanoseconds % 1000000000);
    tool.pose.position.x = data->site_xpos[3 * tool_id_];
    tool.pose.position.y = data->site_xpos[3 * tool_id_ + 1];
    tool.pose.position.z = data->site_xpos[3 * tool_id_ + 2];
    mjtNum quaternion[4];
    mju_mat2Quat(quaternion, data->site_xmat + 9 * tool_id_);
    tool.pose.orientation.w = quaternion[0];
    tool.pose.orientation.x = quaternion[1];
    tool.pose.orientation.y = quaternion[2];
    tool.pose.orientation.z = quaternion[3];
    tool_->publish(tool);
    geometry_msgs::msg::PoseStamped object = tool;
    object.pose.position.x = data->xpos[3 * object_id_];
    object.pose.position.y = data->xpos[3 * object_id_ + 1];
    object.pose.position.z = data->xpos[3 * object_id_ + 2];
    object.pose.orientation.w = data->xquat[4 * object_id_];
    object.pose.orientation.x = data->xquat[4 * object_id_ + 1];
    object.pose.orientation.y = data->xquat[4 * object_id_ + 2];
    object.pose.orientation.z = data->xquat[4 * object_id_ + 3];
    object_->publish(object);
    std_msgs::msg::Int32MultiArray contacts;
    contacts.data.reserve(2 * data->ncon);
    for (int index = 0; index < data->ncon; ++index) {
      contacts.data.push_back(data->contact[index].geom1);
      contacts.data.push_back(data->contact[index].geom2);
    }
    contacts_->publish(contacts);
  }
  void pre_step(mjData* data) override {
    // Never publish/block the live physics thread. Forward a private copy so
    // derived poses/contacts refer to these exact q/dq rather than a stale cache.
    const auto w=write_.load(std::memory_order_relaxed);
    const auto r=read_.load(std::memory_order_acquire);
    if (snapshot_fault_.load(std::memory_order_acquire) || w-r>=packets_.size() || w==UINT64_MAX || !model_ || !frame_ || !private_forward_supported(model_)) {
      snapshot_fault_.store(true,std::memory_order_release);
    } else {
      mj_copyData(frame_.get(),model_,data);
      mj_forward(model_,frame_.get());
      auto *s=frame_.get();
      if (!std::isfinite(s->time) || s->time<0 || s->time>=18446744073.0 || s->ncon<0 || s->ncon>1024) {
        snapshot_fault_.store(true,std::memory_order_release);
      } else {
        auto &p=packets_[w%packets_.size()];
        p.simulation_ns=static_cast<std::uint64_t>(s->time*1000000000.0);p.step_index=w+1;
        p.attached=s->eq_active[weld_id_]?1:0;
        for (std::size_t i=0;i<q_ids_.size();++i) {
          p.q[i]=s->qpos[q_ids_[i]];p.dq[i]=s->qvel[v_ids_[i]];
          if (!std::isfinite(p.q[i]) || !std::isfinite(p.dq[i])) snapshot_fault_.store(true,std::memory_order_release);
        }
        for(std::size_t i=0;i<robot_q_ids_.size();++i) {
          p.robot_q[i]=s->qpos[robot_q_ids_[i]];p.robot_dq[i]=s->qvel[robot_v_ids_[i]];
          if(!std::isfinite(p.robot_q[i])||!std::isfinite(p.robot_dq[i]))snapshot_fault_.store(true,std::memory_order_release);
        }
        mjtNum quaternion[4];mju_mat2Quat(quaternion,s->site_xmat+9*tool_id_);
        p.tool={s->site_xpos[3*tool_id_],s->site_xpos[3*tool_id_+1],s->site_xpos[3*tool_id_+2],quaternion[1],quaternion[2],quaternion[3],quaternion[0]};
        p.cube={s->xpos[3*object_id_],s->xpos[3*object_id_+1],s->xpos[3*object_id_+2],s->xquat[4*object_id_+1],s->xquat[4*object_id_+2],s->xquat[4*object_id_+3],s->xquat[4*object_id_]};
        p.contact_count=2*s->ncon;
        for(int i=0;i<s->ncon;++i){p.contacts[2*i]=s->contact[i].geom1;p.contacts[2*i+1]=s->contact[i].geom2;}
        write_.store(w+1,std::memory_order_release);
      }
    }
    // Snapshot describes the completed previous state, before changing the
    // equality for the next step. An acknowledgement alone is never application.
    data->eq_active[weld_id_]=desired_.load();
  }
  void on_reset(mjData* data) override {
    desired_.store(false);
    data->eq_active[weld_id_] = 0;
  }
  void cleanup() override {
    service_.reset();
    state_.reset();
    tool_.reset();
    object_.reset();
    contacts_.reset();
    snapshot_.reset();
  }
 private:
  static bool private_forward_supported(const mjModel *model) {
    // A global callback or model plugin could mutate shared external state when
    // forwarding a copy. Refuse that unsupported configuration explicitly.
    return model->nplugin==0 && !mjcb_control && !mjcb_passive && !mjcb_contactfilter && !mjcb_sensor && !mjcb_time && !mjcb_act_dyn && !mjcb_act_gain && !mjcb_act_bias;
  }
  int weld_id_ = -1;
  int tool_id_ = -1;
  int object_id_ = -1;
  int root_id_ = -1;
  int table_id_ = -1;
  struct Packet {
    std::uint64_t simulation_ns=0,step_index=0;
    std::array<double,@JOINTS@> q{},dq{};
    std::array<double,@ROBOT_JOINTS@> robot_q{},robot_dq{};
    std::array<double,7> tool{},cube{};
    std::uint32_t attached=0;
    std::array<std::int32_t,2048> contacts{};
    std::size_t contact_count=0;
  };
  const mjModel *model_=nullptr;
  std::unique_ptr<mjData,decltype(&mj_deleteData)> frame_{nullptr,mj_deleteData};
  std::array<Packet,256> packets_{};
  std::atomic<std::uint64_t> write_{0},read_{0};
  std::atomic<bool> snapshot_fault_{false};
  using Snapshot=invar_cell_interfaces::msg::CellSnapshot@JOINTS@V1;
  std::array<int,@JOINTS@> q_ids_{};
  std::array<int,@JOINTS@> v_ids_{};
  std::array<int,@ROBOT_JOINTS@> robot_q_ids_{},robot_v_ids_{};
  std::vector<std::uint8_t> classes_;
  rclcpp::Publisher<Snapshot>::SharedPtr snapshot_;
  std::atomic<bool> desired_{false};
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr service_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr state_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr tool_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr object_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr contacts_;
};
}  // namespace invar_cell_bringup
PLUGINLIB_EXPORT_CLASS(invar_cell_bringup::AttachmentPlugin, mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
