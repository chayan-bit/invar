#include "core.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc < 7)
      throw std::invalid_argument(
          "usage: pinocchio_fk_cli MODEL BASE TOOL COUNT JOINT_NAMES... Q_RAD...");
    std::size_t end = 0;
    const auto count = std::stoul(argv[4], &end);
    if (argv[4][end] != '\0' || count == 0 || count > 16 ||
        argc != static_cast<int>(5 + 2 * count))
      throw std::invalid_argument("invalid joint count or argument list");

    std::vector<std::string> names;
    std::vector<double> q;
    names.reserve(count);
    q.reserve(count);
    for (std::size_t i = 0; i < count; ++i) names.emplace_back(argv[5 + i]);
    for (std::size_t i = 0; i < count; ++i) {
      const std::string text = argv[5 + count + i];
      const double value = std::stod(text, &end);
      if (end != text.size()) throw std::invalid_argument("invalid joint number");
      q.push_back(value);
    }
    invar_fixture::PinocchioFk fk(argv[1], argv[2], argv[3], names);
    const auto pose = fk.step(q);
    std::cout << std::setprecision(17);
    for (const auto value : pose.translation_m) std::cout << value << ' ';
    for (const auto value : pose.quaternion_wxyz) std::cout << value << ' ';
    std::cout << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pinocchio_fk_cli: " << error.what() << '\n';
    return 1;
  }
}
