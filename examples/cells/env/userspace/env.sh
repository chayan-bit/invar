# Source to use the Route 1 environment in the current bash shell:
#   source examples/cells/env/userspace/env.sh ROOT
# Activates the conda environment as build.sh's `micromamba run` does (the
# activation scripts set the conda compilers and their rpath flags), then the
# ROS workspace. No LD_LIBRARY_PATH, so host tools keep their own libraries.
invar_env_root=${1:?usage: source env.sh ROOT}
# Activation and colcon setup scripts read unset variables; keep the caller's nounset mode.
case $- in *u*) invar_env_nounset=1; set +u ;; *) invar_env_nounset= ;; esac
export MAMBA_ROOT_PREFIX=$invar_env_root/mamba
eval "$("$invar_env_root/bin/micromamba" shell hook -s bash)"
micromamba activate "$invar_env_root/env"
export CMAKE_PREFIX_PATH=$invar_env_root/env${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
# shellcheck disable=SC1091
source "$invar_env_root/ws/install/setup.bash"
[[ -z $invar_env_nounset ]] || set -u
