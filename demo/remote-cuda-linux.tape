# #333: a ref's CUDA suites on a shared Linux CUDA host, at low load (vhs demo/remote-cuda-linux.tape;
# REMOTE_CUDA_HOST and REMOTE_CUDA_DIR set in the environment).
Output demo/remote-cuda-linux.gif
Set Shell bash
Set FontSize 13
Set Width 1400
Set Height 800
Set TypingSpeed 15ms

Hide
Type `export PS1='$ ' && clear` Enter
Show

Type "# The CUDA runtime round trips: idle-GPU gate, cached build, one suite at a time, footprint, verdict." Enter
Type `REMOTE_CUDA_TESTS=gpu_cuda scripts/remote-cuda.sh HEAD runtime_round_trip 2>&1 | grep -v -e '^test .* ok$' -e '^remote-cuda: skipped'; echo exit=${PIPESTATUS[0]}` Enter
Wait+Screen@240s /exit=[0-9]/
Sleep 4s
