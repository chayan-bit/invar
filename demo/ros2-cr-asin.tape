# #273: the UR5e base orientation uses the correctly rounded asin and atan2, so system gen ros2 is
# identical on every platform (vhs demo/ros2-cr-asin.tape from the repository root, invar on PATH).
Output demo/ros2-cr-asin.gif
Set Shell bash
Set FontSize 14
Set Width 1400
Set Height 600
Set TypingSpeed 20ms

Hide
Type "export PS1='$ ' T=$(mktemp -d)/ros2 && clear" Enter
Show

Type "# invar-math-cr-v1 pitch for the UR5e shoulder (Windows' libm gave ...722, glibc ...724):" Enter
Type "invar system gen ros2 examples/cells/reach_place/cell-ur5e.inv --out $T" Enter
Sleep 3s
Type `grep -rho 'rpy="0 1.57[0-9]* 0"' $T | sort | uniq -c` Enter
Sleep 2s
Type "# the tree hash is the example's golden, so the Windows exclusion of that test is gone:" Enter
Type `git diff origin/main -- .github/workflows/ci.yml | grep '^[-+] '` Enter
Sleep 6s
