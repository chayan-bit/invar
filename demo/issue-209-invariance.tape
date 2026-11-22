# #209: worker-count invariance checked and recorded in the run receipt
# (vhs demo/issue-209-invariance.tape, invar on PATH).
Output demo/issue-209-invariance.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 820
Set TypingSpeed 12ms

Hide
Type "cd demo && P=issue-209-invariance.inv && clear" Enter
Show

Type "grep -v '^//' $P" Enter
Sleep 3s
Type "# Run on the reference and on 1, 2, 7 and 64 workers; compare result, trap and fuel:" Enter
Type "invar run $P --engine cpu-parallel --check-workers 1,2,7,64 --fuel 100000000" Enter
Sleep 8s
Type "clear; # The receipt records what was compared and how each run split the work:" Enter
Type `invar run $P --engine cpu-parallel --check-workers 1,2,7,64 --fuel 100000000 --format json | jq -c '.invariance | {engines, workers, compared, fuel_used, levels}, (.runs[] | {workers, published_regions, worker_joins, chunks: (.last_split | length)})'` Enter
Sleep 10s
Type "# Bad lists are refused (exit 2):" Enter
Type "invar run $P --engine cpu-parallel --check-workers 7,2 2>/dev/null; echo exit $?" Enter
Sleep 3s
