# #50 part 2: a placement run records its float reductions (vhs demo/issue-50-placement.tape, invar on PATH).
Output demo/issue-50-placement.gif
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 760
Set TypingSpeed 20ms

Hide
Type "cd demo && P=issue-50-placement-fsum.inv && A='--engine auto --format json --host-class demo'" Enter
# A provisional CPU-only profile of this host, as tests/cli/issue_50_placement_receipts.rs builds it.
Type `invar run $P $A | jq '{schema_version:1,probe_version:"placement-v1",units:"ms,bytes,work",identity:.placement.resource_identity,provisional:true,source_revision:"0123456789012345678901234567890123456789",observations:{before:{load:null,swap_bytes:null},after:{load:null,swap_bytes:null}},refusals:[],measurements:[{family:"placement.wrapping_add1.u32",numeric:"u32-wrapping",numeric_contract_version:"wrapping-u32-v1",engine:"cpu",boundary:"llvm-host-array-v1",region:null,sizes:[1,10000],passes:[range(5)|{fixed_ms:0.0,per_work_ms:1.0,link_bytes_per_ms:1000.0,link_latency_ms:0.0,launch_floor_ms:0.0,page_in_ms_per_byte:null}],best:{fixed_ms:0.0,per_work_ms:1.0,link_bytes_per_ms:1000.0,link_latency_ms:0.0,launch_floor_ms:0.0,page_in_ms_per_byte:null}}]}' > profile.json && clear` Enter
Show

Type "cat $P" Enter
Sleep 2s
Type "# A placement run (trusted provisional profile) now lists the reductions it reached:" Enter
Type "invar run $P $A --profile profile.json --trust-provisional | jq -c '{auto_path, value: .result.value, route: .placement.route.host, reductions: [.placement.reductions[] | {kind, backend, reason}]}'" Enter
Sleep 4s
Type "# The explain lists the same entries in its device choices, as the established path does:" Enter
Type `invar explain $P $A --profile profile.json --trust-provisional | jq -c '[.device_execution.choices[] | select((.kind // "") | startswith("fsum.")) | {kind, backend, reason}]'` Enter
Sleep 4s
Type "# The established auto path, for comparison:" Enter
Type `invar explain $P $A | jq -c '{auto_path, fsum: [.device_execution.choices[] | select((.kind // "") | startswith("fsum.")) | {kind, backend, reason}]}'` Enter
Sleep 5s
