fn llvm_ir(program: &mir::Program) -> Result<NativeIr, String> {
    llvm_ir_mode(program, None)
}

fn llvm_ir_mode(
    program: &mir::Program,
    exports: Option<(&[usize], &std::collections::HashSet<usize>, &str)>,
) -> Result<NativeIr, String> {
    llvm_ir_debug(program, exports, None)
}

fn llvm_ir_debug(
    program: &mir::Program,
    exports: Option<(&[usize], &std::collections::HashSet<usize>, &str)>,
    source: Option<&debug::Source>,
) -> Result<NativeIr, String> {
    let exporting = exports.is_some();
    if exporting && source.is_some() {
        return Err(error("native debug info is unsupported for exports"));
    }
    let mut debug = source.map(debug::Module::new).transpose()?;
    let mut debug_bytes = 0;
    // Verification also enforces the shared MIR size limits.
    mir::verify(program).map_err(|error| format!("{}: {}", error.code, error.message))?;
    let main = program
        .functions
        .iter()
        .position(|function| function.name == "main")
        .or(exporting.then_some(0))
        .ok_or_else(|| error("verified MIR main missing"))?;
    let functions = program
        .functions
        .iter()
        .enumerate()
        .map(|(id, function)| (function.name.as_str(), id))
        .collect::<HashMap<_, _>>();
    let mut names = Vec::new();
    let mut name_ids = HashMap::new();
    let blobs = program
        .functions
        .iter()
        .flat_map(|function| function.blocks.iter())
        .flat_map(|block| block.instructions.iter())
        .filter_map(|instruction| match &instruction.kind {
            Operation::Blob { bytes, text } => Some((bytes.clone(), *text)),
            _ => None,
        })
        .collect::<Vec<_>>();
    for (function_id, function) in program.functions.iter().enumerate() {
        for name in &function.declaration_order {
            let id = names.len();
            names.push(name.clone());
            name_ids.insert((function_id, name.clone()), id);
        }
        let id = names.len();
        names.push("<result>".to_owned());
        name_ids.insert((function_id, "<result>".to_owned()), id);
    }
    let foreign = program
        .functions
        .iter()
        .enumerate()
        .map(|(function_id, function)| {
            if exporting || function.foreign_oracle.is_none() {
                return Ok(None);
            }
            let Type::Named(result) = &function.result else {
                return Err(error("verified foreign result missing"));
            };
            let (ok, _) = crate::front::result_parts(result)
                .ok_or_else(|| error("verified foreign Result missing"))?;
            let args = function
                .params
                .iter()
                .map(|param| {
                    let tag = foreign_tag(&param.ty)?;
                    let mode = match param.mode {
                        None => 0,
                        Some(crate::front::Mode::Read) => 1,
                        Some(crate::front::Mode::Take) => 2,
                        Some(crate::front::Mode::Mut) => {
                            return Err(error("mutable foreign argument is unsupported"));
                        }
                    };
                    let name_id = *name_ids
                        .get(&(function_id, param.name.clone()))
                        .ok_or_else(|| error("foreign parameter name ID missing"))?;
                    Ok((tag, mode, name_id))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Some(ForeignSpec {
                symbol: function.name.clone(),
                result_tag: foreign_tag(&ok)?,
                args,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut output = String::from(
        "; Invar native ABI v1: all values i64, arrays are host-owned handles.\n\
         declare i64 @invar_tc_step_v1(i64, i64, i64)\n\
         declare void @invar_tc_reference_v1(i64, i64, i64, i64, i64, i64)\n\
         declare void @invar_tick_v1()\n\
         declare void @invar_enter_v1()\n\
         declare void @invar_leave_v1()\n\
         declare void @invar_trap_v1(i32, i32, i32)\n\
         declare i64 @invar_angle_convert_v1(i64, i32, i32, i32, i32)\n\
         declare void @invar_tool_v1(i32)\n\
         declare i64 @invar_sort_v1(i32, i32, i64, i64, i32, i32)\n\
         declare i64 @invar_spatial_v1(i32, i64, i64, i64, i64, i64, i64, i32, i32)\n\
         declare i64 @invar_array_new_v1(i32, i32)\n\
         declare void @invar_array_set_v1(i64, i32, i64)\n\
         declare void @invar_object_set_v1(i64, i32, i64, i32)\n\
         declare i64 @invar_object_take_v1(i64, i32)\n\
         declare i64 @invar_array_fill_v1(i32, i64, i64)\n\
         declare i64 @invar_array_index_v1(i64, i64, i32, i32)\n\
         declare i64 @invar_variant_payload_v1(i64, i32)\n\
         declare i64 @invar_variant_take_payload_v1(i64, i32)\n\
         declare void @invar_array_store_v1(i64, i64, i64, i32, i32)\n\
         declare i64 @invar_array_len_v1(i64)\n\
         declare i64 @invar_blob_new_v1(i32)\n\
         declare i64 @invar_text_v1(i32, i64, i64, i64, i32, i32)\n\
         declare i64 @invar_fsum_v1(i32, i64, i64, i32, i32)\n\
         declare i64 @invar_foreign_call_v1(i32, i64, i64, i64, i64, i64, i64, i64, i64)\n\
         declare i64 @invar_view_new_v1(i64, i64, i64, i64, i32, i32, i32)\n\
         declare void @invar_view_drop_v1(i64)\n\
         declare void @invar_array_drop_v1(i64, i32)\n\n",
    );
    output.push_str(&float_declarations());
    if exporting {
        output = String::from(
            "declare float @llvm.fma.f32(float, float, float)\n\
             declare hidden i32 @invar_tick_v1()\n\
             declare hidden i32 @invar_enter_v1()\n\
             declare hidden void @invar_leave_v1()\n\
             declare hidden void @invar_trap_v1(i32, i32, i32)\n\
             declare hidden void @invar_tool_v1(i32)\n\
             declare hidden ptr @invar_frame_new_v1(i32)\n\
             declare hidden i32 @invar_array_index_v1(i64, i64, i32, i32, ptr)\n\
             declare hidden i32 @invar_array_len_v1(i64, ptr)\n\
             declare hidden i32 @invar_array_store_v1(i64, i64, i64, i32, i32)\n\n",
        );
    }
    output.push_str(&trig_helpers(program, &output));
    let mut next_blob = 0;
    for (function_id, function) in program.functions.iter().enumerate() {
        if function.foreign_oracle.is_some()
            || exports.is_some_and(|(_, reachable, _)| !reachable.contains(&function_id))
        {
            continue;
        }
        let helper = crate::check::is_frame_helper(&function.name);
        let mut args = (0..function.params.len())
            .map(|id| format!("i64 %a{id}"))
            .collect::<Vec<_>>();
        if helper {
            args.extend(["i32 %site_start".to_owned(), "i32 %site_end".to_owned()]);
        }
        let args = args.join(", ");
        let result_type = if exporting { "{ i32, i64 }" } else { "i64" };
        let linkage = if exporting { "internal " } else { "" };
        let scope = debug
            .as_mut()
            .map(|debug| debug.subprogram(function, helper));
        let attachment = scope.map_or(String::new(), |scope| format!(" !dbg !{scope}"));
        writeln!(
            output,
            "define {linkage}{result_type} @invar_f{function_id}({args}) \"denormal-fp-math\"=\"ieee,ieee\" \"denormal-fp-math-f32\"=\"ieee,ieee\"{attachment} {{"
        )
        .map_err(|_| error("native IR formatting failed"))?;
        let locals = function
            .declaration_order
            .iter()
            .enumerate()
            .map(|(id, name)| (name.as_str(), id))
            .collect::<HashMap<_, _>>();
        let value_types = function
            .blocks
            .iter()
            .flat_map(|block| {
                block.params.iter().cloned().chain(
                    block
                        .instructions
                        .iter()
                        .filter_map(|instruction| instruction.output.clone()),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut ir = Ir {
            text: String::new(),
            temp: 0,
            block: 0,
            locals,
            names: &name_ids,
            functions: &functions,
            value_types,
            function_id,
            program,
            next_blob,
            exporting,
            helper,
            location: String::new(),
            debug_bytes: 0,
        };
        debug_bytes += attachment.len();
        locate(&mut ir, debug.as_mut(), scope, function.span);
        ir.label("entry");
        if exporting {
            ir.checked_helper("invar_enter_v1");
        } else {
            ir.line("call void @invar_enter_v1()");
        }
        if exporting {
            let slots = function.declaration_order.len() + ir.value_types.len();
            ir.line(format!(
                "%frame = call ptr @invar_frame_new_v1(i32 {slots})"
            ));
            let absent = ir.tmp();
            ir.line(format!("{absent} = icmp eq ptr %frame, null"));
            ir.tool_if(&absent, 3);
        }
        for (slot, _) in function.declaration_order.iter().enumerate() {
            if exporting {
                ir.line(format!(
                    "%s{slot} = getelementptr i64, ptr %frame, i64 {slot}"
                ));
            } else {
                ir.line(format!("%s{slot} = alloca i64"));
            }
            ir.line(format!("store i64 0, ptr %s{slot}"));
        }
        for (value_slot, (id, _)) in function
            .blocks
            .iter()
            .flat_map(|block| block.params.iter())
            .chain(
                function
                    .blocks
                    .iter()
                    .flat_map(|block| block.instructions.iter())
                    .filter_map(|instruction| instruction.output.as_ref()),
            )
            .enumerate()
        {
            if exporting {
                let slot = function.declaration_order.len() + value_slot;
                ir.line(format!(
                    "%v{id} = getelementptr i64, ptr %frame, i64 {slot}"
                ));
            } else {
                ir.line(format!("%v{id} = alloca i64"));
            }
            ir.line(format!("store i64 0, ptr %v{id}"));
        }
        if let (Some(debug), Some(scope)) = (debug.as_mut(), scope) {
            let records = debug.declarations(function, scope, helper)?;
            ir.debug_bytes += records.len();
            ir.text.push_str(&records);
        }
        for (param_id, param) in function.params.iter().enumerate() {
            let slot = ir.local(&param.name)?;
            ir.line(format!("store i64 %a{param_id}, ptr %s{slot}"));
        }
        ir.line(format!("br label %b{}", function.entry));
        for (block_id, block) in function.blocks.iter().enumerate() {
            ir.label(format!("b{block_id}"));
            for instruction in &block.instructions {
                locate(&mut ir, debug.as_mut(), scope, instruction.span);
                ir.operation(
                    function,
                    &instruction.kind,
                    instruction.output.as_ref().map(|(id, ty)| (*id, ty)),
                    instruction.span,
                )?;
            }
            let end = block
                .end
                .as_ref()
                .ok_or_else(|| error("verified MIR terminator missing"))?;
            let (Terminator::Goto { span, .. }
            | Terminator::Branch { span, .. }
            | Terminator::Return { span, .. }) = end;
            locate(&mut ir, debug.as_mut(), scope, *span);
            ir.terminator(function, end)?;
        }
        output.push_str(&ir.text);
        debug_bytes += ir.debug_bytes;
        next_blob = ir.next_blob;
        output.push_str("}\n\n");
    }
    if let Some((selected, _, namespace)) = exports {
        for (ordinal, id) in selected.iter().enumerate() {
            let arguments = (0..program.functions[*id].params.len())
                .map(|n| format!("i64 %a{n}"))
                .collect::<Vec<_>>()
                .join(", ");
            let suffix = if arguments.is_empty() {
                String::new()
            } else {
                format!(", {arguments}")
            };
            writeln!(output, "define hidden i32 @invar_entry_{ordinal}(ptr %out{suffix}) {{\nentry:\n  %pair = call {{ i32, i64 }} @invar_f{id}({arguments})\n  %status = extractvalue {{ i32, i64 }} %pair, 0\n  %ok = icmp eq i32 %status, 0\n  br i1 %ok, label %success, label %failure\nsuccess:\n  %value = extractvalue {{ i32, i64 }} %pair, 1\n  store i64 %value, ptr %out\n  ret i32 0\nfailure:\n  ret i32 %status\n}}")
                .map_err(|_| error("native IR formatting failed"))?;
        }
        output = output.replace("@invar_", &format!("@{namespace}"));
    } else {
        writeln!(output,
            "define i64 @invar_entry_v1() {{\nentry:\n  %result = call i64 @invar_f{main}()\n  ret i64 %result\n}}")
            .map_err(|_| error("native IR formatting failed"))?;
    }
    // The code limit excludes debug metadata, so `--debug` refuses no program
    // the plain build accepts; the metadata has its own bound.
    if output.len() - debug_bytes > MAX_IR_BYTES {
        return Err(error("native generated IR limit exceeded"));
    }
    if let Some(debug) = &debug {
        output.push_str(&debug.finish());
        if output.len() > 4 * MAX_IR_BYTES {
            return Err(error("native debug IR limit exceeded"));
        }
    }
    Ok((output, names, blobs, foreign))
}

/// Debug builds: the location of the next IR lines.
fn locate(ir: &mut Ir, debug: Option<&mut debug::Module>, scope: Option<usize>, span: Span) {
    if let (Some(debug), Some(scope)) = (debug, scope) {
        ir.location = format!(", !dbg !{}", debug.location(span, scope, ir.helper));
    }
}

/// `invar-trig-f32-v1` helper definitions (from `invar_device::trig_f32`, the
/// operation sequence's single source of truth) for the intrinsics `program`
/// calls, with any LLVM declarations `header` lacks. Empty when unused, so
/// programs without these intrinsics keep byte-identical IR.
fn trig_helpers(program: &mir::Program, header: &str) -> String {
    let called = |names: &[&str]| {
        program.functions.iter().any(|function| {
            function
                .blocks
                .iter()
                .flat_map(|block| block.instructions.iter())
                .any(|instruction| {
                    matches!(&instruction.kind, Operation::Call { name, .. } if names.contains(&name.as_str()))
                })
        })
    };
    let trig = called(&["sin", "cos"]);
    let root = called(&["sqrt"]);
    let mut text = String::new();
    if trig && !header.contains("declare float @llvm.fma.f32(") {
        text.push_str("declare float @llvm.fma.f32(float, float, float)\n");
    }
    if root && !header.contains("declare float @llvm.sqrt.f32(") {
        text.push_str("declare float @llvm.sqrt.f32(float)\n");
    }
    if trig || root {
        text.push_str(&invar_device::trig_f32::helpers(
            invar_device::trig_f32::Lang::Llvm,
            trig,
            root,
        ));
        text.push('\n');
    }
    text
}
