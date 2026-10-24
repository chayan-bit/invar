impl Ir<'_> {
    // Export-mode callbacks return status instead of terminating the process.
    // Each failure edge returns before a source value is consumed or stored.
    fn propagate_status(&mut self, status: &str) {
        let bad = self.fresh("failure");
        let good = self.fresh("continue");
        let failed = self.tmp();
        self.line(format!("{failed} = icmp ne i32 {status}, 0"));
        self.line(format!("br i1 {failed}, label %{bad}, label %{good}"));
        self.label(bad);
        self.line("ret { i32, i64 } { i32 1, i64 0 }");
        self.label(good);
    }

    fn checked_helper(&mut self, name: &str) {
        let status = self.tmp();
        self.line(format!("{status} = call i32 @{name}()"));
        self.propagate_status(&status);
    }

    fn line(&mut self, line: impl AsRef<str>) {
        self.text.push_str("  ");
        self.text.push_str(line.as_ref());
        self.text.push_str(&self.location);
        self.debug_bytes += self.location.len();
        self.text.push('\n');
    }

    fn label(&mut self, label: impl AsRef<str>) {
        self.text.push_str(label.as_ref());
        self.text.push_str(":\n");
    }

    fn tmp(&mut self) -> String {
        let id = self.temp;
        self.temp += 1;
        format!("%t{id}")
    }

    fn fresh(&mut self, prefix: &str) -> String {
        let id = self.block;
        self.block += 1;
        format!("{prefix}{id}")
    }

    fn load(&mut self, value: ValueId) -> String {
        let result = self.tmp();
        self.line(format!("{result} = load i64, ptr %v{value}"));
        result
    }

    fn local(&self, name: &str) -> Result<usize, String> {
        self.locals
            .get(name)
            .copied()
            .ok_or_else(|| error("verified MIR local missing in native lowering"))
    }

    fn name_id(&self, name: &str) -> Result<usize, String> {
        self.names
            .get(&(self.function_id, name.to_owned()))
            .copied()
            .ok_or_else(|| error("verified MIR cleanup name missing"))
    }

    /// Source span of a trap or host check at `at`: the hidden caller site
    /// inside a frame helper, whose own spans are synthetic.
    fn site(&self, at: Span) -> Result<(String, String), String> {
        if self.helper {
            return Ok(("%site_start".to_owned(), "%site_end".to_owned()));
        }
        let (start, end) = span(at)?;
        Ok((start.to_string(), end.to_string()))
    }

    fn trap_if(&mut self, predicate: &str, category: u32, at: Span) -> Result<(), String> {
        let (start, end) = self.site(at)?;
        let bad = self.fresh("trap");
        let good = self.fresh("ok");
        self.line(format!("br i1 {predicate}, label %{bad}, label %{good}"));
        self.label(&bad);
        self.line(format!(
            "call void @invar_trap_v1(i32 {category}, i32 {start}, i32 {end})"
        ));
        if self.exporting {
            self.line("ret { i32, i64 } { i32 1, i64 0 }");
        } else {
            self.line("unreachable");
        }
        self.label(&good);
        Ok(())
    }

    fn tool_if(&mut self, predicate: &str, code: u32) {
        let bad = self.fresh("tool");
        let good = self.fresh("ok");
        self.line(format!("br i1 {predicate}, label %{bad}, label %{good}"));
        self.label(&bad);
        self.line(format!("call void @invar_tool_v1(i32 {code})"));
        if self.exporting {
            self.line("ret { i32, i64 } { i32 1, i64 0 }");
        } else {
            self.line("unreachable");
        }
        self.label(&good);
    }

    fn range_check(&mut self, value: &str, signed: bool, at: Span) -> Result<(), String> {
        let low = self.tmp();
        let high = self.tmp();
        let bad = self.tmp();
        if signed {
            self.line(format!("{low} = icmp slt i64 {value}, -2147483648"));
            self.line(format!("{high} = icmp sgt i64 {value}, 2147483647"));
            self.line(format!("{bad} = or i1 {low}, {high}"));
        } else {
            self.line(format!("{low} = icmp slt i64 {value}, 0"));
            self.line(format!("{high} = icmp ugt i64 {value}, 4294967295"));
            self.line(format!("{bad} = or i1 {low}, {high}"));
        }
        self.trap_if(&bad, 1, at)
    }

    fn cleanup(&mut self, function: &Function, name: &str) -> Result<(), String> {
        let binding = function
            .locals
            .get(name)
            .ok_or_else(|| error("verified MIR cleanup place missing"))?;
        if matches!(binding.ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool | Type::Named(_))
            && !matches!(
                binding.mode,
                Some(crate::front::Mode::Read | crate::front::Mode::Mut)
            )
        {
            let slot = self.local(name)?;
            let value = self.tmp();
            let name_id = self.name_id(name)?;
            self.line(format!("{value} = load i64, ptr %s{slot}"));
            if matches!(&binding.ty, Type::Named(view) if crate::front::view_parts(view).is_some())
            {
                self.line(format!("call void @invar_view_drop_v1(i64 {value})"));
            } else {
                self.line(format!(
                    "call void @invar_array_drop_v1(i64 {value}, i32 {name_id})"
                ));
            }
            self.line(format!("store i64 0, ptr %s{slot}"));
        }
        Ok(())
    }

    fn operation(
        &mut self,
        function: &Function,
        op: &Operation,
        output: Option<(ValueId, &Type)>,
        at: Span,
    ) -> Result<(), String> {
        let mut result = None;
        match op {
            Operation::Tick => {
                if self.exporting {
                    self.checked_helper("invar_tick_v1");
                } else {
                    self.line("call void @invar_tick_v1()");
                }
            }
            Operation::I32(value) => result = Some(value.to_string()),
            Operation::F32(value) => result = Some(value.to_bits().to_string()),
            Operation::F64(value) => result = Some((value.to_bits() as i64).to_string()),
            Operation::U32(value) => result = Some(value.to_string()),
            Operation::Bool(value) => result = Some(u8::from(*value).to_string()),
            Operation::Blob { .. } => {
                let id = self.next_blob;
                self.next_blob += 1;
                let value = self.tmp();
                self.line(format!("{value} = call i64 @invar_blob_new_v1(i32 {id})"));
                result = Some(value);
            }
            Operation::Read(name)
            | Operation::ArrayRef { name, .. }
            | Operation::RecordRef { name, .. } => {
                let slot = self.local(name)?;
                let value = self.tmp();
                self.line(format!("{value} = load i64, ptr %s{slot}"));
                result = Some(value);
            }
            Operation::Move(name) => {
                let slot = self.local(name)?;
                let value = self.tmp();
                self.line(format!("{value} = load i64, ptr %s{slot}"));
                self.line(format!("store i64 0, ptr %s{slot}"));
                result = Some(value);
            }
            Operation::MoveField { name, index } => {
                let slot = self.local(name)?;
                let handle = self.tmp();
                self.line(format!("{handle} = load i64, ptr %s{slot}"));
                let value = self.tmp();
                self.line(format!(
                    "{value} = call i64 @invar_object_take_v1(i64 {handle}, i32 {index})"
                ));
                result = Some(value);
            }
            Operation::Array(items) => {
                if items.len() > crate::array_limits::MAX_ARRAY_LEN {
                    return Err(error("native array literal exceeds resource limit"));
                }
                let kind = match output.map(|(_, ty)| ty) {
                    Some(Type::ArrayI32) => 1,
                    Some(Type::ArrayU32) | Some(Type::ArrayF32) | Some(Type::ArrayBool) => 2,
                    Some(Type::ArrayF64) => 4,
                    _ => return Err(error("verified array type missing")),
                };
                let handle = self.tmp();
                self.line(format!(
                    "{handle} = call i64 @invar_array_new_v1(i32 {kind}, i32 {})",
                    items.len()
                ));
                for (index, item) in items.iter().enumerate() {
                    let value = self.load(*item);
                    self.line(format!(
                        "call void @invar_array_set_v1(i64 {handle}, i32 {index}, i64 {value})"
                    ));
                }
                result = Some(handle);
            }
            Operation::Record { name, fields } => {
                let length = fields.len();
                let declaration = self
                    .program
                    .records
                    .iter()
                    .find(|record| record.name == *name)
                    .ok_or_else(|| error("verified record declaration missing"))?;
                let handle = self.tmp();
                self.line(format!(
                    "{handle} = call i64 @invar_array_new_v1(i32 3, i32 {length})"
                ));
                for (index, item) in fields {
                    let value = self.load(*item);
                    let owned = u8::from(matches!(
                        declaration.fields[*index].ty,
                        Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool | Type::Named(_)
                    ));
                    self.line(format!("call void @invar_object_set_v1(i64 {handle}, i32 {index}, i64 {value}, i32 {owned})"));
                }
                result = Some(handle);
            }
            Operation::Variant { name, tag, payload } => {
                let declaration = self
                    .program
                    .variants
                    .iter()
                    .find(|variant| variant.name == *name)
                    .ok_or_else(|| error("verified variant declaration missing"))?;
                let handle = self.tmp();
                self.line(format!(
                    "{handle} = call i64 @invar_array_new_v1(i32 3, i32 2)"
                ));
                self.line(format!(
                    "call void @invar_object_set_v1(i64 {handle}, i32 0, i64 {tag}, i32 0)"
                ));
                if let Some(payload) = payload {
                    let value = self.load(*payload);
                    let owned =
                        u8::from(declaration.arms[*tag].payload.as_ref().is_some_and(|ty| {
                            matches!(ty, Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::ArrayF64 | Type::ArrayBool | Type::Named(_))
                        }));
                    self.line(format!("call void @invar_object_set_v1(i64 {handle}, i32 1, i64 {value}, i32 {owned})"));
                }
                result = Some(handle);
            }
            Operation::VariantTag(value) => {
                let handle = self.load(*value);
                let tag = self.tmp();
                let (start, end) = self.site(at)?;
                self.line(format!("{tag} = call i64 @invar_array_index_v1(i64 {handle}, i64 0, i32 {start}, i32 {end})"));
                result = Some(tag);
            }
            Operation::VariantPayload { value, tag } => {
                let handle = self.load(*value);
                let raw = self.tmp();
                self.line(format!(
                    "{raw} = call i64 @invar_variant_payload_v1(i64 {handle}, i32 {tag})"
                ));
                result = if matches!(output.map(|(_, ty)| ty), Some(Type::I32)) {
                    let narrow = self.tmp();
                    let signed = self.tmp();
                    self.line(format!("{narrow} = trunc i64 {raw} to i32"));
                    self.line(format!("{signed} = sext i32 {narrow} to i64"));
                    Some(signed)
                } else {
                    Some(raw)
                };
            }
            Operation::VariantTakePayload { value, tag } => {
                let handle = self.load(*value);
                let raw = self.tmp();
                self.line(format!(
                    "{raw} = call i64 @invar_variant_take_payload_v1(i64 {handle}, i32 {tag})"
                ));
                result = if matches!(output.map(|(_, ty)| ty), Some(Type::I32)) {
                    let narrow = self.tmp();
                    let signed = self.tmp();
                    self.line(format!("{narrow} = trunc i64 {raw} to i32"));
                    self.line(format!("{signed} = sext i32 {narrow} to i64"));
                    Some(signed)
                } else {
                    Some(raw)
                };
            }
            Operation::DropValue(value) => {
                let handle = self.load(*value);
                let name = self.name_id("<result>")?;
                self.line(format!(
                    "call void @invar_array_drop_v1(i64 {handle}, i32 {name})"
                ));
            }
            Operation::Field { record, index } => {
                let handle = self.load(*record);
                let raw = self.tmp();
                let (start, end) = self.site(at)?;
                self.line(format!("{raw} = call i64 @invar_array_index_v1(i64 {handle}, i64 {index}, i32 {start}, i32 {end})"));
                result = if matches!(output.map(|(_, ty)| ty), Some(Type::I32)) {
                    let narrow = self.tmp();
                    let signed = self.tmp();
                    self.line(format!("{narrow} = trunc i64 {raw} to i32"));
                    self.line(format!("{signed} = sext i32 {narrow} to i64"));
                    Some(signed)
                } else {
                    Some(raw)
                };
            }
            Operation::Unary {
                op,
                value,
                operator,
            } => {
                let input = self.load(*value);
                match op {
                    UnaryOp::Not => {
                        let compared = self.tmp();
                        let converted = self.tmp();
                        self.line(format!("{compared} = icmp eq i64 {input}, 0"));
                        self.line(format!("{converted} = zext i1 {compared} to i64"));
                        result = Some(converted);
                    }
                    UnaryOp::Neg if float_format(self.value_types.get(value)).is_some() => {
                        let format = float_format(self.value_types.get(value))
                            .ok_or_else(|| error("verified float negation type missing"))?;
                        result = Some(self.negate_float(format, &input));
                    }
                    UnaryOp::Neg => {
                        let negated = self.tmp();
                        self.line(format!("{negated} = sub i64 0, {input}"));
                        self.range_check(&negated, true, *operator)?;
                        result = Some(negated);
                    }
                }
            }
            Operation::Binary {
                op,
                left,
                right,
                operator,
            } => {
                let a = self.load(*left);
                let b = self.load(*right);
                if let Some(format) = float_format(self.value_types.get(left)) {
                    result = Some(self.binary_float(format, *op, &a, &b)?);
                } else {
                let signed = match output.map(|(_, ty)| ty) {
                    Some(Type::I32) => true,
                    Some(Type::U32) => false,
                    Some(Type::Bool) => self.value_types.get(left) == Some(&Type::I32),
                    _ => return Err(error("verified binary result type missing")),
                };
                let value = self.tmp();
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                        let opcode = match op {
                            BinaryOp::Add => "add",
                            BinaryOp::Sub => "sub",
                            _ => "mul",
                        };
                        self.line(format!("{value} = {opcode} i64 {a}, {b}"));
                        self.range_check(&value, signed, *operator)?;
                        result = Some(value);
                    }
                    BinaryOp::Div | BinaryOp::Rem => {
                        let zero = self.tmp();
                        self.line(format!("{zero} = icmp eq i64 {b}, 0"));
                        self.trap_if(&zero, 2, *operator)?;
                        let opcode = match (op, signed) {
                            (BinaryOp::Div, true) => "sdiv",
                            (BinaryOp::Div, false) => "udiv",
                            (BinaryOp::Rem, true) => "srem",
                            _ => "urem",
                        };
                        self.line(format!("{value} = {opcode} i64 {a}, {b}"));
                        if *op == BinaryOp::Div && signed {
                            self.range_check(&value, true, *operator)?;
                        }
                        result = Some(value);
                    }
                    BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge => {
                        let condition = match op {
                            BinaryOp::Eq => "eq",
                            BinaryOp::Ne => "ne",
                            BinaryOp::Lt if signed => "slt",
                            BinaryOp::Le if signed => "sle",
                            BinaryOp::Gt if signed => "sgt",
                            BinaryOp::Ge if signed => "sge",
                            BinaryOp::Lt => "ult",
                            BinaryOp::Le => "ule",
                            BinaryOp::Gt => "ugt",
                            _ => "uge",
                        };
                        let compared = self.tmp();
                        self.line(format!("{compared} = icmp {condition} i64 {a}, {b}"));
                        self.line(format!("{value} = zext i1 {compared} to i64"));
                        result = Some(value);
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        return Err(error("short circuit operation must be MIR branches"));
                    }
                }
                }
            }
            Operation::Index { array, index } if self.exporting => {
                // HA-1.2: status-returning host-array runtime; the element is
                // written to this instruction's own frame slot.
                let array = self.load(*array);
                let index = self.load(*index);
                let (start, end) = self.site(at)?;
                let id = output.map(|(id, _)| id).ok_or_else(|| error("verified MIR index output missing"))?;
                let status = self.tmp();
                self.line(format!(
                    "{status} = call i32 @invar_array_index_v1(i64 {array}, i64 {index}, i32 {start}, i32 {end}, ptr %v{id})"
                ));
                self.propagate_status(&status);
                result = Some(self.load(id));
            }
            Operation::Index { array, index } => {
                let array = self.load(*array);
                let index = self.load(*index);
                let (start, end) = self.site(at)?;
                let value = self.tmp();
                self.line(format!(
                    "{value} = call i64 @invar_array_index_v1(i64 {array}, i64 {index}, i32 {start}, i32 {end})"
                ));
                result = Some(value);
            }
            Operation::Length(name) if self.exporting => {
                let slot = self.local(name)?;
                let handle = self.tmp();
                let id = output.map(|(id, _)| id).ok_or_else(|| error("verified MIR length output missing"))?;
                let status = self.tmp();
                self.line(format!("{handle} = load i64, ptr %s{slot}"));
                self.line(format!(
                    "{status} = call i32 @invar_array_len_v1(i64 {handle}, ptr %v{id})"
                ));
                self.propagate_status(&status);
                result = Some(self.load(id));
            }
            Operation::Length(name) => {
                let slot = self.local(name)?;
                let handle = self.tmp();
                let length = self.tmp();
                self.line(format!("{handle} = load i64, ptr %s{slot}"));
                self.line(format!(
                    "{length} = call i64 @invar_array_len_v1(i64 {handle})"
                ));
                result = Some(length);
            }
            Operation::Call { name, args } => {
                let arg_types = args.iter().map(|id| self.value_types.get(id).cloned()).collect::<Vec<_>>();
                let sort_kind = if crate::check::sort_modes(name).is_some() {
                    if self.exporting { return Err(error("sort calls are not yet supported in native shared-library exports")); }
                    match args.first().and_then(|id| self.value_types.get(id)) {
                        Some(Type::ArrayU32) => 0,
                        Some(Type::ArrayI32) => 1,
                        Some(Type::ArrayF32) => 2,
                        _ => return Err(error("verified sort key type missing")),
                    }
                } else { 0 };
                let args = args.iter().map(|id| self.load(*id)).collect::<Vec<_>>();
                let value = self.tmp();
                if let Some(word) = self.float_intrinsic(name, &args, &arg_types, at)? {
                    self.line(format!("{value} = add i64 {word}, 0"));
                    result = Some(value);
                } else {
                match (name.as_str(), args.as_slice()) {
                    (spatial, _) if crate::spatial::result(spatial).is_some() => {
                        if self.exporting { return Err(error("spatial calls are not supported in native shared-library exports")); }
                        let opcode = match spatial { "voxel_grid_build" => 0, "radius_neighbors" => 1, "knn" => 2, "grid_excluded" => 3, _ => return Err(error("spatial opcode missing")) };
                        let mut padded = args.clone();
                        padded.resize(6, "0".into());
                        let arguments = padded.iter().map(|v| format!("i64 {v}")).collect::<Vec<_>>().join(", ");
                        let (start,end) = self.site(at)?;
                        self.line(format!("{value} = call i64 @invar_spatial_v1(i32 {opcode}, {arguments}, i32 {start}, i32 {end})"));
                    }
                    (sort, [keys, ..]) if crate::check::sort_modes(sort).is_some() => {
                        let opcode = match sort { "sort" => 0, "sort_by_key" => 1, "argsort" => 2, "segmented_sort" => 3, _ => return Err(error("sort opcode missing")) };
                        let other = args.get(1).map(String::as_str).unwrap_or("0");
                        let (start,end) = self.site(at)?;
                        self.line(format!("{value} = call i64 @invar_sort_v1(i32 {opcode}, i32 {sort_kind}, i64 {keys}, i64 {other}, i32 {start}, i32 {end})"));
                    }
                    (scale, [input]) if crate::check::scale_call(scale).is_some() => {
                        let (numerator, denominator) = crate::check::scale_call(scale)
                            .ok_or_else(|| error("verified scale factor missing"))?;
                        let product = self.tmp();
                        let remainder = self.tmp();
                        let inexact = self.tmp();
                        let quotient = self.tmp();
                        self.line(format!("{product} = mul i64 {input}, {numerator}"));
                        self.line(format!("{remainder} = srem i64 {product}, {denominator}"));
                        self.line(format!("{inexact} = icmp ne i64 {remainder}, 0"));
                        self.trap_if(&inexact, 4, at)?;
                        self.line(format!("{quotient} = sdiv i64 {product}, {denominator}"));
                        self.range_check(
                            &quotient,
                            output.map(|(_, ty)| ty) == Some(&Type::I32),
                            at,
                        )?;
                        self.line(format!("{value} = add i64 {quotient}, 0"));
                    }
                    (angle, [input]) if crate::check::angle_call(angle).is_some() => {
                        let direction = u8::from(crate::check::angle_call(angle) == Some(true));
                        let signed = u8::from(output.map(|(_, ty)| ty) == Some(&Type::I32));
                        let (start, end) = self.site(at)?;
                        self.line(format!("{value} = call i64 @invar_angle_convert_v1(i64 {input}, i32 {direction}, i32 {signed}, i32 {start}, i32 {end})"));
                    }
                    ("wrapping_add", [a, b]) | ("wrapping_mul", [a, b]) => {
                        let opcode = if name == "wrapping_add" { "add" } else { "mul" };
                        let wide = self.tmp();
                        let narrow = self.tmp();
                        self.line(format!("{wide} = {opcode} i64 {a}, {b}"));
                        self.line(format!("{narrow} = trunc i64 {wide} to i32"));
                        self.line(format!("{value} = zext i32 {narrow} to i64"));
                    }
                    ("tc_mul_add", [a,b,c]) => {
                        if self.exporting { return Err(error("TC native export is unsupported")); }
                        self.line(format!("{value} = call i64 @invar_tc_step_v1(i64 {a}, i64 {b}, i64 {c})"));
                    }
                    ("filled_f32", [len,item]) => {
                        self.line(format!("{value} = call i64 @invar_array_fill_v1(i32 2, i64 {len}, i64 {item})"));
                    }
                    ("filled_f64", [len,item]) => {
                        self.line(format!("{value} = call i64 @invar_array_fill_v1(i32 4, i64 {len}, i64 {item})"));
                    }
                    ("filled_i32", [len, item]) | ("filled_u32", [len, item]) => {
                        let kind = if name == "filled_i32" { 1 } else { 2 };
                        self.line(format!(
                            "{value} = call i64 @invar_array_fill_v1(i32 {kind}, i64 {len}, i64 {item})"
                        ));
                    }
                    ("view" | "view_mut", [owner, start, end])
                    | ("view" | "view_mut", [owner, start, end, _]) => {
                        let stride = args.get(3).cloned().unwrap_or_else(|| "1".to_owned());
                        let mutable = u8::from(name == "view_mut");
                        let (span_start, span_end) = self.site(at)?;
                        self.line(format!("{value} = call i64 @invar_view_new_v1(i64 {owner}, i64 {start}, i64 {end}, i64 {stride}, i32 {mutable}, i32 {span_start}, i32 {span_end})"));
                    }
                    ("bytes_slice", [owner, start, end]) | ("utf8_slice", [owner, start, end]) => {
                        let kind = if name == "bytes_slice" { 1 } else { 2 };
                        let (span_start, span_end) = self.site(at)?;
                        self.line(format!("{value} = call i64 @invar_text_v1(i32 {kind}, i64 {owner}, i64 {start}, i64 {end}, i32 {span_start}, i32 {span_end})"));
                    }
                    (reduction, [a]) | (reduction, [a, _])
                        if crate::fsum::Kind::from_name(reduction).is_some() =>
                    {
                        if self.exporting {
                            return Err(error("native exports do not admit float reductions"));
                        }
                        let code = crate::fsum::Kind::from_name(reduction)
                            .ok_or_else(|| error("verified reduction missing"))?
                            .code();
                        let b = args.get(1).unwrap_or(a);
                        let (span_start, span_end) = self.site(at)?;
                        self.line(format!("{value} = call i64 @invar_fsum_v1(i32 {code}, i64 {a}, i64 {b}, i32 {span_start}, i32 {span_end})"));
                    }
                    ("utf8_decode", [owner]) | ("utf8_encode", [owner]) => {
                        let kind = if name == "utf8_decode" { 3 } else { 4 };
                        let (span_start, span_end) = self.site(at)?;
                        self.line(format!("{value} = call i64 @invar_text_v1(i32 {kind}, i64 {owner}, i64 0, i64 0, i32 {span_start}, i32 {span_end})"));
                    }
                    _ => {
                        let id = self
                            .functions
                            .get(name.as_str())
                            .ok_or_else(|| error("verified native callee missing"))?;
                        if self.program.functions[*id].foreign_oracle.is_some() {
                            if args.len() > 8 {
                                return Err(error("verified foreign arity exceeds eight"));
                            }
                            let mut padded = args.clone();
                            padded.resize(8, "0".to_owned());
                            let arguments = padded
                                .iter()
                                .map(|arg| format!("i64 {arg}"))
                                .collect::<Vec<_>>()
                                .join(", ");
                            self.line(format!(
                                "{value} = call i64 @invar_foreign_call_v1(i32 {id}, {arguments})"
                            ));
                        } else {
                            let mut arguments = args
                                .iter()
                                .map(|arg| format!("i64 {arg}"))
                                .collect::<Vec<_>>();
                            if crate::check::is_frame_helper(name) {
                                let (start, end) = self.site(at)?;
                                arguments.extend([format!("i32 {start}"), format!("i32 {end}")]);
                            }
                            let arguments = arguments.join(", ");
                            if self.exporting {
                                let pair = self.tmp();
                                let status = self.tmp();
                                self.line(format!(
                                    "{pair} = call {{ i32, i64 }} @invar_f{id}({arguments})"
                                ));
                                self.line(format!(
                                    "{status} = extractvalue {{ i32, i64 }} {pair}, 0"
                                ));
                                self.propagate_status(&status);
                                self.line(format!(
                                    "{value} = extractvalue {{ i32, i64 }} {pair}, 1"
                                ));
                            } else {
                                self.line(format!("{value} = call i64 @invar_f{id}({arguments})"));
                            }
                        }
                    }
                }
                result = Some(value);
                }
            }
            Operation::Bind { name, value } => {
                let slot = self.local(name)?;
                let value = self.load(*value);
                self.line(format!("store i64 {value}, ptr %s{slot}"));
            }
            Operation::Store { name, value } => {
                let new = self.load(*value);
                self.cleanup(function, name)?;
                let slot = self.local(name)?;
                self.line(format!("store i64 {new}, ptr %s{slot}"));
            }
            Operation::StoreIndex { name, index, value } => {
                let slot = self.local(name)?;
                let handle = self.tmp();
                let index = self.load(*index);
                let value = self.load(*value);
                let (start, end) = self.site(at)?;
                self.line(format!("{handle} = load i64, ptr %s{slot}"));
                if self.exporting {
                    let status = self.tmp();
                    self.line(format!(
                        "{status} = call i32 @invar_array_store_v1(i64 {handle}, i64 {index}, i64 {value}, i32 {start}, i32 {end})"
                    ));
                    self.propagate_status(&status);
                } else {
                    self.line(format!(
                        "call void @invar_array_store_v1(i64 {handle}, i64 {index}, i64 {value}, i32 {start}, i32 {end})"
                    ));
                }
            }
            Operation::Drop(names) => {
                for name in names {
                    self.cleanup(function, name)?;
                }
            }
            Operation::ForNext(value) => {
                let input = self.load(*value);
                let overflow = self.tmp();
                self.line(format!("{overflow} = icmp eq i64 {input}, 4294967295"));
                self.tool_if(&overflow, 4);
                let next = self.tmp();
                self.line(format!("{next} = add i64 {input}, 1"));
                result = Some(next);
            }
        }
        match (output, result) {
            (Some((id, _)), Some(value)) => self.line(format!("store i64 {value}, ptr %v{id}")),
            (None, None) => {}
            _ => return Err(error("verified MIR operation output mismatch")),
        }
        Ok(())
    }

    fn terminator(&mut self, function: &Function, term: &Terminator) -> Result<(), String> {
        match term {
            Terminator::Goto { target, args, .. } => {
                let values = args.iter().map(|arg| self.load(*arg)).collect::<Vec<_>>();
                for ((id, _), value) in function.blocks[*target].params.iter().zip(values) {
                    self.line(format!("store i64 {value}, ptr %v{id}"));
                }
                self.line(format!("br label %b{target}"));
            }
            Terminator::Branch {
                condition, yes, no, ..
            } => {
                let value = self.load(*condition);
                let truth = self.tmp();
                self.line(format!("{truth} = icmp ne i64 {value}, 0"));
                self.line(format!("br i1 {truth}, label %b{yes}, label %b{no}"));
            }
            Terminator::Return { value, cleanup, .. } => {
                let result = value
                    .map(|value| self.load(value))
                    .unwrap_or_else(|| "0".into());
                if crate::contraction_source::mir_is_tc(function) {
                    if self.exporting || function.params.len() != 5 { return Err(error("TC native signature/export is unsupported")); }
                    self.line(format!("call void @invar_tc_reference_v1(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 {result})"));
                }
                for name in cleanup {
                    self.cleanup(function, name)?;
                }
                self.line("call void @invar_leave_v1()");
                if self.exporting {
                    let pair = self.tmp();
                    self.line(format!(
                        "{pair} = insertvalue {{ i32, i64 }} {{ i32 0, i64 0 }}, i64 {result}, 1"
                    ));
                    self.line(format!("ret {{ i32, i64 }} {pair}"));
                } else {
                    self.line(format!("ret i64 {result}"));
                }
            }
        }
        Ok(())
    }
}
