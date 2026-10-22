impl Parser {
    fn is_system_item(&self) -> bool {
        !self.is_foreign_fn()
            && (matches!(self.current().kind, TokenKind::Keyword("component" | "extern" | "connect" | "flow" | "product" | "system"))
                || matches!(&self.current().kind, TokenKind::Name(name) if matches!(name.as_str(), "robot" | "cell")))
    }

    fn take_word(&mut self, word: &str) -> bool {
        if matches!(&self.current().kind, TokenKind::Name(actual) if actual == word)
            || matches!(&self.current().kind, TokenKind::Keyword(actual) if *actual == word)
        {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_word(&mut self, word: &str) -> Result<(), FrontError> {
        self.take_word(word).then_some(()).ok_or_else(|| self.expected(word))
    }

    fn system_word(&mut self) -> Result<String, FrontError> {
        match self.current().kind {
            TokenKind::Name(_) | TokenKind::Keyword(_) => {
                let token = self.bump();
                match token.kind {
                    TokenKind::Name(name) => Ok(name),
                    TokenKind::Keyword(name) => Ok(name.to_owned()),
                    _ => unreachable!("checked token kind"),
                }
            }
            _ => Err(self.expected("word")),
        }
    }

    fn system_hyphenated(&mut self) -> Result<String, FrontError> {
        let mut name = self.system_word()?;
        while self.take_symbol("-").is_some() {
            name.push('-');
            name.push_str(&self.system_word()?);
        }
        Ok(name)
    }

    fn system_path(&mut self) -> Result<String, FrontError> {
        let mut name = self.system_word()?;
        while self.take_symbol("::").is_some() {
            name.push_str("::");
            name.push_str(&self.system_word()?);
        }
        Ok(name)
    }

    fn system_port_path(&mut self) -> Result<String, FrontError> {
        let mut name = self.system_word()?;
        self.expect_symbol(".")?;
        name.push('.');
        name.push_str(&self.system_word()?);
        Ok(name)
    }

    fn system_string(&mut self) -> Result<String, FrontError> {
        if !matches!(self.current().kind, TokenKind::StringLit(_)) {
            return Err(self.expected("string literal"));
        }
        let TokenKind::StringLit(value) = self.bump().kind else {
            unreachable!("checked token kind")
        };
        Ok(value)
    }

    fn system_bool(&mut self) -> Result<bool, FrontError> {
        let span = self.current().span;
        match self.system_word()?.as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(FrontError::source("S1000", "expected true or false", span)),
        }
    }

    fn system_quantity(&mut self) -> Result<SystemQuantity, FrontError> {
        let start = self.current().span.start;
        let negative = self.take_symbol("-").is_some();
        let token = self.bump();
        let value = match token.kind {
            TokenKind::Integer(value) | TokenKind::Decimal(value) => {
                if negative { format!("-{value}") } else { value }
            }
            _ => return Err(FrontError::source("S1000", "expected exact decimal quantity", token.span)),
        };
        let mut unit = String::new();
        if matches!(&self.current().kind, TokenKind::Name(name) if !matches!(name.as_str(), "latency" | "layout" | "optional" | "link" | "policy" | "within")) {
            unit = self.system_word()?;
            while self.is_symbol("/") || self.is_symbol("*") {
                let symbol = self.bump();
                let TokenKind::Symbol(operator) = symbol.kind else { unreachable!("checked token kind") };
                unit.push_str(operator);
                unit.push_str(&self.system_word()?);
            }
        }
        if unit.is_empty() {
            unit = "one".to_owned();
        }
        Ok(SystemQuantity { value, unit, span: Span { start, end: self.tokens[self.pos - 1].span.end } })
    }

    fn system_resources(&mut self) -> Result<Vec<(String, SystemQuantity)>, FrontError> {
        let mut values = Vec::new();
        loop {
            if values.len() >= 64 {
                return Err(FrontError::source("S1000", "too many resource entries", self.current().span));
            }
            let mut key = self.system_word()?;
            while self.take_symbol(".").is_some() {
                key.push('.');
                key.push_str(&self.system_word()?);
            }
            self.expect_symbol("=")?;
            values.push((key, self.system_quantity()?));
            if self.take_symbol(",").is_none() {
                break;
            }
        }
        self.expect_symbol(";")?;
        Ok(values)
    }

    fn system_clause(&mut self) -> Result<SystemClause, FrontError> {
        let start = self.current().span.start;
        let head = self.system_word()?;
        let clause = match head.as_str() {
            "port" => {
                let direction = self.system_word()?;
                let name = self.system_word()?;
                self.expect_symbol(":")?;
                let ty = self.ty()?;
                self.expect_symbol("@")?;
                let rate = self.system_quantity()?;
                let mut latency = None;
                let mut layout = None;
                let mut bus = None;
                let mut optional = false;
                loop {
                    if self.take_word("latency") {
                        latency = Some(self.system_quantity()?);
                    } else if self.take_word("layout") {
                        self.expect_word("c")?;
                        layout = Some(self.system_string()?);
                    } else if self.take_word("bus") {
                        if bus.replace(self.system_word()?).is_some() {
                            return Err(FrontError::source("S1000", "duplicate port bus", self.current().span));
                        }
                    } else if self.take_word("optional") {
                        optional = true;
                    } else {
                        break;
                    }
                }
                self.expect_symbol(";")?;
                SystemClause::Port { direction, name, ty, rate, latency, optional, layout, bus, span: Span { start, end: self.tokens[self.pos - 1].span.end } }
            }
            "param" => {
                let name = self.system_word()?;
                self.expect_symbol(":")?;
                let ty = self.ty()?;
                self.expect_symbol("=")?;
                if self.take_word("identity") {
                    self.expect_symbol(";")?;
                    return Ok(SystemClause::IdentityParam { name, ty, span: Span { start, end: self.tokens[self.pos - 1].span.end } });
                }
                let value = self.system_quantity()?;
                let range = if self.take_word("range") {
                    let min = self.system_quantity()?;
                    self.expect_symbol("..")?;
                    let max = self.system_quantity()?;
                    Some((min, max))
                } else {
                    None
                };
                self.expect_symbol(";")?;
                SystemClause::Param { name, ty, value, range, span: Span { start, end: self.tokens[self.pos - 1].span.end } }
            }
            "resource" => SystemClause::Resource(self.system_resources()?),
            "capacity" => SystemClause::Capacity(self.system_resources()?),
            "license" => {
                let expression = self.system_string()?;
                let (mut link, mut modified, mut replaceable, mut network_exposed, mut grant) = (None, None, None, None, None);
                while !self.is_symbol(";") {
                    let option = self.system_word()?;
                    let repeated = match option.as_str() {
                        "link" => link.replace(self.system_word()?).is_some(),
                        "modified" => modified.replace(self.system_bool()?).is_some(),
                        "replaceable" => replaceable.replace(self.system_bool()?).is_some(),
                        "network_exposed" => network_exposed.replace(self.system_bool()?).is_some(),
                        "grant" => grant.replace(self.system_string()?).is_some(),
                        _ => return Err(FrontError::source("S1000", "unknown license modifier", self.current().span)),
                    };
                    if repeated { return Err(FrontError::source("S1000", "duplicate license modifier", self.current().span)); }
                }
                self.expect_symbol(";")?;
                SystemClause::License { expression, link, modified, replaceable, network_exposed, grant, span: Span { start, end: self.tokens[self.pos - 1].span.end } }
            }
            "isolation" => { let value = self.system_hyphenated()?; self.expect_symbol(";")?; SystemClause::Isolation(value) }
            "wcet" => { let value = self.system_quantity()?; self.expect_symbol(";")?; SystemClause::Wcet(value) }
            "deadline" => { let value = self.system_quantity()?; self.expect_symbol(";")?; SystemClause::Deadline(value) }
            "impl" => { let kind = self.system_word()?; let target = self.system_path()?; self.expect_symbol(";")?; SystemClause::Impl { kind, target } }
            "abi" => { let kind = self.system_word()?; let prefix = self.system_string()?; self.expect_symbol(";")?; SystemClause::Abi { kind, prefix } }
            "config" => { self.expect_word("c")?; let layout = self.system_string()?; self.expect_symbol(";")?; SystemClause::Config(layout) }
            "input" => { let path = self.system_string()?; self.expect_symbol(";")?; SystemClause::Input(path) }
            "lifecycle" => {
                let mut calls = vec![self.system_word()?];
                while self.take_symbol(",").is_some() { calls.push(self.system_word()?); }
                self.expect_symbol(";")?;
                SystemClause::Lifecycle(calls)
            }
            "golden" => {
                let path = self.system_string()?;
                let mut tolerances = Vec::new();
                if self.take_word("tolerance") {
                    loop {
                        if tolerances.len() >= 8 {
                            return Err(FrontError::source("S1400", "too many golden tolerances", self.current().span));
                        }
                        self.expect_word("abs")?;
                        tolerances.push(self.system_quantity()?);
                        if self.take_symbol(",").is_none() { break; }
                    }
                }
                self.expect_symbol(";")?;
                SystemClause::Golden { path, tolerances }
            }
            "property" => { let value = self.system_string()?; self.expect_symbol(";")?; SystemClause::Property(value) }
            "shim" => { let value = self.system_string()?; self.expect_symbol(";")?; SystemClause::Shim(value) }
            "distribution" => { let value = self.system_hyphenated()?; self.expect_symbol(";")?; SystemClause::Distribution(value) }
            "commercial" => {
                let value = self.system_word()?;
                if !matches!(value.as_str(), "true" | "false") {
                    return Err(FrontError::source("S1000", "commercial must be true or false", self.current().span));
                }
                self.expect_symbol(";")?;
                SystemClause::Commercial(value == "true")
            }
            "robot" => { let value = self.system_path()?; self.expect_symbol(";")?; SystemClause::Robot(value) }
            _ => return Err(FrontError::source("S1000", format!("unknown system clause '{head}'"), Span { start, end: self.tokens[self.pos - 1].span.end })),
        };
        Ok(clause)
    }

    fn robot_decimal(&mut self) -> Result<String, FrontError> {
        let negative = self.take_symbol("-").is_some();
        let token = self.bump();
        match token.kind {
            TokenKind::Integer(value) | TokenKind::Decimal(value) => {
                Ok(if negative { format!("-{value}") } else { value })
            }
            _ => Err(FrontError::source("S1600", "expected exact decimal", token.span)),
        }
    }

    fn robot_decimal_tuple(&mut self, count: usize) -> Result<Vec<String>, FrontError> {
        self.expect_symbol("(")?;
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            if index != 0 {
                self.expect_symbol(",")?;
            }
            values.push(self.robot_decimal()?);
        }
        self.expect_symbol(")")?;
        Ok(values)
    }

    fn robot_pose(&mut self) -> Result<RobotPoseDecl, FrontError> {
        self.expect_word("xyz")?;
        self.expect_symbol("(")?;
        let mut xyz = Vec::with_capacity(3);
        for index in 0..3 {
            if index != 0 {
                self.expect_symbol(",")?;
            }
            xyz.push(self.system_quantity()?);
        }
        self.expect_symbol(")")?;
        self.expect_word("quat")?;
        let quat = self.robot_decimal_tuple(4)?;
        Ok(RobotPoseDecl { xyz, quat })
    }

    fn robot_geometry(&mut self, role: String) -> Result<RobotGeometryDecl, FrontError> {
        let kind = self.system_word()?;
        let (dimensions, path, sha256) = match kind.as_str() {
            "box" => (self.robot_decimal_tuple(3)?, None, None),
            "sphere" => (vec![self.robot_decimal()?], None, None),
            "cylinder" | "capsule" => (self.robot_decimal_tuple(2)?, None, None),
            "mesh" => {
                let path = self.system_string()?;
                self.expect_word("sha256")?;
                (Vec::new(), Some(path), Some(self.system_string()?))
            }
            _ => return Err(FrontError::source("S1600", "unknown geometry kind", self.current().span)),
        };
        if kind != "mesh" {
            self.expect_word("m")?;
        }
        self.expect_symbol(";")?;
        Ok(RobotGeometryDecl { role, kind, dimensions, path, sha256 })
    }

    fn robot_link(&mut self) -> Result<RobotLinkDecl, FrontError> {
        let name = self.system_word()?;
        self.expect_symbol("{")?;
        self.expect_word("parent")?;
        let parent = self.system_word()?;
        self.expect_symbol(";")?;
        self.expect_word("origin")?;
        let origin = self.robot_pose()?;
        self.expect_symbol(";")?;
        self.expect_word("mass")?;
        let mass = self.system_quantity()?;
        self.expect_symbol(";")?;
        let inertial_origin = if self.take_word("inertial_origin") {
            let pose = self.robot_pose()?;
            self.expect_symbol(";")?;
            Some(pose)
        } else {
            None
        };
        self.expect_word("inertia")?;
        let inertia = self.robot_decimal_tuple(6)?;
        self.expect_word("kg")?;
        self.expect_symbol("*")?;
        self.expect_word("m")?;
        self.expect_symbol("*")?;
        self.expect_word("m")?;
        self.expect_symbol(";")?;
        let mut geometries = Vec::new();
        while !self.is_symbol("}") {
            if geometries.len() >= 256 || matches!(self.current().kind, TokenKind::Eof) {
                return Err(FrontError::source("S1600", "link geometry limit or missing brace", self.current().span));
            }
            let role = self.system_word()?;
            if !matches!(role.as_str(), "visual" | "collision") {
                return Err(FrontError::source("S1600", "unknown link clause", self.current().span));
            }
            geometries.push(self.robot_geometry(role)?);
        }
        self.expect_symbol("}")?;
        Ok(RobotLinkDecl { name, parent, origin, mass, inertial_origin, inertia, geometries })
    }

    fn robot_joint(&mut self) -> Result<RobotJointDecl, FrontError> {
        let name = self.system_word()?;
        self.expect_symbol("{")?;
        self.expect_word("kind")?;
        let kind = self.system_word()?;
        if !matches!(kind.as_str(), "revolute" | "prismatic" | "fixed") {
            return Err(FrontError::source("S1600", "unknown joint kind", self.current().span));
        }
        self.expect_symbol(";")?;
        self.expect_word("parent")?;
        let parent = self.system_word()?;
        self.expect_symbol(";")?;
        self.expect_word("child")?;
        let child = self.system_word()?;
        self.expect_symbol(";")?;
        self.expect_word("axis")?;
        let axis = self.robot_decimal_tuple(3)?;
        self.expect_symbol(";")?;
        self.expect_word("position")?;
        let position = if self.take_word("unbounded") {
            None
        } else {
            let lower = self.system_quantity()?;
            self.expect_symbol("..")?;
            Some((lower, self.system_quantity()?))
        };
        self.expect_symbol(";")?;
        self.expect_word("velocity")?;
        let velocity = if self.take_word("unbounded") { None } else { Some(self.system_quantity()?) };
        self.expect_symbol(";")?;
        self.expect_word("effort")?;
        let effort = if self.take_word("unbounded") { None } else { Some(self.system_quantity()?) };
        self.expect_symbol(";")?;
        self.expect_symbol("}")?;
        Ok(RobotJointDecl { name, kind, parent, child, axis, position, velocity, effort })
    }

    fn robot_actuator(&mut self) -> Result<RobotActuatorDecl, FrontError> {
        let name = self.system_word()?;
        self.expect_symbol("{")?;
        self.expect_word("mode")?;
        let mode = self.system_hyphenated()?;
        if !matches!(mode.as_str(), "position" | "velocity" | "motor-torque") {
            return Err(FrontError::source("S1600", "unknown actuator mode", self.current().span));
        }
        self.expect_symbol(";")?;
        let target_kind = self.system_word()?;
        if !matches!(target_kind.as_str(), "joint" | "tendon") {
            return Err(FrontError::source("S1600", "actuator target must be joint or tendon", self.current().span));
        }
        let target = self.system_word()?;
        self.expect_symbol(";")?;
        self.expect_word("gear")?;
        let gear = self.robot_decimal()?;
        self.expect_symbol(";")?;
        self.expect_word("control")?;
        let lower = self.system_quantity()?;
        self.expect_symbol("..")?;
        let upper = self.system_quantity()?;
        self.expect_symbol(";")?;
        let hold = if self.take_word("hold") {
            let hold = self.system_quantity()?;
            self.expect_symbol(";")?;
            Some(hold)
        } else {
            None
        };
        self.expect_symbol("}")?;
        Ok(RobotActuatorDecl { name, mode, target_kind, target, gear, control: (lower, upper), hold })
    }

    fn robot_sensor(&mut self) -> Result<RobotSensorDecl, FrontError> {
        let name = self.system_word()?;
        self.expect_symbol("{")?;
        self.expect_word("kind")?;
        let kind = self.system_hyphenated()?;
        if !matches!(kind.as_str(), "joint-position" | "joint-velocity" | "force-torque" | "camera") {
            return Err(FrontError::source("S1600", "unknown sensor kind", self.current().span));
        }
        self.expect_symbol(";")?;
        self.expect_word("source")?;
        let source = self.system_word()?;
        self.expect_symbol(";")?;
        let intrinsics = if self.take_word("intrinsics") {
            self.expect_symbol("(")?;
            self.expect_word("width")?;
            let width = self.system_quantity()?;
            self.expect_symbol(",")?;
            self.expect_word("height")?;
            let height = self.system_quantity()?;
            self.expect_symbol(",")?;
            self.expect_word("fx")?;
            let fx = self.system_quantity()?;
            self.expect_symbol(",")?;
            self.expect_word("fy")?;
            let fy = self.system_quantity()?;
            self.expect_symbol(",")?;
            self.expect_word("cx")?;
            let cx = self.system_quantity()?;
            self.expect_symbol(",")?;
            self.expect_word("cy")?;
            let cy = self.system_quantity()?;
            self.expect_symbol(",")?;
            self.expect_word("distortion")?;
            if !self.take_word("none") {
                return Err(FrontError::source("S1607", "camera distortion model is unsupported", self.current().span));
            }
            self.expect_symbol(")")?;
            self.expect_symbol(";")?;
            Some(RobotCameraIntrinsicsDecl { width, height, fx, fy, cx, cy })
        } else {
            None
        };
        self.expect_symbol("}")?;
        Ok(RobotSensorDecl { name, kind, source, intrinsics })
    }

    fn robot_model(&mut self) -> Result<RobotModelDecl, FrontError> {
        let name = self.system_word()?;
        self.expect_symbol("{")?;
        self.expect_word("license")?;
        let license = self.system_string()?;
        self.expect_symbol(";")?;
        let asset = if self.take_word("asset") {
            let kind = self.system_word()?;
            if !matches!(kind.as_str(), "mjcf" | "urdf") {
                return Err(FrontError::source("S1600", "unknown robot asset kind", self.current().span));
            }
            let path = self.system_string()?;
            self.expect_word("sha256")?;
            let sha256 = self.system_string()?;
            self.expect_symbol(";")?;
            Some(RobotAssetDecl { kind, path, sha256 })
        } else {
            None
        };
        let (mut links, mut joints, mut actuators, mut sensors) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut tool = None;
        while !self.is_symbol("}") {
            if links.len() + joints.len() + actuators.len() + sensors.len() + usize::from(tool.is_some()) >= 256
                || matches!(self.current().kind, TokenKind::Eof)
            {
                return Err(FrontError::source("S1600", "robot clause limit or missing brace", self.current().span));
            }
            match self.system_word()?.as_str() {
                "link" => links.push(self.robot_link()?),
                "joint" => joints.push(self.robot_joint()?),
                "actuator" => actuators.push(self.robot_actuator()?),
                "sensor" => sensors.push(self.robot_sensor()?),
                "tool" => {
                    if tool.is_some() {
                        return Err(FrontError::source("S1600", "duplicate robot tool", self.current().span));
                    }
                    let link = self.system_word()?;
                    self.expect_word("pose")?;
                    let pose = self.robot_pose()?;
                    self.expect_symbol(";")?;
                    tool = Some((link, pose));
                }
                _ => return Err(FrontError::source("S1600", "unknown robot clause", self.current().span)),
            }
        }
        self.expect_symbol("}")?;
        Ok(RobotModelDecl { name, license, asset, links, joints, actuators, sensors, tool })
    }

    fn cell_model(&mut self) -> Result<CellModelDecl, FrontError> {
        let name = self.system_word()?;
        self.expect_symbol("{")?;
        self.expect_word("robot")?;
        let robot = self.system_path()?;
        self.expect_symbol(";")?;
        self.expect_word("mount")?;
        let mount = self.robot_pose()?;
        let mount_surface = if self.take_word("on") {
            let fixture = self.system_word()?;
            let mut exclude = Vec::new();
            if self.take_word("exclude") {
                self.expect_symbol("(")?;
                loop {
                    if exclude.len() >= 64 {
                        return Err(FrontError::source("S1600", "mount exclusion limit exceeded", self.current().span));
                    }
                    exclude.push(self.system_path()?);
                    if self.take_symbol(",").is_none() {
                        break;
                    }
                }
                self.expect_symbol(")")?;
            }
            Some((fixture, exclude))
        } else {
            None
        };
        self.expect_symbol(";")?;
        let (mut objects, mut io) = (Vec::new(), Vec::new());
        while !self.is_symbol("}") {
            if objects.len() + io.len() >= 256 || matches!(self.current().kind, TokenKind::Eof) {
                return Err(FrontError::source("S1600", "cell clause limit or missing brace", self.current().span));
            }
            let kind = self.system_word()?;
            match kind.as_str() {
                "fixture" | "object" => {
                    let object_name = self.system_word()?;
                    self.expect_symbol("{")?;
                    self.expect_word("pose")?;
                    let pose = self.robot_pose()?;
                    self.expect_symbol(";")?;
                    let mut geometries = Vec::new();
                    let mut dynamic = None;
                    let mut grasp = None;
                    while !self.is_symbol("}") {
                        if geometries.len() >= 256 || matches!(self.current().kind, TokenKind::Eof) {
                            return Err(FrontError::source("S1600", "cell geometry limit or missing brace", self.current().span));
                        }
                        let role = self.system_word()?;
                        if role == "grasp" {
                            if grasp.is_some() {
                                return Err(FrontError::source("S1600", "duplicate grasp clause", self.current().span));
                            }
                            self.expect_word("pose")?;
                            grasp = Some(self.robot_pose()?);
                            self.expect_symbol(";")?;
                            continue;
                        }
                        if role == "dynamic" {
                            if dynamic.is_some() {
                                return Err(FrontError::source("S1600", "duplicate dynamic clause", self.current().span));
                            }
                            self.expect_word("mass")?;
                            dynamic = Some(self.system_quantity()?);
                            self.expect_word("attachable")?;
                            self.expect_symbol(";")?;
                            continue;
                        }
                        if !matches!(role.as_str(), "visual" | "collision") {
                            return Err(FrontError::source("S1600", "unknown cell object clause", self.current().span));
                        }
                        geometries.push(self.robot_geometry(role)?);
                    }
                    self.expect_symbol("}")?;
                    objects.push(CellObjectDecl { kind, name: object_name, pose, geometries, dynamic, grasp });
                }
                "io" => {
                    let name = self.system_word()?;
                    self.expect_symbol(":")?;
                    let direction = self.system_hyphenated()?;
                    if !matches!(direction.as_str(), "digital-in" | "digital-out") {
                        return Err(FrontError::source("S1600", "unknown cell IO direction", self.current().span));
                    }
                    self.expect_symbol(";")?;
                    io.push((name, direction));
                }
                _ => return Err(FrontError::source("S1600", "unknown cell clause", self.current().span)),
            }
        }
        self.expect_symbol("}")?;
        Ok(CellModelDecl { name, robot, mount, mount_surface, objects, io })
    }

    fn system_item(&mut self, module: Option<&str>) -> Result<SystemItem, FrontError> {
        let start = self.current().span.start;
        let external = self.take_word("extern");
        let head = self.system_word()?;
        let kind = match head.as_str() {
            "component" => {
                let name = self.system_word()?;
                self.expect_symbol("{")?;
                let mut clauses = Vec::new();
                while !self.is_symbol("}") {
                    if clauses.len() >= 256 || matches!(self.current().kind, TokenKind::Eof) {
                        return Err(FrontError::source("S1000", "component clause limit or missing closing brace", self.current().span));
                    }
                    clauses.push(self.system_clause()?);
                }
                self.expect_symbol("}")?;
                SystemItemKind::Component { name, external, clauses }
            }
            "connect" if !external => {
                let source = self.system_port_path()?;
                self.expect_symbol("->")?;
                let target = self.system_port_path()?;
                self.expect_word("policy")?;
                let mut policy = self.system_hyphenated()?;
                if self.take_symbol("(").is_some() {
                    let token = self.bump();
                    let TokenKind::Integer(count) = token.kind else { return Err(FrontError::source("S1000", "policy count must be an integer", token.span)); };
                    policy.push('('); policy.push_str(&count); policy.push(')');
                    self.expect_symbol(")")?;
                }
                self.expect_symbol(";")?;
                SystemItemKind::Connect { source, target, policy }
            }
            "flow" if !external => {
                let name = self.system_word()?;
                self.expect_symbol(":")?;
                let mut ports = vec![self.system_port_path()?];
                while self.take_symbol("->").is_some() {
                    if ports.len() >= 256 { return Err(FrontError::source("S1000", "flow port limit exceeded", self.current().span)); }
                    ports.push(self.system_port_path()?);
                }
                self.expect_word("within")?;
                let within = self.system_quantity()?;
                self.expect_symbol(";")?;
                SystemItemKind::Flow { name, ports, within }
            }
            "product" if !external => {
                let name = self.system_word()?;
                self.expect_symbol("{")?;
                let mut clauses = Vec::new();
                while !self.is_symbol("}") {
                    if clauses.len() >= 256 || matches!(self.current().kind, TokenKind::Eof) { return Err(FrontError::source("S1000", "product clause limit or missing closing brace", self.current().span)); }
                    clauses.push(self.system_clause()?);
                }
                self.expect_symbol("}")?;
                SystemItemKind::Product { name, clauses }
            }
            "system" if !external => {
                let name = self.system_word()?;
                let mut products = Vec::new();
                if self.take_word("for") {
                    products.push(self.system_path()?);
                    while self.take_symbol(",").is_some() { products.push(self.system_path()?); }
                }
                self.expect_symbol("{")?;
                let mut instances = Vec::new();
                while !self.is_symbol("}") {
                    if instances.len() >= 256 || matches!(self.current().kind, TokenKind::Eof) { return Err(FrontError::source("S1000", "system instance limit or missing closing brace", self.current().span)); }
                    let instance = self.system_word()?;
                    self.expect_symbol(":")?;
                    let component = self.system_path()?;
                    self.expect_symbol(";")?;
                    instances.push((instance, component));
                }
                self.expect_symbol("}")?;
                SystemItemKind::System { name, products, instances }
            }
            "robot" if !external => SystemItemKind::RobotModel(self.robot_model()?),
            "cell" if !external => SystemItemKind::CellModel(self.cell_model()?),
            _ => return Err(FrontError::source("S1000", "invalid system declaration", self.current().span)),
        };
        Ok(SystemItem { module: module.map(str::to_owned), kind, span: Span { start, end: self.tokens[self.pos - 1].span.end } })
    }
}
