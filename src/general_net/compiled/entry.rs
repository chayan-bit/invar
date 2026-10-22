//! ABI13 entry packet: parameter words, allocation/live counters, then the
//! canonical owned heap image. The root copies it once, before any source step.
use super::{Error, Program, Status, emit::Writer, fail};
use crate::front::{Mode, Type};

#[derive(Clone, Debug)]
pub struct EntryLayout {
    modes: Vec<Option<Mode>>,
    input_words: usize,
}
impl EntryLayout {
    pub(super) fn new(p: &Program<'_>) -> Result<Self, Error> {
        let params = &p.mir.functions[p.entry].params;
        if params.len() > crate::eval::host_read::MAX_PARAMETERS
            || params.iter().any(|p| {
                p.mode == Some(Mode::Take)
                    || matches!(p.ty, Type::Named(_)) && p.mode != Some(Mode::Read)
            })
        {
            return Err(Error::UnsupportedType);
        }
        Ok(Self {
            modes: params.iter().map(|p| p.mode).collect(),
            input_words: params
                .len()
                .checked_add(2)
                .and_then(|n| n.checked_add(p.owned.as_ref()?.heap_words()))
                .ok_or(Error::Limit)?,
        })
    }
    pub fn modes(&self) -> &[Option<Mode>] {
        &self.modes
    }
    pub fn input_words(&self) -> usize {
        self.input_words
    }
}
pub(super) fn initialize(w: &mut Writer, p: &Program<'_>) -> Result<(), Error> {
    if p.entry_layout.is_none() {
        return Ok(());
    }
    let params = p.mir.functions[p.entry].params.len();
    let owned = p.owned.as_ref().ok_or(Error::State)?;
    w.push(&format!("if(input[{params}]<0 || (gnet_u)input[{params}]>{} || input[{}]<0 || (gnet_u)input[{}]>{}) {{ {} }} allocated=(gnet_u)input[{params}];live=(gnet_u)input[{}]; for(gnet_u j=0;j<{};++j)heap[j]=input[{}+j];", owned.allocations(),params+1,params+1,owned.elements(),fail(Status::InvalidInput,None),params+1,owned.heap_words(),params+2))
}

pub(super) fn owner(ty: &Type) -> bool {
    matches!(
        ty,
        Type::ArrayI32 | Type::ArrayU32 | Type::ArrayF32 | Type::Named(_)
    )
}
