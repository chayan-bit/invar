"""lldb helpers for Invar native debug builds (#305).

    invar build PROGRAM.inv --target native --output app --debug
    lldb -o 'command script import share/invar/invar_lldb.py' app

Stops at every trap and names its category and source position, adds the
`invar-where` command for the current trap, and prints arrays, records,
variants and views (host handles) by calling the host's render hook in the
live process. Without a live process a handle prints as its number.
"""

import lldb

CATEGORIES = {
    1: "INTEGER_OVERFLOW",
    2: "DIVISION_BY_ZERO",
    3: "INDEX_OUT_OF_BOUNDS",
    4: "UNIT_CONVERSION_INEXACT",
}
# The member of every Invar handle structure in the debug info.
HANDLE = "invar_handle"
TRAP = "invar_debug_trap_v1"
RENDER_LIMIT = 4096
RENDER = "((const char *(*)(unsigned long long, unsigned int))invar_debug_render_v1)({}, {})"


def position(path, offset):
    """1-based line and character column of a byte offset in a source file."""
    with open(path, "rb") as source:
        text = source.read()
    line_start = text.rfind(b"\n", 0, offset) + 1
    column = len(text[line_start:offset].decode("utf-8", "replace")) + 1
    return text.count(b"\n", 0, offset) + 1, column


def source_path(thread, index):
    """The Invar source of the nearest program frame at or above frame `index`."""
    for frame in list(thread)[index:]:
        spec = frame.GetLineEntry().GetFileSpec()
        if spec.IsValid() and not spec.GetFilename().endswith(".rs"):
            return spec.fullpath
    return None


def describe(thread, index):
    frame = thread.GetFrameAtIndex(index)
    category, start, end = (
        frame.FindVariable(name).GetValueAsUnsigned() for name in ("category", "start", "end")
    )
    name = CATEGORIES.get(category, f"category {category}")
    path = source_path(thread, index + 1)
    if path is None:
        return f"{name} at bytes {start}..{end}"
    try:
        (line, column), (end_line, end_column) = position(path, start), position(path, end)
    except OSError:
        return f"{name} at {path} bytes {start}..{end}"
    return f"{name} at {path}:{line}:{column}-{end_line}:{end_column}"


def is_trap(name):
    """The hook's frame, named `host::invar_debug_trap_v1` with host debug info."""
    return name is not None and (name == TRAP or name.endswith("::" + TRAP))


def trap_index(thread):
    for index, frame in enumerate(thread):
        if is_trap(frame.GetFunctionName()):
            return index
    return None


def on_trap(frame, location, internal):
    # The stop may be in code inlined into the hook (`black_box`).
    thread = frame.GetThread()
    index = trap_index(thread)
    if index is not None:
        print(f"invar trap: {describe(thread, index)}")
    return True


def where(debugger, command, context, result, internal):
    """Name the current trap's category and source position."""
    thread = context.GetThread()
    index = trap_index(thread) if thread.IsValid() else None
    if index is None:
        result.SetError("invar-where: not stopped at a trap")
        return
    result.AppendMessage(describe(thread, index))


def is_handle(sbtype, internal):
    """Only the handle structures the Invar compile unit emits."""
    sbtype = sbtype.GetCanonicalType()
    return (
        sbtype.GetTypeClass() == lldb.eTypeClassStruct
        and sbtype.GetByteSize() == 8
        and sbtype.GetNumberOfFields() == 1
        and sbtype.GetFieldAtIndex(0).GetName() == HANDLE
    )


def call_text(frame, expression):
    """The NUL-terminated text a host hook returns, or None."""
    address = frame.EvaluateExpression(expression).GetValueAsUnsigned()
    if not address:
        return None
    error = lldb.SBError()
    text = frame.GetThread().GetProcess().ReadCStringFromMemory(address, RENDER_LIMIT, error)
    return text if error.Success() else None


def name_hash(name):
    """FNV-1a (32-bit) of a type name, as the host's `name_hash`."""
    value = 0x811C9DC5
    for byte in name.encode():
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def summary(value, internal):
    name = value.GetType().GetCanonicalType().GetName()
    handle = value.GetChildMemberWithName(HANDLE).GetValueAsUnsigned()
    process = value.GetProcess()
    frame = value.GetFrame()
    if process.IsValid() and process.GetState() == lldb.eStateStopped and frame.IsValid():
        text = call_text(frame, RENDER.format(handle, name_hash(name)))
        if text is not None:
            # Typed text starts with the name; stored words do not.
            return text if text.startswith(name) else f"{name} {text}"
    return f"{name} handle {handle}"


def __lldb_init_module(debugger, internal):
    debugger.HandleCommand(
        f"type summary add --python-function {__name__}.summary --recognizer-function {__name__}.is_handle"
    )
    debugger.HandleCommand(f"command script add --function {__name__}.where invar-where")
    target = debugger.GetSelectedTarget()
    if target.IsValid():
        breakpoint = target.BreakpointCreateByName(TRAP)
        breakpoint.SetScriptCallbackFunction(f"{__name__}.on_trap")
