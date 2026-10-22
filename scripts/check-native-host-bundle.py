#!/usr/bin/env python3
"""Check the standalone native host source bundle without building it.

The native host (src/native.rs) and the export host (src/native/export.rs) are single-file rustc compilations
assembled from include_str! sources, so a source in the bundle cannot declare a file module (`mod x;`) unless the
bundle inlines or strips it: otherwise every native link fails with E0583, which only a `--features native` build
shows (#377: `pub mod fold;` and `pub mod storage;` reached main and turned it red for 3 h). This checks, for every
bundled source:

  * each file module it declares (outside `#[cfg(test)]`) is inlined, or stripped through SEPARATE_MODULES
    (src/native_foreign_source.rs). Inlined means src/native.rs includes the module's file and holds the
    declaration's exact text as a string literal: native.rs swaps one for the other with `replacen` (a tuple list
    since #377, chained calls before), which silently does nothing when the text differs;
  * each `crate::NAME` path it uses names a module the host declares, a host alias or constant, or a top-level item
    of a bundled source (the second half of #377: fold.rs reached crate::fsum_reference).

Usage: check-native-host-bundle.py [--root DIR]      (exit 1 with one line per problem)
       check-native-host-bundle.py --root DIR --list  (print the bundled sources)
"""
import argparse
import re
import sys
from pathlib import Path, PurePosixPath

BUNDLERS = ('src/native.rs', 'src/native_foreign_source.rs', 'src/native/export.rs')
INCLUDE = re.compile(r'include_str!\(\s*"([^"]+\.rs)"\s*\)')
# A declaration string literal in src/native.rs ("mod name;", "pub mod name;"): the text replacen looks for.
LITERAL = re.compile(r'"((?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;)"')
DECLARATION = re.compile(r'^\s*((?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;)')
ITEM = re.compile(r'^(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+|const\s+|async\s+|extern\s+"C"\s+)*'
                  r'(?:fn|struct|enum|union|trait|type|const|static|mod|macro_rules!)\s+(\w+)')
USE_ROOT = re.compile(r'^(?:pub(?:\([^)]*\))?\s+)?use\s+(?:[\w:]+::)?(\w+)\s*(?:as\s+(\w+))?\s*;')
# Text the host format strings declare at the host's root: `mod name {{`, `use x as name;`, `const NAME:`.
HOST_MOD = re.compile(r'\bmod\s+(\w+)\s*\{\{')
HOST_ALIAS = re.compile(r'\buse\s+[\w:]+\s+as\s+(\w+)\s*;')
HOST_CONST = re.compile(r'\bconst\s+(\w+)\s*:')


def code(text):
    """The text without // comments (string literals here never hold //)."""
    return '\n'.join(line.split('//', 1)[0] for line in text.splitlines())


def resolve(including, target):
    parts = []
    for part in (PurePosixPath(including).parent / target).parts:
        if part == '..':
            parts.pop()
        elif part != '.':
            parts.append(part)
    return '/'.join(parts)


def module_file(parent, name):
    """Where `mod name;` in `parent` lives: beside a mod.rs/lib.rs, else in the parent's own directory."""
    path = PurePosixPath(parent)
    base = path.parent if path.name in ('mod.rs', 'lib.rs', 'main.rs') else path.with_suffix('')
    return [f'{base}/{name}.rs', f'{base}/{name}/mod.rs']


def separate_modules(text):
    match = re.search(r'SEPARATE_MODULES\s*:\s*\[&str;\s*\d+\]\s*=\s*\[([^\]]*)\]', text)
    return set(re.findall(r'"(\w+)"', match.group(1))) if match else set()


def check(root):
    root = Path(root)
    read = lambda path: (root / path).read_text()
    problems, bundled, literals, stripped = [], {}, {}, {}
    for bundler in BUNDLERS:
        text = read(bundler)
        for target in INCLUDE.findall(text):
            bundled.setdefault(resolve(bundler, target), bundler)
        if bundler == 'src/native.rs':
            for declaration, name in LITERAL.findall(text):
                literals.setdefault(name, set()).add(declaration)
        modules = separate_modules(text)
        if modules:
            for target in INCLUDE.findall(text):
                stripped[resolve(bundler, target)] = modules
    # The host format strings separate items with \n escapes; make them line breaks so `\b` sees each `mod`.
    host = read('src/native.rs').replace('\\n', '\n')
    allowed = set(HOST_MOD.findall(host)) | set(HOST_ALIAS.findall(host)) | set(HOST_CONST.findall(host))
    sources = {}
    for path in sorted(bundled):
        if not (root / path).is_file():
            problems.append(f'{bundled[path]}: includes {path}, which does not exist')
            continue
        sources[path] = read(path)
        for line in code(sources[path]).splitlines():
            if not line.startswith((' ', '\t')):
                allowed.update(ITEM.findall(line))
                for name, alias in USE_ROOT.findall(line):
                    allowed.add(alias or name)
    for path, text in sources.items():
        lines = text.splitlines()
        for number, line in enumerate(lines, 1):
            match = DECLARATION.match(line.split('//', 1)[0])
            if not match:
                continue
            declaration, name = match.group(1), match.group(2)
            attributes = []
            for previous in reversed(lines[:number - 1]):
                if not previous.strip().startswith('#['):
                    break
                attributes.append(previous.strip())
            if any(a.replace(' ', '') == '#[cfg(test)]' for a in attributes):
                continue
            if name in stripped.get(path, ()):
                continue
            homes = module_file(path, name)
            if not any(home in bundled for home in homes):
                problems.append(f'{path}:{number}: `{declaration}` is a file module, but the native host bundle does '
                                f'not inline it: add ("{declaration}", include_str!("...{homes[0]}")) to the inline '
                                f'list in src/native.rs (or strip it), see #377')
            elif declaration not in literals.get(name, ()):
                problems.append(f'{path}:{number}: src/native.rs inlines {homes[0]} as '
                                f'{" or ".join(f"`{d}`" for d in sorted(literals.get(name, ()))) or "no declaration"}, '
                                f'but the file declares `{declaration}`: replacen would not find it')
        for name in sorted(set(re.findall(r'\bcrate::(\w+)', code(text)))):
            if name not in allowed:
                problems.append(f'{path}: uses crate::{name}, which the native host does not declare: add '
                                f'`mod {name} {{{{ {{}} }}}}` with its source to the host in src/native.rs, see #377')
    for name, declarations in sorted(literals.items()):
        parents = [p for p in sources if any(home in sources for home in module_file(p, name))]
        for declaration in sorted(declarations):
            if parents and not any(declaration in sources[p] for p in parents):
                problems.append(f'src/native.rs: inlines `{declaration}`, which {parents[0]} no longer declares')
    return sorted(sources), problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    parser.add_argument('--root', default=Path(__file__).resolve().parents[1])
    parser.add_argument('--list', action='store_true', help='print the bundled sources')
    args = parser.parse_args()
    sources, problems = check(args.root)
    if args.list:
        print('\n'.join(sources))
        return 0
    for problem in problems:
        print(f'::error::{problem}')
    if problems:
        return 1
    print(f'native host bundle: {len(sources)} sources, every file module inlined or stripped, every crate:: path '
          f'declared')
    return 0


if __name__ == '__main__':
    sys.exit(main())
