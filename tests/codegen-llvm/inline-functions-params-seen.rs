// Checks that the LLVM backend's `CodegenUnitDebugContext::params_seen` hashmap is working. See
// the comment on its declaration for details.

// Note: -Copt-level=1 ensures the required `ReferencePropagation` pass runs.
//@ compile-flags: -Zinline-mir=yes -Copt-level=1 -Cdebuginfo=2 -Ccodegen-units=1 -Zverify-llvm-ir
//@ edition: 2021

#![crate_type = "lib"]

extern "Rust" {
    fn sink(x: &u8);
}

#[inline]
fn callee(x: &u8) {
    // Check the following:
    // - No `define` for `callee` (because it's inlined).
    // - Exactly one `x` variable entry (not three).
    //
    // CHECK-NOT: define{{.*}}callee
    // CHECK: !DILocalVariable(name: "x", arg: 1
    // CHECK-NOT: !DILocalVariable(name: "x", arg: 1
    unsafe { sink(x) }
}

pub fn f1(s: &u8) {
    callee(s)
}

pub fn f2(s: &mut u8) {
    callee(&*s)
}

pub fn f3(s: *const u8) {
    callee(unsafe { &*s })
}
