//go:build amd64

package simd

import "golang.org/x/sys/cpu"

// FastPathEnabled reports whether the AVX2+FMA assembly fast path may be
// used on this amd64 host. It is computed once at package init from the
// CPU features golang.org/x/sys/cpu detects (which honors GODEBUG's
// cpu.avx2/cpu.fma overrides), so it can be safely read at runtime instead
// of assumed from GOARCH alone.
var FastPathEnabled = cpu.X86.HasAVX2 && cpu.X86.HasFMA
