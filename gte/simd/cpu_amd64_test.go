//go:build amd64

package simd

import (
	"testing"

	"golang.org/x/sys/cpu"
)

// TestFastPathEnabledMatchesCPU is a best-effort, informational check that
// FastPathEnabled agrees with the raw AVX2+FMA detection from x/sys/cpu on
// the host running the test (no GODEBUG override in play here). The forced
// AVX2/FMA-off scenario is covered separately by the subprocess test in
// gte/fallback_amd64_test.go, since GODEBUG can't be mutated in-process.
func TestFastPathEnabledMatchesCPU(t *testing.T) {
	want := cpu.X86.HasAVX2 && cpu.X86.HasFMA
	if FastPathEnabled != want {
		t.Errorf("FastPathEnabled = %v, want %v (HasAVX2=%v HasFMA=%v)",
			FastPathEnabled, want, cpu.X86.HasAVX2, cpu.X86.HasFMA)
	}
}
