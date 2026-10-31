//go:build unix

package simd

import (
	"fmt"
	"math/rand"
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// guardedMatrix returns a rows*cols matrix whose last element is the last
// byte before a PROT_NONE page, so any read past the end faults.
func guardedMatrix(t *testing.T, rng *rand.Rand, rows, cols int) []float32 {
	t.Helper()
	page := os.Getpagesize()
	size := rows * cols * 4
	if size > page {
		t.Fatalf("matrix of %d bytes does not fit in one page", size)
	}
	mem, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Skipf("mmap unavailable: %v", err)
	}
	t.Cleanup(func() { _ = syscall.Munmap(mem) })
	if err := syscall.Mprotect(mem[page:], syscall.PROT_NONE); err != nil {
		t.Skipf("mprotect unavailable: %v", err)
	}
	m := unsafe.Slice((*float32)(unsafe.Pointer(&mem[page-size])), rows*cols)
	for i := range m {
		m[i] = rng.Float32()*2 - 1
	}
	return m
}

func TestSgemmNTGebpShortRowBlocksDoNotReadPastA(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly")
	}
	const n, k = gebpNR, 64
	for m := 1; m <= 2*gebpMR+1; m++ {
		t.Run(fmt.Sprintf("m=%d", m), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(m)))
			a := guardedMatrix(t, rng, m, k)
			b := randMatrix(rng, n, k)
			got := make([]float32, m*n)
			ref := make([]float32, m*n)

			SgemmNTGebp(m, n, k, 0.5, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]), k, k, n)
			naiveSgemmNT(m, n, k, 0.5, a, k, b, k, ref, n)

			for i := range got {
				if relErr(got[i], ref[i]) > 1e-4 {
					t.Fatalf("c[%d] = %v, want %v", i, got[i], ref[i])
				}
			}
		})
	}
}
