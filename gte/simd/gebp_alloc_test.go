package simd

import (
	"math/rand"
	"testing"
	"unsafe"
)

func TestSgemmNTGebpShortRowBlockDoesNotAllocate(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly")
	}
	const m, n, k = 7, gebpNR, 64
	rng := rand.New(rand.NewSource(1))
	a := randMatrix(rng, m, k)
	b := randMatrix(rng, n, k)
	c := make([]float32, m*n)
	run := func() {
		SgemmNTGebp(m, n, k, 0.5, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), k, k, n)
	}
	run()

	if allocs := testing.AllocsPerRun(100, run); allocs != 0 {
		t.Fatalf("SgemmNTGebp allocated %v times per call, want 0", allocs)
	}
}

func TestSgemmNTGebpPaddedRowsHonourLda(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("no SGEMM assembly")
	}
	const m, n, k, lda = 7, gebpNR, 64, 64 + 3
	rng := rand.New(rand.NewSource(2))
	a := randMatrix(rng, m, lda)
	b := randMatrix(rng, n, k)
	got := make([]float32, m*n)
	ref := make([]float32, m*n)

	SgemmNTGebp(m, n, k, 0.5, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]), lda, k, n)
	naiveSgemmNT(m, n, k, 0.5, a, lda, b, k, ref, n)

	for i := range got {
		if absErr(got[i], ref[i]) > 1e-3 {
			t.Fatalf("c[%d] = %v, want %v", i, got[i], ref[i])
		}
	}
}
