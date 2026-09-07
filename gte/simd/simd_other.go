//go:build !amd64 && !arm64

package simd

import "unsafe"

const hasSgemmAsm = false

// Sdot computes the dot product of two float32 slices (scalar fallback,
// no SIMD assembly for this architecture).
func Sdot(x, y []float32) float32 {
	return sdotScalar(x, y)
}

// Saxpy computes y[i] += alpha * x[i] (scalar fallback, no SIMD assembly
// for this architecture).
func Saxpy(alpha float32, x []float32, y []float32) {
	saxpyScalar(alpha, x, y)
}

// SgemmNT computes C += alpha * A * B^T (scalar fallback, no SIMD assembly
// for this architecture).
func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	sgemmNTScalar(m, n, k, alpha, a, b, c, lda, ldb, ldc)
}

// SgemmNN computes C += alpha * A * B (scalar fallback, no SIMD assembly for
// this architecture).
func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	sgemmNNScalar(m, n, k, alpha, a, b, c, lda, ldb, ldc)
}
