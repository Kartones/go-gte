package simd

import "unsafe"

const hasSgemmAsm = true

// sgemmNTAsm computes C += alpha * A * B^T, using AVX2+FMA, per the
// semantics documented on SgemmNT.
//
//go:noescape
func sgemmNTAsm(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)

// sgemmNNAsm computes C += alpha * A * B, using AVX2+FMA, per the semantics
// documented on SgemmNN.
//
//go:noescape
func sgemmNNAsm(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)

// SgemmNT computes C += alpha * A * B^T.
// A is [m,lda] row-major, B is [n,ldb] row-major, C is [m,ldc] row-major.
// Caller must handle beta (pre-scale C before calling).
//
// Uses AVX2+FMA assembly when FastPathEnabled, else a scalar Go fallback.
func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	if FastPathEnabled {
		sgemmNTAsm(m, n, k, alpha, a, b, c, lda, ldb, ldc)
		return
	}
	sgemmNTScalar(m, n, k, alpha, a, b, c, lda, ldb, ldc)
}

// SgemmNN computes C += alpha * A * B.
// A is [m,lda] row-major, B is [k,ldb] row-major, C is [m,ldc] row-major.
// Caller must handle beta.
//
// Uses AVX2+FMA assembly when FastPathEnabled, else a scalar Go fallback.
func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	if FastPathEnabled {
		sgemmNNAsm(m, n, k, alpha, a, b, c, lda, ldb, ldc)
		return
	}
	sgemmNNScalar(m, n, k, alpha, a, b, c, lda, ldb, ldc)
}
