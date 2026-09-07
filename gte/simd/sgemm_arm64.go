package simd

import "unsafe"

const hasSgemmAsm = true

// SgemmNT computes C += alpha * A * B^T, using NEON.
// A is [m,lda] row-major, B is [n,ldb] row-major, C is [m,ldc] row-major.
// Caller must handle beta (pre-scale C before calling).
//
//go:noescape
func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)

// SgemmNN computes C += alpha * A * B, using NEON.
// A is [m,lda] row-major, B is [k,ldb] row-major, C is [m,ldc] row-major.
// Caller must handle beta.
//
//go:noescape
func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int)
