//go:build arm64

package simd

import "unsafe"

// DotQ4 computes the dot product of a float32 vector x[0:nBlocks*32] with a
// Q4_0-quantized vector, using NEON. See the block format documented in
// dotq4_amd64.go's DotQ4.
//
//go:noescape
func DotQ4(x unsafe.Pointer, blocks unsafe.Pointer, nBlocks int) float32

// LinearQ4 computes Y[seqLen, outDim] = X[seqLen, inDim] · W_q4^T + bias,
// using NEON.
//
//go:noescape
func LinearQ4(y, x, w, bias unsafe.Pointer, seqLen, inDim, outDim int)
