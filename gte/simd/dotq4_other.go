//go:build !amd64 && !arm64

package simd

import "unsafe"

// DotQ4 scalar fallback (no SIMD assembly for this architecture).
func DotQ4(x unsafe.Pointer, blocks unsafe.Pointer, nBlocks int) float32 {
	return dotQ4Scalar(x, blocks, nBlocks)
}

// LinearQ4 scalar fallback (no SIMD assembly for this architecture).
func LinearQ4(y, x, w, bias unsafe.Pointer, seqLen, inDim, outDim int) {
	linearQ4Scalar(y, x, w, bias, seqLen, inDim, outDim)
}
