//go:build amd64

package simd

import "unsafe"

// dotQ4Asm computes the dot product of a float32 vector x[0:nBlocks*32] with
// a Q4_0-quantized vector using AVX2, per the block format documented on DotQ4.
//
//go:noescape
func dotQ4Asm(x unsafe.Pointer, blocks unsafe.Pointer, nBlocks int) float32

// linearQ4Asm computes Y[seqLen, outDim] = X[seqLen, inDim] · W_q4^T + bias
// using AVX2, per the semantics documented on LinearQ4.
//
//go:noescape
func linearQ4Asm(y, x, w, bias unsafe.Pointer, seqLen, inDim, outDim int)

// DotQ4 computes the dot product of a float32 vector x[0:nBlocks*32] with a
// Q4_0-quantized vector stored as nBlocks blocks.
//
// Block format (20 bytes per 32 elements):
//   - bytes[0:4]:  float32 scale (little-endian)
//   - bytes[4:20]: 16 packed uint8, each containing 2 nibbles
//     low nibble (qs[i]&0xF) → elements 0..15
//     high nibble (qs[i]>>4) → elements 16..31
//   - Dequantized value: scale * (nibble - 8)
//
// Uses AVX2 assembly when FastPathEnabled, else a scalar Go fallback.
func DotQ4(x unsafe.Pointer, blocks unsafe.Pointer, nBlocks int) float32 {
	if FastPathEnabled {
		return dotQ4Asm(x, blocks, nBlocks)
	}
	return dotQ4Scalar(x, blocks, nBlocks)
}

// LinearQ4 computes Y[seqLen, outDim] = X[seqLen, inDim] · W_q4^T + bias.
// W is stored as outDim rows of Q4 blocks (inDim/32 blocks per row, 20 bytes each).
// bias may be nil (pass zero pointer and 0 for hasBias).
//
// Uses AVX2 assembly when FastPathEnabled, else a scalar Go fallback.
func LinearQ4(y, x, w, bias unsafe.Pointer, seqLen, inDim, outDim int) {
	if FastPathEnabled {
		linearQ4Asm(y, x, w, bias, seqLen, inDim, outDim)
		return
	}
	linearQ4Scalar(y, x, w, bias, seqLen, inDim, outDim)
}
