//go:build !amd64

package simd

import "unsafe"

// DotQ4Int scalar fallback — dynamic quantization + integer MAC emulation
// (no SIMD assembly for this architecture).
func DotQ4Int(x unsafe.Pointer, blocks unsafe.Pointer, nBlocks int) float32 {
	return dotQ4IntScalar(x, blocks, nBlocks)
}
