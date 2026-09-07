//go:build arm64

package simd

// Sdot computes the dot product of two float32 slices using NEON.
//
//go:noescape
func Sdot(x, y []float32) float32

// Saxpy computes y[i] += alpha * x[i] using NEON.
//
//go:noescape
func Saxpy(alpha float32, x []float32, y []float32)
