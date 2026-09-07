//go:build arm64

package simd

// FastPathEnabled is always true on arm64: the NEON kernels in this package
// have no optional-instruction-set requirement, unlike amd64's AVX2/FMA path.
var FastPathEnabled = true
