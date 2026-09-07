//go:build !amd64 && !arm64

package simd

// FastPathEnabled is always false on architectures without a SIMD kernel
// in this package; only the scalar Go fallbacks are available.
var FastPathEnabled = false
