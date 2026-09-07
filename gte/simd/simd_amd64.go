//go:build amd64

package simd

// sdotAsm computes the dot product of two float32 slices using AVX2+FMA.
//
//go:noescape
func sdotAsm(x, y []float32) float32

// saxpyAsm computes y[i] += alpha * x[i] using AVX2+FMA.
//
//go:noescape
func saxpyAsm(alpha float32, x []float32, y []float32)

// Sdot computes the dot product of two float32 slices, using AVX2+FMA
// assembly when FastPathEnabled, else a scalar Go fallback.
func Sdot(x, y []float32) float32 {
	if FastPathEnabled {
		return sdotAsm(x, y)
	}
	return sdotScalar(x, y)
}

// Saxpy computes y[i] += alpha * x[i], using AVX2+FMA assembly when
// FastPathEnabled, else a scalar Go fallback.
func Saxpy(alpha float32, x []float32, y []float32) {
	if FastPathEnabled {
		saxpyAsm(alpha, x, y)
		return
	}
	saxpyScalar(alpha, x, y)
}
