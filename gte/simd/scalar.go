package simd

import (
	"math"
	"unsafe"
)

// This file holds the pure Go scalar implementations shared by:
//   - the amd64 dispatchers, when FastPathEnabled is false (no AVX2/FMA), and
//   - architectures with no SIMD assembly kernel at all.
// Keeping a single copy avoids duplicating the math in multiple build-tagged
// files.

// sdotScalar computes the dot product of x and y without SIMD.
func sdotScalar(x, y []float32) float32 {
	sum := float32(0)
	i := 0
	for ; i+8 <= len(x); i += 8 {
		sum += x[i]*y[i] + x[i+1]*y[i+1] + x[i+2]*y[i+2] + x[i+3]*y[i+3] +
			x[i+4]*y[i+4] + x[i+5]*y[i+5] + x[i+6]*y[i+6] + x[i+7]*y[i+7]
	}
	for ; i < len(x); i++ {
		sum += x[i] * y[i]
	}
	return sum
}

// saxpyScalar computes y[i] += alpha * x[i] without SIMD.
func saxpyScalar(alpha float32, x []float32, y []float32) {
	for i := range x {
		y[i] += alpha * x[i]
	}
}

// dotQ4Scalar computes a Q4 dot product without SIMD.
func dotQ4Scalar(x unsafe.Pointer, blocks unsafe.Pointer, nBlocks int) float32 {
	xp := (*[1 << 30]float32)(x)
	bp := (*[1 << 30]byte)(blocks)
	sum := float32(0)

	for b := 0; b < nBlocks; b++ {
		bOff := b * 20
		// Read scale (little-endian)
		bits := uint32(bp[bOff]) | uint32(bp[bOff+1])<<8 | uint32(bp[bOff+2])<<16 | uint32(bp[bOff+3])<<24
		scale := *(*float32)(unsafe.Pointer(&bits))
		xOff := b * 32

		for i := 0; i < 16; i++ {
			q := float32(int(bp[bOff+4+i]&0x0F) - 8)
			sum += xp[xOff+i] * scale * q
		}
		for i := 0; i < 16; i++ {
			q := float32(int(bp[bOff+4+i]>>4) - 8)
			sum += xp[xOff+16+i] * scale * q
		}
	}
	return sum
}

// linearQ4Scalar computes Y = X·W_q4^T + bias without SIMD, reusing dotQ4Scalar.
func linearQ4Scalar(y, x, w, bias unsafe.Pointer, seqLen, inDim, outDim int) {
	yp := (*[1 << 30]float32)(y)
	xp := (*[1 << 30]float32)(x)
	bp := (*[1 << 30]byte)(w)
	var biasP *[1 << 30]float32
	if bias != nil {
		biasP = (*[1 << 30]float32)(bias)
	}

	nBlocks := inDim / 32
	rowBytes := nBlocks * 20

	for s := 0; s < seqLen; s++ {
		for o := 0; o < outDim; o++ {
			xPtr := unsafe.Pointer(&xp[s*inDim])
			wPtr := unsafe.Pointer(&bp[o*rowBytes])
			dot := dotQ4Scalar(xPtr, wPtr, nBlocks)
			if biasP != nil {
				dot += biasP[o]
			}
			yp[s*outDim+o] = dot
		}
	}
}

// dotQ4IntScalar computes a Q4 dot product using dynamic int8 quantization
// and integer MAC emulation, without SIMD.
func dotQ4IntScalar(x unsafe.Pointer, blocks unsafe.Pointer, nBlocks int) float32 {
	xp := (*[1 << 30]float32)(x)
	bp := (*[1 << 30]byte)(blocks)
	total := float32(0)

	for b := 0; b < nBlocks; b++ {
		bOff := b * 20
		xOff := b * 32

		// Read w_scale
		bits := uint32(bp[bOff]) | uint32(bp[bOff+1])<<8 | uint32(bp[bOff+2])<<16 | uint32(bp[bOff+3])<<24
		wScale := *(*float32)(unsafe.Pointer(&bits))

		// Find absmax of x block
		absmax := float32(0)
		for i := 0; i < 32; i++ {
			v := xp[xOff+i]
			if v < 0 {
				v = -v
			}
			if v > absmax {
				absmax = v
			}
		}
		if absmax == 0 {
			continue
		}

		// Quantize x to int8
		invScale := float32(127.0) / absmax
		xI8 := [32]int8{}
		for i := 0; i < 32; i++ {
			q := int(math.Round(float64(xp[xOff+i] * invScale)))
			if q > 127 {
				q = 127
			}
			if q < -127 {
				q = -127
			}
			xI8[i] = int8(q)
		}

		// Integer dot with nibbles (unsigned, no -8 subtraction)
		// Then correct by subtracting 8*sum(x_i8)
		dot := int32(0)
		sumX := int32(0)
		for i := 0; i < 16; i++ {
			nibLo := int32(bp[bOff+4+i] & 0x0F)
			nibHi := int32(bp[bOff+4+i] >> 4)
			dot += int32(xI8[i]) * nibLo
			dot += int32(xI8[16+i]) * nibHi
			sumX += int32(xI8[i]) + int32(xI8[16+i])
		}
		dot -= 8 * sumX // correction for nibble-8 centering

		// Descale
		total += (wScale * absmax / 127.0) * float32(dot)
	}
	return total
}

// sgemmNTScalar computes C += alpha * A * B^T without SIMD.
// A is [m,lda] row-major, B is [n,ldb] row-major.
func sgemmNTScalar(m, n, k int, alpha float32, aPtr, bPtr, cPtr unsafe.Pointer, lda, ldb, ldc int) {
	a := unsafe.Slice((*float32)(aPtr), m*lda)
	b := unsafe.Slice((*float32)(bPtr), n*ldb)
	c := unsafe.Slice((*float32)(cPtr), m*ldc)

	for i := 0; i < m; i++ {
		aRow := a[i*lda : i*lda+k]
		for j := 0; j < n; j++ {
			bRow := b[j*ldb : j*ldb+k]
			c[i*ldc+j] += alpha * sdotScalar(aRow, bRow)
		}
	}
}

// sgemmNNScalar computes C += alpha * A * B without SIMD.
// A is [m,lda] row-major, B is [k,ldb] row-major.
func sgemmNNScalar(m, n, k int, alpha float32, aPtr, bPtr, cPtr unsafe.Pointer, lda, ldb, ldc int) {
	a := unsafe.Slice((*float32)(aPtr), m*lda)
	b := unsafe.Slice((*float32)(bPtr), k*ldb)
	c := unsafe.Slice((*float32)(cPtr), m*ldc)

	for i := 0; i < m; i++ {
		cRow := c[i*ldc : i*ldc+n]
		for p := 0; p < k; p++ {
			av := alpha * a[i*lda+p]
			if av == 0 {
				continue
			}
			bRow := b[p*ldb : p*ldb+n]
			saxpyScalar(av, bRow, cRow)
		}
	}
}
