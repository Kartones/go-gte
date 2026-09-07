//go:build amd64

package gte

import (
	"math"
	"os"
	"os/exec"
	"strings"
	"testing"
	"unsafe"

	"github.com/kartones/go-gte/gte/simd"
)

// fallbackSubprocessEnv marks the re-exec'd child invocation of
// TestForcedFallback_NoAVX2FMA so it runs the real assertions instead of
// spawning another subprocess.
const fallbackSubprocessEnv = "GTE_FALLBACK_SUBPROCESS_CHECK"

// TestForcedFallback_NoAVX2FMA proves that with AVX2/FMA reported
// unavailable at runtime (GODEBUG=cpu.avx2=off,cpu.fma=off), embedding-
// relevant computations (Sdot, Saxpy, Q4 dot/linear, SGEMM) still produce
// correct results through the scalar/Gonum fallback path, with no
// illegal-instruction crash.
//
// golang.org/x/sys/cpu reads GODEBUG at package init, so the override must
// take effect before the process starts — this test re-execs itself as a
// subprocess with the env var set, rather than mutating cpu.X86 in-process.
func TestForcedFallback_NoAVX2FMA(t *testing.T) {
	if os.Getenv(fallbackSubprocessEnv) == "1" {
		runForcedFallbackChecks(t)
		return
	}

	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}

	cmd := exec.Command(exe, "-test.run=^TestForcedFallback_NoAVX2FMA$", "-test.v")
	cmd.Env = append(os.Environ(),
		"GODEBUG=cpu.avx2=off,cpu.fma=off",
		fallbackSubprocessEnv+"=1",
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("subprocess with AVX2/FMA forced off failed: %v\noutput:\n%s", err, out)
	}
	if !strings.Contains(string(out), "PASS") {
		t.Fatalf("subprocess did not report PASS; output:\n%s", out)
	}
	t.Logf("forced-fallback subprocess output:\n%s", out)
}

// runForcedFallbackChecks runs in the re-exec'd child process, with
// GODEBUG=cpu.avx2=off,cpu.fma=off already in effect.
func runForcedFallbackChecks(t *testing.T) {
	if simd.FastPathEnabled {
		t.Fatal("simd.FastPathEnabled = true; want false with GODEBUG=cpu.avx2=off,cpu.fma=off")
	}

	n := 384 // GTE hidden size
	x := make([]float32, n)
	y := make([]float32, n)
	for i := range x {
		x[i] = float32(i%17-8) * 0.01
		y[i] = float32(i%13-6) * 0.02
	}

	// Sdot: compare against a naive Go reference, same tolerance as
	// gte/simd/simd_test.go's TestSdotSmall (relErr <= 1e-5... use 1e-4,
	// matching the more permissive dot-product tests for larger n).
	naiveDot := float32(0)
	for i := range x {
		naiveDot += x[i] * y[i]
	}
	gotDot := simd.Sdot(x, y)
	if relErrF(gotDot, naiveDot) > 1e-4 {
		t.Errorf("Sdot = %v, want %v (relErr too high)", gotDot, naiveDot)
	}

	// Saxpy: y2[i] += alpha*x[i], compare against naive loop.
	alpha := float32(0.375)
	y2 := make([]float32, n)
	copy(y2, y)
	yRef := make([]float32, n)
	copy(yRef, y)
	simd.Saxpy(alpha, x, y2)
	for i := range yRef {
		yRef[i] += alpha * x[i]
	}
	for i := range y2 {
		if relErrF(y2[i], yRef[i]) > 1e-4 {
			t.Fatalf("Saxpy[%d] = %v, want %v", i, y2[i], yRef[i])
		}
	}

	// DotQ4: quantize y to Q4, compare simd.DotQ4 against FP32 reference,
	// same 15% relative-error tolerance as gte/quant_test.go's
	// TestDotQ4Accuracy (Q4 quantization is inherently lossy).
	bQ4 := quantizeQ4(y)
	nBlocks := n / QK4_0
	q4Dot := simd.DotQ4(unsafe.Pointer(&x[0]), unsafe.Pointer(&bQ4[0]), nBlocks)
	if relErrF(q4Dot, naiveDot) > 0.15 {
		t.Errorf("DotQ4 = %v, want ~%v (relErr too high)", q4Dot, naiveDot)
	}

	// LinearQ4: same tolerance pattern as gte/quant_test.go's TestLinearQ4.
	inDim, outDim, seqLen := 64, 32, 2
	xr := make([]float32, seqLen*inDim)
	for i := range xr {
		xr[i] = float32(i%19-9) * 0.01
	}
	w := make([]float32, outDim*inDim)
	for i := range w {
		w[i] = float32(i%23-11) * 0.005
	}
	bias := make([]float32, outDim)
	for i := range bias {
		bias[i] = float32(i) * 0.01
	}
	yRefLin := make([]float32, seqLen*outDim)
	for s := 0; s < seqLen; s++ {
		for o := 0; o < outDim; o++ {
			sum := float32(0)
			for k := 0; k < inDim; k++ {
				sum += xr[s*inDim+k] * w[o*inDim+k]
			}
			yRefLin[s*outDim+o] = sum + bias[o]
		}
	}
	wQ4 := make([]byte, outDim*(inDim/QK4_0)*BlockQ4Size)
	for o := 0; o < outDim; o++ {
		row := quantizeQ4(w[o*inDim : (o+1)*inDim])
		copy(wQ4[o*len(row):], row)
	}
	yGotLin := make([]float32, seqLen*outDim)
	simd.LinearQ4(
		unsafe.Pointer(&yGotLin[0]), unsafe.Pointer(&xr[0]), unsafe.Pointer(&wQ4[0]), unsafe.Pointer(&bias[0]),
		seqLen, inDim, outDim,
	)
	// gte/quant_test.go's TestLinearQ4 uses 25% for small test values.
	for i := range yGotLin {
		if yRefLin[i] == 0 {
			continue
		}
		if relErrF(yGotLin[i], yRefLin[i]) > 0.25 {
			t.Errorf("LinearQ4[%d] = %v, want ~%v (relErr too high)", i, yGotLin[i], yRefLin[i])
		}
	}

	// DotQ4Int: same block/tolerance pattern as gte/quant_test.go's
	// TestDotQ4IntAccuracy (20% — generous for double quantization).
	q4IntDot := simd.DotQ4Int(unsafe.Pointer(&x[0]), unsafe.Pointer(&bQ4[0]), nBlocks)
	if relErrF(q4IntDot, naiveDot) > 0.20 {
		t.Errorf("DotQ4Int = %v, want ~%v (relErr too high)", q4IntDot, naiveDot)
	}

	// sgemm(): exercises the top-level dispatch restructured in gte/sgemm.go
	// (amd64 fast-path branch skipped, falls through to gonum BLAS), both
	// NT (attention-style) and NN (FFN-style) shapes.
	m, kk, nn := 3, 8, 5
	a := make([]float32, m*kk)
	bMat := make([]float32, nn*kk)
	for i := range a {
		a[i] = float32(i%7-3) * 0.1
	}
	for i := range bMat {
		bMat[i] = float32(i%5-2) * 0.2
	}
	gotNT := make([]float32, m*nn)
	refNT := make([]float32, m*nn)
	sgemm(false, true, m, nn, kk, 1.0, a, kk, bMat, kk, 0, gotNT, nn)
	for i := 0; i < m; i++ {
		for j := 0; j < nn; j++ {
			sum := float32(0)
			for p := 0; p < kk; p++ {
				sum += a[i*kk+p] * bMat[j*kk+p]
			}
			refNT[i*nn+j] = sum
		}
	}
	for i := range gotNT {
		if relErrF(gotNT[i], refNT[i]) > 1e-3 {
			t.Errorf("sgemm NT[%d] = %v, want %v", i, gotNT[i], refNT[i])
		}
	}

	bNN := make([]float32, kk*nn)
	for i := range bNN {
		bNN[i] = float32(i%5-2) * 0.2
	}
	gotNN := make([]float32, m*nn)
	refNN := make([]float32, m*nn)
	sgemm(false, false, m, nn, kk, 1.0, a, kk, bNN, nn, 0, gotNN, nn)
	for i := 0; i < m; i++ {
		for j := 0; j < nn; j++ {
			sum := float32(0)
			for p := 0; p < kk; p++ {
				sum += a[i*kk+p] * bNN[p*nn+j]
			}
			refNN[i*nn+j] = sum
		}
	}
	for i := range gotNN {
		if relErrF(gotNN[i], refNN[i]) > 1e-3 {
			t.Errorf("sgemm NN[%d] = %v, want %v", i, gotNN[i], refNN[i])
		}
	}

	if math.IsNaN(float64(gotDot)) || math.IsInf(float64(gotDot), 0) {
		t.Errorf("Sdot produced non-finite result: %v", gotDot)
	}
}
