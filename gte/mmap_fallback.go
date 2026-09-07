//go:build !unix

package gte

// LoadMmap loads a .gtemodel model. On platforms without a unix-style mmap
// syscall (e.g. windows), it falls back to a plain buffered read via Load.
func LoadMmap(modelPath string) (*Model, error) {
	return Load(modelPath)
}

// munmapData is a no-op on platforms without a unix-style mmap syscall.
// It exists only so common code (gte/model.go) can call it unconditionally;
// LoadMmap on this platform never sets mmapData, so Close() never calls this.
func munmapData(data []byte) error {
	return nil
}
