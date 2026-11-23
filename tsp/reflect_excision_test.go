// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test enforces the zero-reflection policy in extracted packages.
// It checks imports, not arbitrary substrings in documentation or comments.
package tsp_test

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
)

// TestReflect_Excision rejects reflect imports in tsp and match Go source files.
// The policy includes test files to prevent future reflection-heavy assertions.
func TestReflect_Excision(t *testing.T) {
	_, caller, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Dir(filepath.Dir(caller))
	for _, pkg := range []string{"tsp", "match"} {
		dir := filepath.Join(root, pkg)
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("ReadDir(%s): %v", pkg, err)
		}
		for _, entry := range entries {
			if entry.IsDir() || filepath.Ext(entry.Name()) != ".go" {
				continue
			}
			path := filepath.Join(dir, entry.Name())
			file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
			if err != nil {
				t.Fatalf("parse %s: %v", path, err)
			}
			for _, imported := range file.Imports {
				name, err := strconv.Unquote(imported.Path.Value)
				if err != nil {
					t.Fatalf("bad Go import in %s: %v", path, err)
				}
				if name == "reflect" {
					t.Fatalf("forbidden reflect import: %s", path)
				}
			}
		}
	}
}
