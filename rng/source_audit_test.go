// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng_test checks prohibited production imports and forbidden direct calls.
// AST inspection avoids textual false positives in comments and documentation.
package rng_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestProductionSource_NoGlobalRandomReflectOrPanic locks the no-implicit-entropy
// rule against accidental future refactors of the production package.
func TestProductionSource_NoGlobalRandomReflectOrPanic(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(".", name), nil, 0)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for _, imported := range file.Imports {
			path, err := strconv.Unquote(imported.Path.Value)
			if err != nil {
				t.Fatalf("invalid import in %s: %v", name, err)
			}
			if path == "reflect" || path == "time" || path == "math/rand" || path == "math/rand/v2" || path == "crypto/rand" || path == "sync" {
				t.Errorf("forbidden production dependency %q in %s", path, name)
			}
		}
		ast.Inspect(file, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			ident, ok := call.Fun.(*ast.Ident)
			if ok && (ident.Name == "panic" || ident.Name == "recover") {
				t.Errorf("forbidden call %s in %s", ident.Name, name)
			}
			return true
		})
	}
}
