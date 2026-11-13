// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package hmm_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

func BenchmarkFilterCategorical(b *testing.B) {
	for _, k := range []int{2, 8, 32} {
		for _, length := range []int{100, 100000} {
			for _, history := range []bool{false, true} {
				b.Run(fmt.Sprintf("K%d_T%d_history%t", k, length, history), func(b *testing.B) {
					transition := make([][]float64, k)
					emission := make([][]float64, k)
					prior := make([]float64, k)
					for i := 0; i < k; i++ {
						prior[i] = 1 / float64(k)
						transition[i] = make([]float64, k)
						for j := 0; j < k; j++ {
							transition[i][j] = 1 / float64(k)
						}
						emission[i] = []float64{.7, .3}
					}
					m, e := hmm.NewCategorical(hmm.CategoricalSpec{Common: hmm.Common{Prior: prior, Transition: dense(b, transition)}, Emission: dense(b, emission)})
					if e != nil {
						b.Fatal(e)
					}
					observations := make([]uint32, length)
					ctx := context.Background()
					opts := hmm.FilterOptions{ReturnHistory: history}
					b.ReportAllocs()
					b.ResetTimer()
					for run := 0; run < b.N; run++ {
						if _, e = m.Filter(ctx, observations, opts); e != nil {
							b.Fatal(e)
						}
					}
				})
			}
		}
	}
}

func BenchmarkFilterGaussian(b *testing.B) {
	for _, k := range []int{2, 8, 32} {
		for _, length := range []int{100, 100000} {
			for _, dimension := range []int{1, 16} {
				for _, history := range []bool{false, true} {
					b.Run(fmt.Sprintf("K%d_T%d_D%d_history%t", k, length, dimension, history), func(b *testing.B) {
						prior := make([]float64, k)
						trans := make([][]float64, k)
						means := make([][]float64, k)
						vars := make([][]float64, k)
						for i := 0; i < k; i++ {
							prior[i] = 1 / float64(k)
							trans[i] = make([]float64, k)
							for j := 0; j < k; j++ {
								trans[i][j] = 1 / float64(k)
							}
							means[i] = make([]float64, dimension)
							vars[i] = make([]float64, dimension)
							for d := range vars[i] {
								vars[i][d] = 1
							}
						}
						model, err := hmm.NewGaussian(hmm.GaussianSpec{Common: hmm.Common{Prior: prior, Transition: dense(b, trans)}, Means: dense(b, means), Variances: dense(b, vars)})
						if err != nil {
							b.Fatal(err)
						}
						samples, err := matrix.NewZeros(length, dimension)
						if err != nil {
							b.Fatal(err)
						}
						ctx := context.Background()
						opts := hmm.FilterOptions{ReturnHistory: history}
						b.ReportAllocs()
						b.ResetTimer()
						for run := 0; run < b.N; run++ {
							if _, err = model.Filter(ctx, samples, opts); err != nil {
								b.Fatal(err)
							}
						}
					})
				}
			}
		}
	}
}
