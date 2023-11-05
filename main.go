package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const (
	N       = 5
	detCalc = 15540.0
)

func solveWithGauss(A [][]float64, b []float64) []float64 {
	x := make([]float64, N)

	for i := 0; i < N; i++ {
		maxRow := i
		for k := i + 1; k < N; k++ {
			if math.Abs(A[k][i]) > math.Abs(A[maxRow][i]) {
				maxRow = k
			}
		}

		if maxRow != i {
			A[i], A[maxRow] = A[maxRow], A[i]
			b[i], b[maxRow] = b[maxRow], b[i]
		}

		for k := i + 1; k < N; k++ {
			factor := A[k][i] / A[i][i]
			b[k] -= factor * b[i]
			for j := i; j < N; j++ {
				A[k][j] -= factor * A[i][j]
			}
		}
	}

	for i := N - 1; i >= 0; i-- {
		x[i] = b[i] / A[i][i]
		for j := i + 1; j < N; j++ {
			x[i] -= A[i][j] * x[j] / A[i][i]
		}
	}

	return x
}

func computeDeterminant(A [][]float64) float64 {
	determinant := 1.0

	for i := 0; i < N; i++ {
		determinant *= A[i][i]
	}

	return determinant
}

func solveWithCholesky(A [][]float64, b []float64) []float64 {
	L := make([][]float64, N)
	U := make([][]float64, N)

	for i := 0; i < N; i++ {
		L[i] = make([]float64, N)
		U[i] = make([]float64, N)
	}

	for i := 0; i < N; i++ {
		for j := 0; j <= i; j++ {
			s := 0.0
			for k := 0; k < j; k++ {
				s += L[i][k] * U[k][j]
			}
			L[i][j] = A[i][j] - s
		}

		for j := i; j < N; j++ {
			s := 0.0
			for k := 0; k < i; k++ {
				s += L[i][k] * U[k][j]
			}
			U[i][j] = (A[i][j] - s) / L[i][i]
		}
	}

	y := make([]float64, N)
	y[0] = b[0] / L[0][0]

	for i := 1; i < N; i++ {
		s := 0.0
		for j := 0; j < i; j++ {
			s += L[i][j] * y[j]
		}
		y[i] = (b[i] - s) / L[i][i]
	}

	x := make([]float64, N)
	x[N-1] = y[N-1] / U[N-1][N-1]

	for i := N - 2; i >= 0; i-- {
		s := 0.0
		for j := i + 1; j < N; j++ {
			s += U[i][j] * x[j]
		}
		x[i] = (y[i] - s) / U[i][i]
	}

	return x
}

func printMatrix(A [][]float64, b []float64) {
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			fmt.Printf("%8.1f ", A[i][j])
		}
		fmt.Printf(" | %8.1f\n", b[i])
	}
}

func printVector(x []float64) {
	for i := 0; i < N; i++ {
		fmt.Printf("| %8.6f |\n", x[i])
	}
}

func computeMaxNorm(A [][]float64, x, b []float64) float64 {
	maxNorm := 0.0

	for i := 0; i < N; i++ {
		sum := 0.0
		for j := 0; j < N; j++ {
			sum += A[i][j] * x[j]
		}
		residual := math.Abs(sum - b[i])
		if residual > maxNorm {
			maxNorm = residual
		}
	}

	return maxNorm
}

func main() {
	rand.Seed(time.Now().UnixNano())

	A := make([][]float64, N)
	for i := range A {
		A[i] = make([]float64, N)
	}

	b := make([]float64, N)

	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			if i == j {
				A[i][j] = 5 * math.Sqrt(float64(i+1))
			} else {
				A[i][j] = math.Sqrt(float64((i + 1) * (j + 1)))
			}
		}
	}

	for i := 0; i < N; i++ {
		b[i] = rand.Float64()
	}

	fmt.Println("Matrix:")
	printMatrix(A, b)

	start := time.Now()
	xGauss := solveWithGauss(A, b)
	fmt.Println("Solve with Gauss:")
	printVector(xGauss)
	fmt.Printf("Work time of Gauss: %s\n", time.Since(start))

	start = time.Now()
	xCholesky := solveWithCholesky(A, b)
	fmt.Println("Solve with Cholesky:")
	printVector(xCholesky)
	fmt.Printf("Work time of Cholesky: %s\n", time.Since(start))

	maxNormGauss := computeMaxNorm(A, xGauss, b)
	fmt.Printf("Max Norm Residual (Gauss): %8.6f\n", maxNormGauss)

	maxNormCholesky := computeMaxNorm(A, xCholesky, b)
	fmt.Printf("Max Norm Residual (Cholesky): %8.6f\n", maxNormCholesky)

	determinantA := computeDeterminant(A)
	fmt.Printf("Det A: %8.6f\n", determinantA)
	fmt.Printf("DeFacto Det A: %8.6f\n", detCalc)
	fmt.Printf("Absolute Error: %8.6f\n", math.Abs(determinantA-detCalc))
	fmt.Printf("Relative Error: %8.6f%%\n", (math.Abs(determinantA-detCalc)/math.Abs(determinantA))*100)
}
