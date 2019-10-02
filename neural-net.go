package main

//figured out log and errors packages are actually pretty useful
//using SGD  and sigmoid equation in this
//SGD to optimize the weights and biases after they are initially randomized
//only works with number values
import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/floats" //matrices and stuff mostly
	"gonum.org/v1/gonum/mat"
)

//has the visible and hidden neurons and some of the architecture
type nNet struct {
	config  nNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

//learning parameters

type nNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int //three layers, input is first, then program figured out hidden ones, then uses those to do output neurons
	learningRate  float64
}

//initializes new neural network
func newNetwork(config nNetConfig) nNet {
	return nNet{config: config}
}

//trains neural network using backpropagation
//made some matrices but could be optimized later idk
func (nn *nNet) train(x, y *mat.Dense) error {
	//new biases and weights
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, paramater := range [][]float64{
		wHiddenRaw, //weight raw
		bHiddenRaw, //bias hidden
		wOutRaw,    //weight raw
		bOutRaw,    //bias raw
	} {
		for i := range paramater {
			paramater[i] = randGen.Float64()
		}
	}

	//define the output of the neural network
	output := new(mat.Dense)

	//use backpropagation to adjust the weights and biases
	if err := nn.backp(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	//define our trained neural network
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

//backpropagate completes the backpropagation methods
func (nn *nNet) backp(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error { //only can return error or nil
	//loop over # of epochs (why more epochs is more accurate)
	for i := 0; i < 2000; i++ {
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		//finish
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		//adjust parameters
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, _ := sumAlongAxis(0, dHiddenLayer)
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}
	return nil
}

//predicts outcome with trained net
func (nn *nNet) predict(x *mat.Dense) (*mat.Dense, error) {
	output := new(mat.Dense)
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

//sigmoid function for later
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

//implements the derivative
//of the sigmoid function for backpropagation
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

//sums a matrix in 1 dimension but keeps the other 1
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()
	var output *mat.Dense

	switch axis { //shorter way to write if else
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m) //copies the elements in the jth column of the matrix into the slice dst
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}
	return output, nil
}

func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)

	//----------------------------- UPDATE IF CHANGED

	reader.FieldsPerRecord = 7 //UPDATE THIS IF CHANGED

	// ---------------------------- UPDATE IF CHANGED

	rawCSVData, err := reader.ReadAll()

	//inputsData and labelsData will hold all the
	//float values that will eventually be
	//used to form matrices.
	inputsData := make([]float64, 4*len(rawCSVData)) //I didn't make this code (makeInputsandLabels), I found it on github and used it bc I couldn't get it to read files
	labelsData := make([]float64, 3*len(rawCSVData))

	//tracks the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	//move rows into slice of floats
	for idex, record := range rawCSVData {
		// Skip the header row.
		if idex == 0 {
			continue
		}
		//loops float columns
		for i, val := range record {

			//convert to float
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			//add to labelsdata if needed
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			//add float value to slice of floats
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)
	return inputs, labels
}

func main() {
	//form training matrices
	inputs, labels := makeInputsAndLabels("C:/test.csv")

	//defines learning parameters
	config := nNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 10, //adjust this
		learningRate:  .3,
	}

	//train the neural network
	network := newNetwork(config)
	network.train(inputs, labels)

	//form the testing matricies
	testInputs, testLabels := makeInputsAndLabels("C:/test.csv")

	//make predictions using trained model
	predictions, _ := network.predict(testInputs)
	//calculate accuracy
	var trueCount int
	numPreds, _ := predictions.Dims() //dimensions of matrix
	for i := 0; i < numPreds; i++ {
		//get label
		labelRow := mat.Row(nil, i, testLabels)
		var prediction int
		for idex, label := range labelRow {
			if label == 1.0 {
				prediction = idex
			}
		}
		//get right/wrong count
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			trueCount++
		}
	}
	accuracy := float64(trueCount) / float64(numPreds)
	//output accuracy
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy) //rounds to 2 places
}
