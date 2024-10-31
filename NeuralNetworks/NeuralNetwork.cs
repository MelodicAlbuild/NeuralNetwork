namespace NeuralNetworks;

public class NeuralNetwork
{
    private ModelManager modelManager;

    private NeuronLayer inputLayer;
    private NeuronLayer hiddenLayer;
    private NeuronLayer outputLayer;

    private double[] hiddenOutputs;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    {
        inputLayer = new NeuronLayer(inputSize, inputSize);
        hiddenLayer = new NeuronLayer(hiddenSize, inputSize);
        outputLayer = new NeuronLayer(outputSize, hiddenSize);

        hiddenOutputs = new double[hiddenSize];

        modelManager = new ModelManager(this);
    }

    public NeuralNetwork(NeuronLayer inputLayer, NeuronLayer hiddenLayer, NeuronLayer outputLayer, ModelManager mm)
    {
        this.inputLayer = inputLayer;
        this.hiddenLayer = hiddenLayer;
        this.outputLayer = outputLayer;

        hiddenOutputs = new double[hiddenLayer.NumNeurons];

        modelManager = mm;
    }

    public NeuronLayer GetInputLayer()
    {
        return inputLayer;
    }

    public NeuronLayer GetHiddenLayer()
    {
        return hiddenLayer;
    }

    public NeuronLayer GetOutputLayer()
    {
        return outputLayer;
    }

    private double ReLU(double x)
    {
        return Math.Max(0, x);
    }

    private double ReLUDerivative(double x)
    {
        return x > 0 ? 1 : 0;
    }

    private double[] ForwardPropagate(double[] inputs)
    {
        double[] hiddenOutputs = new double[hiddenLayer.NumNeurons];
        for (int i = 0; i < hiddenLayer.NumNeurons; i++)
        {
            double sum = 0;
            for (int j = 0; j < inputs.Length; j++)
            {
                sum += inputs[j] * hiddenLayer.Weights[i, j];
            }

            sum += hiddenLayer.Biases[i];
            hiddenOutputs[i] = ReLU(sum);
        }

        double[] outputValues = new double[outputLayer.NumNeurons];
        for (int i = 0; i < outputLayer.NumNeurons; i++)
        {
            double sum = 0;
            for (int j = 0; j < hiddenOutputs.Length; j++)
            {
                sum += hiddenOutputs[j] * outputLayer.Weights[i, j];
            }

            sum += outputLayer.Biases[i];
            outputValues[i] = ReLU(sum);
        }

        return outputValues;
    }

    public void Train(double[][] inputs, double[][] targets, int epochs, double learningRate, string modelFilePath,
        int checkpointInterval = 5)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                // Forward pass
                double[] output = ForwardPropagate(inputs[i]);

                // Calculate output layer error
                double[] outputErrors = new double[outputLayer.NumNeurons];
                for (int j = 0; j < outputLayer.NumNeurons; j++)
                {
                    outputErrors[j] = targets[i][j] - output[j];
                }

                // Back Propagate errors to hidden layer with ReLU derivative
                double[] hiddenErrors = new double[hiddenLayer.NumNeurons];
                for (int j = 0; j < hiddenLayer.NumNeurons; j++)
                {
                    double error = 0;
                    for (int k = 0; k < outputLayer.NumNeurons; k++)
                    {
                        error += outputErrors[k] * outputLayer.Weights[k, j];
                    }

                    hiddenErrors[j] = error * ReLUDerivative(hiddenOutputs[j]);
                }

                // Update weights and biases for output layer
                for (int j = 0; j < outputLayer.NumNeurons; j++)
                {
                    for (int k = 0; k < hiddenLayer.NumNeurons; k++)
                    {
                        outputLayer.Weights[j, k] += learningRate * outputErrors[j] * hiddenOutputs[k];
                    }

                    outputLayer.Biases[j] += learningRate * outputErrors[j];
                }

                // Update weights and biases for hidden layer
                for (int j = 0; j < hiddenLayer.NumNeurons; j++)
                {
                    for (int k = 0; k < inputs[i].Length; k++)
                    {
                        hiddenLayer.Weights[j, k] += learningRate * hiddenErrors[j] * inputs[i][k];
                    }

                    hiddenLayer.Biases[j] += learningRate * hiddenErrors[j];
                }
            }

            // Save model checkpoint after specified number of epochs
            if ((epoch + 1) % checkpointInterval == 0)
            {
                modelManager.SaveModel(modelFilePath);
                Console.WriteLine($"Checkpoint saved at epoch {epoch + 1}");
            }
        }
    }
}