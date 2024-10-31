namespace NeuralNetworks;

public class NeuronLayer
{
    public int NumNeurons { get; }
    public double[,] Weights { get; set; }
    public double[] Biases { get; set; }

    public NeuronLayer(int numNeurons, int numInputs)
    {
        NumNeurons = numNeurons;
        Weights = new double[numNeurons, numInputs];
        Biases = new double[numNeurons];
        InitializeWeights();
    }

    public NeuronLayer(double[,] newWeights, double[] newBiases)
    {
        Weights = newWeights;
        Biases = newBiases;
    }

    private void InitializeWeights()
    {
        Random rand = new Random();
        for (int i = 0; i < NumNeurons; i++)
        {
            for (int j = 0; j < Weights.GetLength(1); j++)
            {
                Weights[i, j] = rand.NextDouble() - 0.5;
            }

            Biases[i] = rand.NextDouble() - 0.5;
        }
    }
}