namespace NeuralNetworks;

using System.IO;
using System.Text.Json;

public class ModelParameters
{
    public List<List<double>> InputLayerWeights { get; set; }
    public List<double> InputLayerBiases { get; set; }
    public List<List<double>> HiddenLayerWeights { get; set; }
    public List<double> HiddenLayerBiases { get; set; }
    public List<List<double>> OutputLayerWeights { get; set; }
    public List<double> OutputLayerBiases { get; set; }
}

public class ModelManager
{
    private NeuralNetwork network;

    public ModelManager(NeuralNetwork nn)
    {
        network = nn;
    }

    public ModelManager()
    {
    }

    public void SaveModel(string filePath)
    {
        var parameters = new ModelParameters
        {
            InputLayerWeights = ArrayUtils.ToList(network.GetInputLayer().Weights),
            InputLayerBiases = network.GetInputLayer().Biases.ToList(),
            HiddenLayerWeights = ArrayUtils.ToList(network.GetHiddenLayer().Weights),
            HiddenLayerBiases = network.GetHiddenLayer().Biases.ToList(),
            OutputLayerWeights = ArrayUtils.ToList(network.GetOutputLayer().Weights),
            OutputLayerBiases = network.GetOutputLayer().Biases.ToList()
        };

        string json = JsonSerializer.Serialize(parameters);
        File.WriteAllText(filePath, json);
    }

    public NeuralNetwork LoadModel(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException("Model file not found.");

        string json = File.ReadAllText(filePath);
        var parameters = JsonSerializer.Deserialize<ModelParameters>(json);

        NeuralNetwork nn = new NeuralNetwork(
            new NeuronLayer(ArrayUtils.To2DArray(parameters.InputLayerWeights), parameters.InputLayerBiases.ToArray()),
            new NeuronLayer(ArrayUtils.To2DArray(parameters.HiddenLayerWeights),
                parameters.HiddenLayerBiases.ToArray()),
            new NeuronLayer(ArrayUtils.To2DArray(parameters.OutputLayerWeights),
                parameters.OutputLayerBiases.ToArray()), this);

        network = nn;

        return nn;
    }
}