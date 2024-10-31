using NeuralNetworks;

double[][] inputs = new double[200][];
double[][] targets = new double[200][];

int n = 100;
double noise = 0.2;
Random random = new Random();

for (int i = 0; i < n; i++)
{
    double r = i / (double)n;
    double theta = r * 4 * Math.PI;

    // Class 0
    inputs[i] = new double[]
    {
        r * Math.Sin(theta) + noise * (random.NextDouble() - 0.5),
        r * Math.Cos(theta) + noise * (random.NextDouble() - 0.5)
    };
    targets[i] = new double[] { 0 }; // Spiral class 0

    // Class 1
    inputs[i + n] = new double[]
    {
        r * Math.Sin(theta + Math.PI) + noise * (random.NextDouble() - 0.5),
        r * Math.Cos(theta + Math.PI) + noise * (random.NextDouble() - 0.5)
    };
    targets[i + n] = new double[] { 1 }; // Spiral class 1
}

string modelFilePath = "model_checkpoint.json";
var nn = new NeuralNetwork(2, 2, 1);

if (File.Exists(modelFilePath))
{
    ModelManager modelManager = new ModelManager();
    nn = modelManager.LoadModel(modelFilePath);
    Console.WriteLine("Model loaded from checkpoint.");
}
else
{
    Console.WriteLine("Starting new training session.");
}

nn.Train(inputs, targets, epochs: 1000000000, learningRate: 0.1, modelFilePath: modelFilePath, checkpointInterval: 500);