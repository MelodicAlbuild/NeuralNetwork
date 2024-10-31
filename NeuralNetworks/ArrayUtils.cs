namespace NeuralNetworks;

public static class ArrayUtils
{
    public static List<List<double>> ToList(double[,] array)
    {
        var list = new List<List<double>>();
        for (int i = 0; i < array.GetLength(0); i++)
        {
            var row = new List<double>();
            for (int j = 0; j < array.GetLength(1); j++)
            {
                row.Add(array[i, j]);
            }

            list.Add(row);
        }

        return list;
    }

    public static double[,] To2DArray(List<List<double>> list)
    {
        int rows = list.Count;
        int cols = list[0].Count;
        var array = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[i, j] = list[i][j];
            }
        }

        return array;
    }
}